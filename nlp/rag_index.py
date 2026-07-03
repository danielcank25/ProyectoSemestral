"""Índice semántico FAISS + SBERT sobre las bases técnicas/administrativas
y los itemizados de cada licitación en data/.

Permite buscar texto libre y, para cada ítem del itemizado, deja
precalculadas las cláusulas de bases técnicas más relacionadas
semánticamente (mismo licitación).
"""
import json
import os
import re
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer

CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
RELATED_TOP_K = 3

LICITACION_RE = re.compile(
    r"(\d{3,4}-\d{1,3}-LP\d{2})"   # ej. 2788-68-LP25
    r"|([A-Z]\d{2}[A-Z]\d{3})",     # ej. A22M451, T21C404
    re.IGNORECASE,
)

_UNIT_WORDS = [
    # básicos
    "glb", "und", "gl", "un", "ml", "m2", "m3", "kg", "ton",
    "hrs", "hr", "lt", "dia", "m", "lm", "uni", "mes",
    # mano de obra / maquinaria
    "hh", "hm", "hme", "hmo",
    # área / volumen / distancia
    "ha", "km", "cm", "mm", "m3.", "m2.",
    # piezas / conjuntos
    "pza", "pzas", "jgo", "eq", "pt",
    # transporte / tiempo
    "vje", "sem", "qna",
    # construcción chilena
    "cu", "cbm", "mts", "ml.", "kw", "kwh", "hp", "lt.",
    # adicionales
    "gl.", "un.", "dia.", "mes.", "tpo", "pulg",
]
# Excluir palabras vacías que podrían colisionar
#_UNIT_EXCLUDE = {
#    "no", "si", "ok", "total", "item", "unidad", "precio", "nombre",
#    "desc", "cant", "neto", "iva", "sub", "ref", "obs", "nota",
#    "a", "b", "c", "d", "e", "i", "ii", "iii",
#}

#Esta generando problemas al borrar algunos items
_UNIT_EXCLUDE = {None}

_UNIT_SET = {u.lower().rstrip(".") for u in _UNIT_WORDS} - _UNIT_EXCLUDE
_UNIT_ALT = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))

ITEMIZADO_LINE_RE = re.compile(
    r"^(?P<item>\d+(?:\.\d+)*)\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<unidad>" + _UNIT_ALT + r")\s+"
    r"(?P<cantidad>[\d.,]+)\s+"
    r"(?P<precio>[\d.,]+)\s+"
    r"(?P<total>[\d.,]+)$",
    re.IGNORECASE,
)


def licitacion_id(filename: str) -> str:
    m = LICITACION_RE.search(filename)
    if not m:
        return "desconocida"
    # grupo 1 → patrón LP (ej. 2788-68-LP25), grupo 2 → patrón alfanumérico (ej. A22M451)
    return (m.group(1) or m.group(2)).upper()


def classify_doc_type(filename: str) -> str:
    """Clasifica el tipo de documento según el nombre de archivo.

    Estructura esperada:
      · Itemizado  : {contrato}_{empresa}_{ECO##}_{YYYY-MM-DD}_ADJUDICADO.xlsx
      · Bases tec. : {contrato}_{EETT|BBTT|ESPECIFICACIONES}_{YYYY-MM}.pdf
    """
    name = filename.lower()
    # Itemizado: ECO## (formulario económico)
    if "itemizado" in name or "itimizado" in name or re.search(r"eco\d+", name):
        return "itemizado"
    # Bases técnicas: EETT (Especificaciones Técnicas), BBTT, o variantes textuales
    if (
        "eett" in name
        or "bbtt" in name
        or "especificaciones" in name
        or "tecnica" in name
        or "técnica" in name
    ):
        return "bases_tecnicas"
    if "bases" in name or "administrativas" in name:
        return "bases_administrativas"
    return "otro"


def _chunk_words(text: str, source: dict):
    words = text.split()
    if not words:
        return
    step = CHUNK_WORDS - CHUNK_OVERLAP
    start = 0
    while start < len(words):
        piece = words[start:start + CHUNK_WORDS]
        yield {**source, "text": " ".join(piece)}
        if start + CHUNK_WORDS >= len(words):
            break
        start += step


def extract_pdf_chunks(path: str, licitacion: str, doc_type: str):
    """Trozos de texto de un PDF con capa de texto. Devuelve (chunks, warnings)."""
    chunks = []
    warnings = []
    filename = os.path.basename(path)
    with pdfplumber.open(path) as pdf:
        any_text = False
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                any_text = True
            source = {
                "licitacion": licitacion,
                "doc_type": doc_type,
                "file": filename,
                "page": page_num,
            }
            chunks.extend(_chunk_words(text, source))
        if not any_text:
            warnings.append(
                f"{filename}: sin texto extraíble (PDF escaneado, requiere OCR) — omitido"
            )
    return chunks, warnings


def _itemizado_text(item, desc, unidad, cantidad, precio, total) -> str:
    return (f"Ítem {item}: {desc}— Unidad: {unidad}, "
        f"Cantidad: {cantidad}, Precio unitario: {precio}, Total: {total}"
    )


def extract_itemizado_pdf_chunks(path: str, licitacion: str):
    chunks = []
    filename = os.path.basename(path)
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                m = ITEMIZADO_LINE_RE.match(line.strip())
                if not m:
                    continue
                item, desc, unidad, cantidad, precio, total = (
                    m.group("item"), m.group("desc").strip(), m.group("unidad"),
                    m.group("cantidad"), m.group("precio"), m.group("total"),
                )
                chunks.append({
                    "licitacion": licitacion,
                    "doc_type": "itemizado",
                    "file": filename,
                    "page": page_num,
                    "item": item,
                    "descripcion": desc,
                    "unidad": unidad,
                    "cantidad": cantidad,
                    "precio_unitario": precio,
                    "total": total,
                    "text": _itemizado_text(item, desc, unidad, cantidad, precio, total),
                })
    return chunks


def _cell_str(row, col_idx) -> str:
    """Extrae el valor de una columna como string limpio, o '' si es nulo."""
    if col_idx is None:
        return ""
    val = row.iloc[col_idx]
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def _parse_num(val) -> str:
    """Convierte un valor numérico —incluyendo texto con separadores de miles chilenos
    (1.069.630,06) o americanos (1,069,630.06)— a string de float limpio.
    Maneja correctamente valores menores a 1.000 sin separador de miles.
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return ""
    if isinstance(val, (int, float)):
        return "" if pd.isna(val) else str(float(val))

    s = re.sub(r"[$\s]", "", str(val).strip())
    if not s or s.lower() == "nan":
        return ""

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # 1.069.630,06 → 1069630.06
        else:
            s = s.replace(",", "")                     # 1,069,630.06 → 1069630.06
    elif has_comma:
        if s.count(",") > 1:
            s = s.replace(",", "")                     # 1,069,630 → 1069630
        else:
            s = s.replace(",", ".")                    # 630,06 → 630.06
    elif has_dot and s.count(".") > 1:
        s = s.replace(".", "")                         # 1.069.630 → 1069630

    try:
        return str(float(s))
    except ValueError:
        return ""


def _cell_num(row, col_idx) -> str:
    """Extrae un valor numérico de una columna como string de float limpio."""
    if col_idx is None:
        return ""
    return _parse_num(row.iloc[col_idx])


def _detect_eco_columns(raw: pd.DataFrame):
    """Encuentra la primera fila con unidad+valores y deduce posiciones de columnas.

    Estrategia: todo lo que está ANTES de la columna de unidad es ítem/descripción;
    todo lo que está DESPUÉS son cantidad, precio unitario y total — en ese orden.
    Retorna (anchor_row_index, col_map) o (None, {}).
    """
    for i, row in raw.iterrows():
        non_null = [
            (col, val) for col, val in enumerate(row)
            if pd.notna(val) and str(val).strip() not in ("", "nan")
        ]

        # 1. Buscar columna de unidad — primero en _UNIT_SET, fallback estructural
        unit_hits = [
            (col, val) for col, val in non_null
            if str(val).strip().lower().rstrip(".") in _UNIT_SET
        ]
        if not unit_hits:
            # Fallback: token alfabético corto (1-6 chars) entre descripción y valores numéricos,
            # excluyendo palabras comunes no-unidad
            unit_hits = [
                (col, val) for col, val in non_null
                if (isinstance(val, str)
                    and 1 <= len(str(val).strip()) <= 6
                    and re.match(r'^[a-záéíóúüñ./]+$', str(val).strip(), re.I)
                    and str(val).strip().lower() not in _UNIT_EXCLUDE)
            ]
        if not unit_hits:
            continue
        unit_col = unit_hits[0][0]

        # 2. Todo antes de unit_col → ítem y descripción
        before = [(col, val) for col, val in non_null if col < unit_col]

        print(before)
        
        if len(before) < 2:
            continue
        item_col = before[0][0]   # columna más a la izquierda
        # descripción: la celda de texto más a la derecha antes de la unidad
        desc_candidates = [
            (col, val) for col, val in before
            if isinstance(val, str) and len(str(val).strip()) > 3
        ]
        if not desc_candidates:
            continue
        desc_col = desc_candidates[-1][0]

        # 3. Todo después de unit_col → celdas que representen un número válido
        # Acepta tanto float/int nativos como strings con formato de miles (ej. "1.069.630,06")
        numeric_after = sorted(
            [
                (col, val) for col, val in non_null
                if col > unit_col and _parse_num(val) != ""
            ],
            key=lambda x: x[0],
        )
        if len(numeric_after) < 1:
            continue

        qty_col   = numeric_after[0][0]
        precio_col = numeric_after[1][0]
        total_col  = numeric_after[-1][0]  # último numérico

        return i, {
            "item":       item_col,
            "descripcion": desc_col,
            "unidad":     unit_col,
            "cantidad":   qty_col,
            "precio":     precio_col,
            "total":      total_col,
        }

    return None, {}


def _build_breadcrumb(item_num: str, section_map: dict) -> list:
    """Devuelve la lista de descripciones de los nodos padre de un ítem.

    Para '2.1.4.1.4' busca '2', '2.1', '2.1.4', '2.1.4.1' en section_map.
    """
    clean = str(item_num).rstrip(".")
    parts = clean.split(".")
    result = []
    for i in range(1, len(parts)):
        ancestor = ".".join(parts[:i])
        if ancestor in section_map:
            result.append(section_map[ancestor])
    return result


def extract_itemizado_excel_chunks(path: str, licitacion: str):
    filename = os.path.basename(path)
    raw = pd.read_excel(path, engine="openpyxl", header=None)
    


    anchor, col_map = _detect_eco_columns(raw)
    if not col_map:
        return []

    # Pasada 1: recopilar encabezados de sección (filas sin unidad)
    # y guardar filas hoja para la pasada 2.
    section_map: dict = {}   # item_num (limpio) → descripción
    leaf_rows: list = []

    for _, row in raw.iloc[anchor:].iterrows():
        item   = _cell_str(row, col_map["item"])
        desc   = _cell_str(row, col_map["descripcion"])
        unidad = _cell_str(row, col_map["unidad"])
        if not item or not desc:
            continue
        if not unidad:
            # Fila de sección (sin unidad): guarda en el mapa jerárquico
            section_map[item.rstrip(".")] = desc
        else:
            leaf_rows.append((
                item,
                desc,
                unidad,
                _cell_num(row, col_map["cantidad"]),
                _cell_num(row, col_map["precio"]),
                _cell_num(row, col_map["total"]),
            ))

    # Pasada 2: construir chunks con contexto jerárquico completo
    chunks = []
    for item, desc, unidad, cantidad, precio, total in leaf_rows:


        breadcrumb = _build_breadcrumb(item, section_map)
        # El texto incluye la ruta completa para mejorar la búsqueda semántica
        if breadcrumb:
            path_str = " > ".join(breadcrumb) + " > " + desc
        else:
            path_str = desc

        text = (
            f"Ítem {item}: {path_str} — "
            f"Unidad: {unidad}, Cantidad: {cantidad}, "
            f"Precio unitario: {precio}, Total: {total}"
        )

        chunks.append({
            "licitacion": licitacion,
            "doc_type": "itemizado",
            "file": filename,
            "page": None,
            "item": item,
            "descripcion": desc,
            "breadcrumb": " > ".join(breadcrumb),
            "unidad": unidad,
            "cantidad": cantidad,
            "precio_unitario": precio,
            "total": total,
            "text": text,
        })
    return chunks


def extract_itemizado_excel_rows(path: str, licitacion: str) -> list:
    """Como extract_itemizado_excel_chunks pero devuelve TODAS las filas del Excel
    (secciones + ítems hoja) con campo 'tipo': 'titulo' | 'item'.
    Usado para renderizar la estructura completa en el frontend de valorización.
    """
    filename = os.path.basename(path)
    raw = pd.read_excel(path, engine="openpyxl", header=None)

    print(raw)

    anchor, col_map = _detect_eco_columns(raw)
    if not col_map:
        return []

    # Pasada 1: recopilar todas las filas en orden y construir section_map
    section_map: dict = {}
    raw_rows: list = []

    for _, row in raw.iloc[anchor:].iterrows():
        #revision para la fila
        #print(row)
        item   = _cell_str(row, col_map["item"])
        desc   = _cell_str(row, col_map["descripcion"])
        unidad = _cell_str(row, col_map["unidad"])
        if not item or not desc:
            continue
        if not unidad:
            section_map[item.rstrip(".")] = desc
            raw_rows.append(("titulo", item, desc, "", "", "", ""))
        else:
            raw_rows.append((
                "item", item, desc, unidad,
                _cell_num(row, col_map["cantidad"]),
                _cell_num(row, col_map["precio"]),
                _cell_num(row, col_map["total"]),
            ))
    #print(raw_rows)
    # Pasada 2: construir resultado final con breadcrumb en ítems hoja
    result: list = []
    for tipo, item, desc, unidad, cantidad, precio, total in raw_rows:
        if tipo == "titulo":
            result.append({"tipo": "titulo", "item": item, "descripcion": desc})
        else:
            breadcrumb = _build_breadcrumb(item, section_map)
            path_str = (" > ".join(breadcrumb) + " > " + desc) if breadcrumb else desc
            text = (
                f"Ítem {item}: {path_str} — "
                f"Unidad: {unidad}, Cantidad: {cantidad}, "
                f"Precio unitario: {precio}, Total: {total}"
            )
            result.append({
                "tipo": "item",
                "licitacion": licitacion,
                "file": filename,
                "item": item,
                "descripcion": desc,
                "breadcrumb": " > ".join(breadcrumb),
                "unidad": unidad,
                "cantidad": cantidad,
                "precio_unitario": precio,
                "total": total,
                "text": text,
            })
    return result


def collect_documents(data_dir: str):
    """Recorre data_dir y arma todos los chunks indexables. Devuelve (chunks, warnings)."""
    all_chunks = []
    warnings = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        lic = licitacion_id(filename)
        doc_type = classify_doc_type(filename)

        if doc_type == "itemizado" and ext == ".pdf":
            all_chunks.extend(extract_itemizado_pdf_chunks(path, lic))
        elif doc_type == "itemizado" and ext == ".xlsx":
            all_chunks.extend(extract_itemizado_excel_chunks(path, lic))
        elif ext == ".pdf":
            chunks, warns = extract_pdf_chunks(path, lic, doc_type)
            all_chunks.extend(chunks)
            warnings.extend(warns)
    return all_chunks, warnings


class RagIndex:
    """Índice FAISS (cosine, vía IndexFlatIP + normalización) sobre embeddings SBERT."""

    def __init__(self, index_dir: str, model_name: str = "all-MiniLM-L6-v2",
                 embedder: Optional[SentenceTransformer] = None):
        self.index_dir = index_dir
        self.model_name = model_name
        self._embedder = embedder
        self.index = None
        self.metadata: list = []
        self.warnings: list = []

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _paths(self):
        return (
            os.path.join(self.index_dir, "faiss.index"),
            os.path.join(self.index_dir, "metadata.json"),
        )

    def is_built(self) -> bool:
        idx_path, meta_path = self._paths()
        return os.path.exists(idx_path) and os.path.exists(meta_path)

    def build(self, data_dir: str):
        chunks, warnings = collect_documents(data_dir)
        if not chunks:
            raise ValueError(f"No se encontró texto indexable en {data_dir}")

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.metadata = chunks
        self.warnings = warnings
        self._compute_related_clauses(embeddings)
        self._save()
        return {"chunks": len(chunks), "warnings": warnings}

    def _compute_related_clauses(self, embeddings: np.ndarray):
        bt_positions = [i for i, c in enumerate(self.metadata) if c["doc_type"] == "bases_tecnicas"]
        if not bt_positions:
            return
        bt_embeddings = embeddings[bt_positions]
        bt_index = faiss.IndexFlatIP(bt_embeddings.shape[1])
        bt_index.add(bt_embeddings)

        fetch_k = min(len(bt_positions), RELATED_TOP_K + 10)
        for i, chunk in enumerate(self.metadata):
            if chunk["doc_type"] != "itemizado":
                continue
            scores, ids = bt_index.search(embeddings[i:i + 1], fetch_k)
            related = []
            for score, local_id in zip(scores[0], ids[0]):
                if local_id < 0:
                    continue
                cand = self.metadata[bt_positions[local_id]]
                if cand["licitacion"] != chunk["licitacion"]:
                    continue
                related.append({
                    "text": cand["text"],
                    "file": cand["file"],
                    "page": cand["page"],
                    "score": float(score),
                })
                if len(related) >= RELATED_TOP_K:
                    break
            chunk["related_bases_tecnicas"] = related

    def _save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        idx_path, meta_path = self._paths()
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"chunks": self.metadata, "warnings": self.warnings},
                fh, ensure_ascii=False, indent=2,
            )

    def load(self) -> bool:
        idx_path, meta_path = self._paths()
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return False
        loaded_index = faiss.read_index(idx_path)
        # Valida que la dimensión del índice coincida con el embedder actual
        expected_dim = self.embedder.get_embedding_dimension()
        if loaded_index.d != expected_dim:
            print(
                f"RAG: dimensión del índice en disco ({loaded_index.d}d) "
                f"no coincide con el embedder ({expected_dim}d) — reconstruyendo."
            )
            return False
        self.index = loaded_index
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.metadata = data["chunks"]
        self.warnings = data.get("warnings", [])
        return True

    def licitaciones(self):
        return sorted({c["licitacion"] for c in self.metadata})

    def search(self, query: str, top_k: int = 8, doc_type: Optional[str] = None,
               licitacion: Optional[str] = None):
        if not query or self.index is None or not self.metadata:
            return []
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")

        fetch_k = top_k if not (doc_type or licitacion) else min(len(self.metadata), top_k * 10 + 20)
        scores, ids = self.index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.metadata[idx]
            if doc_type and chunk["doc_type"] != doc_type:
                continue
            if licitacion and chunk["licitacion"] != licitacion:
                continue
            hit = dict(chunk)
            hit["score"] = float(score)

            results.append(hit)
            if len(results) >= top_k:
                break
        return results



# ── Índice temporal de BBTT subida por el usuario ───────────────────────────
# Se construye en memoria por cada petición /valorizar y es accesible
# globalmente dentro del mismo proceso (Flask no es multihilo aquí).

_temp_pdf_index = None          # faiss.IndexFlatIP | None
_temp_pdf_texts: list[str] = [] # fragmentos de texto alineados con el índice
_temp_pdf_embedder = None       # SentenceTransformer reutilizado del RagIndex principal


def build_pdf_index(rag_instance: RagIndex, pdf_text: str) -> int:
    """Construye un índice FAISS temporal con el texto de un PDF subido por el usuario.

    Parámetros
    ----------
    rag_instance : RagIndex
        Índice principal; se reutiliza su embedder para no cargar el modelo dos veces.
    pdf_text : str
        Texto completo extraído del PDF (ya procesado con pdfplumber).

    Retorna el número de fragmentos indexados.
    """
    global _temp_pdf_index, _temp_pdf_texts, _temp_pdf_embedder

    _temp_pdf_embedder = rag_instance.embedder

    # Trocear el texto completo con el mismo tamaño de ventana que el índice principal
    texts = [c["text"] for c in _chunk_words(pdf_text, {})]
    if not texts:
        _temp_pdf_index = None
        _temp_pdf_texts = []
        return 0

    embeddings = _temp_pdf_embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    _temp_pdf_index = idx
    _temp_pdf_texts = texts
    return len(texts)


def search_pdf(query: str, top_k: int = 3) -> list[str]:
    """Busca en el índice temporal de BBTT subido por el usuario.

    Retorna una lista de hasta *top_k* fragmentos de texto relevantes.
    Si el índice no fue construido aún, retorna lista vacía.
    """
    if _temp_pdf_index is None or not _temp_pdf_texts or _temp_pdf_embedder is None:
        return []

    q_emb = _temp_pdf_embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    fetch_k = min(top_k, len(_temp_pdf_texts))
    scores, ids = _temp_pdf_index.search(q_emb, fetch_k)
    return [_temp_pdf_texts[int(i)] for i in ids[0] if int(i) >= 0]

