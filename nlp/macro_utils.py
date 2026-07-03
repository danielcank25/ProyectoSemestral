"""macro_utils.py — Utilidades macroeconómicas para ajuste de precios por IPC.

Cambios respecto a la versión anterior:
  · extraer_fecha_de_archivo()    nueva — infiere año/mes del nombre del archivo
  · inferir_ipc_hit()             nueva — IPC histórico de un hit RAG por su archivo fuente
  · normalizar_precio_ipc()       nueva — deflacta/infla precio a valores actuales
  · extraer_fecha_y_ipc()         mejorada — patrones EETT/ECO de minería chilena
"""

from __future__ import annotations

import re
import pandas as pd
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de mapeo de meses
# ──────────────────────────────────────────────────────────────────────────────

_MES_MAP: dict[str, int] = {
    # nombres completos
    'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,
    'julio':7,'agosto':8,'septiembre':9,'octubre':10,'noviembre':11,'diciembre':12,
    # abreviaturas 3 letras (castellano)
    'ene':1,'feb':2,'mar':3,'abr':4,'may':5,'jun':6,
    'jul':7,'ago':8,'sep':9,'oct':10,'nov':11,'dic':12,
    # abreviaturas alternativas
    'sept':9,
}

# ──────────────────────────────────────────────────────────────────────────────
# Carga del CSV del IPC (Banco Central de Chile)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_ipc(ruta_csv: str) -> dict:
    """Lee el CSV del Banco Central y devuelve {(año, mes): ipc_acumulado}."""
    try:
        df = pd.read_csv(ruta_csv, sep=";", skiprows=2)
        df.columns = ['Periodo', 'IPC_var', 'IPCX', 'IPCX1', 'IPC_SAE']

        def parse_period(p):
            if pd.isna(p):
                return None, None
            partes = str(p).split('.')
            if len(partes) != 2:
                return None, None
            mes_str, year_str = partes
            return int(year_str), _MES_MAP.get(mes_str.lower().strip())

        df = df.dropna(subset=['Periodo'])
        df[['Year', 'Month']] = df['Periodo'].apply(
            lambda x: pd.Series(parse_period(x))
        )
        df = df.dropna(subset=['Year', 'Month'])
        df['IPC_var'] = pd.to_numeric(
            df['IPC_var'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        df['IPC_acumulado'] = 100 * (1 + df['IPC_var'] / 100).cumprod()

        ipc_dict: dict[tuple[int,int], float] = {}
        for _, row in df.iterrows():
            try:
                ipc_dict[(int(row['Year']), int(row['Month']))] = float(row['IPC_acumulado'])
            except (ValueError, TypeError):
                pass

        return dict(sorted(ipc_dict.items()))
    except Exception as e:
        print(f"Error cargando IPC: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Normalización de precios por IPC
# ──────────────────────────────────────────────────────────────────────────────

def normalizar_precio_ipc(
    precio_nominal: float,
    ipc_hist: float,
    ipc_actual: float,
) -> float:
    """Ajusta un precio histórico a valores del período actual.

    Fórmula: precio_real = precio_nominal × (ipc_actual / ipc_hist)

    Si ipc_hist es 0 o None no se aplica ajuste (devuelve precio_nominal).
    """
    if not ipc_hist or ipc_hist <= 0 or not ipc_actual or ipc_actual <= 0:
        return precio_nominal
    return precio_nominal * (ipc_actual / ipc_hist)


# ──────────────────────────────────────────────────────────────────────────────
# Extracción de fecha de un nombre de archivo
# ──────────────────────────────────────────────────────────────────────────────

def extraer_fecha_de_archivo(filename: str) -> Optional[tuple[int, int]]:
    """Intenta extraer (año, mes) del nombre de archivo.

    Reconoce los formatos más comunes en documentos de licitaciones de minería:
      · YYYY-MM-DD  / YYYY_MM_DD           ej. ECO01_2026-02-11
      · ECO/BBTT_YYYYMM                    ej. ECO01_202602
      · REV_MMMYYYY / MMMYYYYrev           ej. ENE2024, ene2024
      · _YYYY_                             ej. EETT_2024_REV1 (usa julio como mes)
    """
    nombre = str(filename)

    # 1. YYYY-MM-DD o YYYY_MM_DD completo  (ej. ECO01_2026-02-11)
    m = re.search(r'(\d{4})[-_](\d{2})[-_]\d{2}', nombre)
    if m:
        anio, mes = int(m.group(1)), int(m.group(2))
        if 2000 <= anio <= 2050 and 1 <= mes <= 12:
            return anio, mes

    # 1b. YYYY-MM sin día  (ej. EETT_2024-03, BBTT_2025-01)
    m = re.search(r'(20\d{2})[-_](0[1-9]|1[0-2])(?![-_\d])', nombre)
    if m:
        anio, mes = int(m.group(1)), int(m.group(2))
        if 2000 <= anio <= 2050 and 1 <= mes <= 12:
            return anio, mes

    # 2. YYYYMM compacto sin separador  (ej. ECO01_202602)
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])(?!\d)', nombre)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 3. Abreviatura de mes + año: ENE2024, ene2024, Ene2024
    patron_mes_abrev = '|'.join(sorted(_MES_MAP.keys(), key=len, reverse=True))
    m = re.search(rf'({patron_mes_abrev})(20\d{{2}})', nombre, re.IGNORECASE)
    if m:
        mes = _MES_MAP.get(m.group(1).lower())
        if mes:
            return int(m.group(2)), mes

    # 4. Solo año (4 dígitos) — asume julio como mes central
    m = re.search(r'(20\d{2})', nombre)
    if m:
        return int(m.group(1)), 7

    return None


# ──────────────────────────────────────────────────────────────────────────────
# IPC para un hit del RAG
# ──────────────────────────────────────────────────────────────────────────────

def inferir_ipc_hit(hit: dict, ipc_dict: dict) -> float:
    """Devuelve el IPC correspondiente al período de origen de un hit RAG.

    Estrategia en orden de prioridad:
      1. Fecha del nombre de archivo del hit  (hit['file'])
      2. Año en el identificador de licitación (hit['licitacion'])
      3. Fallback → último IPC disponible (sin ajuste efectivo)
    """
    if not ipc_dict:
        return 100.0

    ultimo_ipc = list(ipc_dict.values())[-1]

    def _buscar_ipc(anio: int, mes_base: int) -> Optional[float]:
        """Busca el IPC exacto o el más cercano dentro de ±6 meses."""
        v = ipc_dict.get((anio, mes_base))
        if v:
            return v
        for delta in range(1, 7):
            for signo in (-1, 1):
                m2 = mes_base + signo * delta
                a2 = anio + (m2 - 1) // 12
                m2 = ((m2 - 1) % 12) + 1
                v = ipc_dict.get((a2, m2))
                if v:
                    return v
        return None

    # 1. Desde el nombre de archivo
    fecha = extraer_fecha_de_archivo(hit.get('file', ''))
    if fecha:
        v = _buscar_ipc(*fecha)
        if v:
            return v

    # 2. Desde el identificador de licitación
    m = re.search(r'(20\d{2})', str(hit.get('licitacion', '')))
    if m:
        anio = int(m.group(1))
        v = _buscar_ipc(anio, 7)
        if v:
            return v

    return ultimo_ipc


# ──────────────────────────────────────────────────────────────────────────────
# Extracción de fecha desde texto de EETT y nombre de PDF
# ──────────────────────────────────────────────────────────────────────────────

# Patrones específicos para EETT/BBTT/ECO de licitaciones de minería chilena.
# Se evalúan en orden; el primero que haga match gana.
_PATRONES_EETT = [
    # ── Encabezados de control de documentos ──────────────────────────────
    # "FECHA EMISIÓN: 15-03-2024" / "Fecha de Emisión: 15/03/2024"
    (r'(?:fecha\s*(?:de\s*)?emisi[oó]n|fecha\s*publicaci[oó]n|fecha\s*vigencia'
     r'|fecha\s*elaboraci[oó]n|elaborado|emitido|aprobado)\s*[:\-]?\s*'
     r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
     lambda m: (int(m.group(3)), int(m.group(2)))),   # DD/MM/YYYY → (año, mes)

    # "Válido desde: Enero 2024" / "Vigencia: Marzo 2024"
    (r'(?:v[aá]lid[ao]\s*desde|vigencia(?:\s*desde)?|vigente\s*desde)\s*[:\-]?\s*'
     r'([a-záéíóú]+)\s+(20\d{2})',
     lambda m: (int(m.group(2)), _MES_MAP.get(m.group(1).lower()))),

    # "Rev. 0 — Enero 2024" / "Revisión 2 - Enero 2024" (tablas de revisiones)
    (r'(?:rev(?:isi[oó]n)?\.?\s*\d+\s*[-–—]?\s*)'
     r'([a-záéíóú]+)\s+(20\d{2})',
     lambda m: (int(m.group(2)), _MES_MAP.get(m.group(1).lower()))),

    # "FECHA: 01/2024" o "FECHA: 01-2024"
    (r'(?:fecha)\s*[:\-]\s*(0?[1-9]|1[0-2])[/\-](20\d{2})',
     lambda m: (int(m.group(2)), int(m.group(1)))),

    # ── Patrones textuales genéricos ──────────────────────────────────────
    # "15 de marzo de 2024" / "marzo de 2024"
    (r'(?:\d{1,2}\s+de\s+)?([a-záéíóú]+)\s+(?:de\s+)?(20\d{2})',
     lambda m: (int(m.group(2)), _MES_MAP.get(m.group(1).lower()))),

    # ── Patrones numéricos ────────────────────────────────────────────────
    # "15/03/2024" / "15-03-2024"
    (r'(\d{1,2})[/\-](\d{1,2})[/\-](20\d{2})',
     lambda m: (int(m.group(3)), int(m.group(2)))
     if 1 <= int(m.group(2)) <= 12 else None),
]


def extraer_fecha_y_ipc(
    texto_pdf: str,
    nombre_archivo: str,
    ipc_dict: dict,
) -> tuple[int, int, float]:
    """Extrae (año, mes, ipc) desde el PDF de EETT/BBTT de una licitación minera.

    Prioridad de búsqueda:
      1. Nombre del archivo  (YYYY-MM-DD, ECO_YYYYMM, ENE2024, …)
      2. Patrones específicos de EETT/ECO de minería en el texto
      3. Fallback → año/mes actuales con el IPC más reciente disponible
    """
    anio_default, mes_default = 2026, 1
    ultimo_ipc = list(ipc_dict.values())[-1] if ipc_dict else 100.0

    def _lookup(anio: int, mes: int) -> Optional[float]:
        v = ipc_dict.get((anio, mes))
        if v:
            return v
        # Buscar mes más cercano del mismo año
        for delta in range(1, 7):
            for signo in (-1, 1):
                m2 = mes + signo * delta
                a2 = anio + (m2 - 1) // 12
                m2 = ((m2 - 1) % 12) + 1
                v = ipc_dict.get((a2, m2))
                if v:
                    return v
        return None

    # ── 1. Desde el nombre de archivo ────────────────────────────────────
    fecha = extraer_fecha_de_archivo(nombre_archivo)
    if fecha:
        anio, mes = fecha
        ipc = _lookup(anio, mes)
        if ipc:
            return anio, mes, ipc

    # ── 2. Desde el texto del PDF (patrones EETT/ECO) ────────────────────
    texto_lower = str(texto_pdf).lower()
    for patron, extractor in _PATRONES_EETT:
        m = re.search(patron, texto_lower, re.IGNORECASE)
        if m:
            try:
                resultado = extractor(m)
            except Exception:
                continue
            if resultado is None:
                continue
            anio, mes = resultado
            if not mes or not (2000 <= anio <= 2050 and 1 <= mes <= 12):
                continue
            ipc = _lookup(anio, mes)
            if ipc:
                return anio, mes, ipc

    # ── 3. Fallback ───────────────────────────────────────────────────────
    return anio_default, mes_default, ultimo_ipc
