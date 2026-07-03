"""valuation_service.py — Servicio de valorización con ajuste IPC.

Cambios respecto a la versión anterior:
  · _normalize_ipc_candidates()   nueva — ajusta precios históricos al IPC actual
  · _build_candidates()           recibe ipc_actual + ipc_dict → agrega precio_ajustado
  · ValuationService.estimate()   recibe ipc_actual + ipc_dict
  · ValuationService.estimate_batch() recibe ipc_actual + ipc_dict
  · El modelo Gaussiano trabaja sobre precios_ajustados (valores reales)
  · El resultado expone tanto precio_nominal como precio_ajustado por transparencia
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import numpy as np

from nlp.gaussian_estimator import GaussianValuationModel
from nlp.macro_utils import inferir_ipc_hit, normalizar_precio_ipc
from nlp.rag_index import RagIndex
from nlp.rag_index import search_pdf as rag_search_pdf

SCORE_THRESHOLD = 0.70
#SCORE_THRESHOLD_NOFIL = 0.85

_UNIT_ALIASES: dict[str, str] = {
    "c/u": "cu", "und": "un", "unid": "un", "uni": "un", "unidades": "un",
    "hr": "hh", "hrs": "hh",
    "m3.": "m3", "m2.": "m2", "ml.": "lm", "mts": "m", "mt": "m",
    "kg.": "kg", "ton.": "ton", "lt.": "lt", "gl.": "gl", "glb": "gl",
    "dia.": "dia", "mes.": "mes", "sem": "mes",
}


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ".").strip())
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def _normalize_unit(unit: str) -> str:
    s = re.sub(r"\s+", "", unit).lower().rstrip(".")
    return _UNIT_ALIASES.get(s, s)


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str)


def _apply_iqr_filter(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filtra outliers usando IQR sobre precio_ajustado (ya normalizado)."""
    if len(candidates) < 4:
        return candidates
    # Usar precio ajustado si existe, sino nominal
    prices = np.array(
        [c.get("precio_ajustado", c["precio_unitario"]) for c in candidates],
        dtype=float,
    )
    q1, q3 = float(np.percentile(prices, 25)), float(np.percentile(prices, 75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [
        c for c in candidates
        if lo <= c.get("precio_ajustado", c["precio_unitario"]) <= hi
    ]


def _build_candidates(
    rag_results: list[dict],
    query_desc: str,
    target_unit: Optional[str],
    threshold: float,
    apply_unit_filter: bool,
    ipc_actual: Optional[float] = None,
    ipc_dict: Optional[dict] = None,
) -> tuple[list[dict], int, bool]:
    """Filtra resultados RAG y aplica ajuste IPC a cada precio.

    Cada candidato ahora incluye:
      · precio_unitario   — precio nominal (original del índice)
      · ipc_hist          — IPC del período del documento fuente
      · factor_ipc        — ipc_actual / ipc_hist (1.0 si sin ajuste)
      · precio_ajustado   — precio_unitario × factor_ipc (a valores actuales)
    """
    candidates: list[dict[str, Any]] = []
    descartados_unidad = 0
    exact_match_found = False

    for hit in rag_results:
        precio_nominal = _to_float(hit.get("precio_unitario"))
        if precio_nominal is None:
            continue

        if apply_unit_filter and target_unit is not None:
            if _normalize_unit(str(hit.get("unidad", ""))) != target_unit:
                descartados_unidad += 1
                continue

        score = float(hit.get("score", 0.0))

        hit_desc = _normalize_text(
            str(hit.get("descripcion") or hit.get("text") or "")
        )
        if hit_desc == query_desc:
            exact_match_found = True
            score = 1.0

        # ── Ajuste IPC ────────────────────────────────────────────────────
        ipc_hist: float = 100.0
        factor_ipc: float = 1.0
        precio_ajustado: float = precio_nominal

        if ipc_dict and ipc_actual:
            ipc_hist = inferir_ipc_hit(hit, ipc_dict)
            factor_ipc = ipc_actual / ipc_hist if ipc_hist > 0 else 1.0
            precio_ajustado = normalizar_precio_ipc(precio_nominal, ipc_hist, ipc_actual)

        candidates.append({
            "licitacion": hit.get("licitacion"),
            "item": hit.get("item"),
            "descripcion": hit.get("descripcion"),
            "unidad": hit.get("unidad"),
            "precio_unitario": precio_nominal,       # valor histórico original
            "ipc_hist": round(ipc_hist, 4),
            "factor_ipc": round(factor_ipc, 6),
            "precio_ajustado": round(precio_ajustado, 2),  # a valores actuales
            "score": score,
            "text": hit.get("text", "")[:200],
            "exact_match": hit_desc == query_desc,
            "file": hit.get("file", ""),
        })

    if exact_match_found:
        candidates = [c for c in candidates if c["exact_match"]]
    else:
        candidates = [c for c in candidates if c["score"] >= threshold]

    candidates = _apply_iqr_filter(candidates)
    return candidates, descartados_unidad, exact_match_found


def _fit_and_metrics(
    candidates: list[dict[str, Any]],
) -> tuple[dict, Optional[dict]]:
    """Ajusta GaussianValuationModel sobre precios_ajustados y calcula LOO."""
    # Usar precio ajustado si disponible (ya normalizado a valores actuales)
    precios = [c.get("precio_ajustado", c["precio_unitario"]) for c in candidates]
    scores  = [c["score"] for c in candidates]

    model = GaussianValuationModel().fit(precios, scores)
    prediction = model.predict()

    metricas: Optional[dict] = None
    if len(precios) >= 2:
        y_true, y_pred = [], []
        for i in range(len(precios)):
            rest_p = precios[:i] + precios[i + 1:]
            rest_s = scores[:i] + scores[i + 1:]
            if not rest_p:
                continue
            loo = GaussianValuationModel().fit(rest_p, rest_s).predict()
            y_true.append(precios[i])
            y_pred.append(loo["pu_estimado"])
        if y_true:
            metricas = GaussianValuationModel.evaluate_model(y_true, y_pred)

    return prediction, metricas


class ValuationService:
    def __init__(self, rag: RagIndex, top_k: int = 5) -> None:
        self.rag = rag
        self.top_k = top_k

    def estimate(
        self,
        nombre_item: str,
        especificacion_tecnica: str,
        unidad_filtro: Optional[str] = None,
        licitacion: Optional[str] = None,
        breadcrumb: str = None,
        ipc_actual: Optional[float] = None,
        ipc_dict: Optional[dict] = None,
    ) -> dict[str, Any]:
        query = f"{breadcrumb} {nombre_item}".strip() if breadcrumb else nombre_item.strip()

        fetch_k = self.top_k * 8
        rag_results = self.rag.search(query, top_k=fetch_k, licitacion=licitacion)

        target_unit = _normalize_unit(unidad_filtro) if unidad_filtro else None
        query_desc  = _normalize_text(nombre_item)

        candidates, descartados_unidad, exact_match = _build_candidates(
            rag_results,
            query_desc,
            target_unit,
            SCORE_THRESHOLD,
            apply_unit_filter=True,
            ipc_actual=ipc_actual,
            ipc_dict=ipc_dict,
        )
        unit_filter_used = True
        fallback_used    = False

        candidates = candidates[: self.top_k]

        if not candidates:
            return {
                "query": query,
                "items_referencia": [],
                "descartados_unidad": descartados_unidad,
                "exact_match": False,
                "unit_filter_used": unit_filter_used,
                "fallback_used": False,
                "pu_estimado": None,
                "margen_error": None,
                "ipc_actual": ipc_actual,
                "metricas_validacion": None,
                "error": f"Sin referencias con score ≥ {SCORE_THRESHOLD}.",
            }

        prediction, metricas = _fit_and_metrics(candidates)

        return {
            "query": query,
            "items_referencia": candidates,
            "descartados_unidad": descartados_unidad,
            "exact_match": exact_match,
            "unit_filter_used": unit_filter_used,
            "fallback_used": fallback_used,
            "pu_estimado": prediction["pu_estimado"],        # ya en valores actuales
            "margen_error": prediction["margen_error"],
            "ipc_actual": ipc_actual,
            "metricas_validacion": metricas,
            "error": None,
        }

    def estimate_batch(
        self,
        items: list[dict[str, Any]],
        bbtt_texto: str,
        licitacion: Optional[str] = None,
        ipc_actual: Optional[float] = None,
        ipc_dict: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        resultados: list[dict[str, Any]] = []

        for item in items:
            nombre    = str(item.get("nombre") or item.get("descripcion") or "")
            unidad    = str(item.get("unidad") or "")
            breadcrumb = str(item.get("breadcrumb") or "")

            bbtt_snippet = "\n".join(rag_search_pdf(nombre, top_k=3))
            resultado = self.estimate(
                nombre_item=nombre,
                especificacion_tecnica=bbtt_snippet,
                unidad_filtro=unidad or None,
                licitacion=licitacion,
                breadcrumb=breadcrumb,
                ipc_actual=ipc_actual,
                ipc_dict=ipc_dict,
            )
            resultado["item_num"] = item.get("item", "")
            resultado["nombre"]   = nombre
            resultado["unidad"]   = unidad

            resultados.append(resultado)

        return resultados
