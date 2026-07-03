import io
import os
import tempfile
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pdfplumber
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from nlp.text_processor import TextProcessor
from nlp.rag_index import RagIndex, extract_itemizado_excel_rows, build_pdf_index
from nlp.valuation_service import ValuationService
from nlp.xgboost_estimator import XGBoostValuationModel
from nlp.macro_utils import cargar_ipc, extraer_fecha_y_ipc


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_DIR = os.path.join(DATA_DIR, 'index')

# ── IPC ────────────────────────────────────────────────────────────────────────
ruta_ipc = os.path.join(DATA_DIR, 'IPC_VAR_MEN1_HIST_NEW.csv')
diccionario_ipc = cargar_ipc(ruta_ipc)
if diccionario_ipc:
    print(f"IPC Cargado: {len(diccionario_ipc)} meses históricos indexados.")

# IPC más reciente — se usa como denominador para normalizar ipc_factor en inferencia
ipc_referencia: float = max(diccionario_ipc.values()) if diccionario_ipc else 100.0

# ── Flask / NLP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
processor = TextProcessor()
embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# RAG search — FAISS + SBERT
rag = RagIndex(INDEX_DIR, embedder=embedder)
if rag.load():
    print(f"RAG: índice cargado ({len(rag.metadata)} fragmentos)")
else:
    try:
        info = rag.build(DATA_DIR)
        print(f"RAG: índice construido ({info['chunks']} fragmentos)")
        for w in info['warnings']:
            print(f"RAG: aviso — {w}")
    except Exception as e:
        print(f"RAG: error construyendo índice — {e}")

# Módulo de Valorización Gaussiana
vs = ValuationService(rag, top_k=5)

# ── historical.csv — auto-generación y carga ───────────────────────────────────
hist_path = os.path.join(DATA_DIR, 'historical.csv')

if not os.path.exists(hist_path):
    print("[Historical] historical.csv no encontrado — generando automáticamente desde ECO xlsx...")
    try:
        from build_historical import main as _build_hist
        _build_hist()
    except SystemExit:
        print("[Historical] build_historical finalizó con error (¿no hay ECO xlsx?). XGBoost inactivo.")
    except Exception as exc:
        print(f"[Historical] Error al generar: {exc}")

historical_df = None
if os.path.exists(hist_path):
    try:
        historical_df = pd.read_csv(hist_path)

        # Garantizar columnas obligatorias con valores por defecto
        for col, default in [('anio', 2024), ('mes', 1), ('ipc_factor', 1.0),
                              ('descripcion', ''), ('unidad', ''),
                              ('cantidad', 1.0), ('precio_unitario', 0.0)]:
            if col not in historical_df.columns:
                historical_df[col] = default

        n_valid = (historical_df['precio_unitario'] > 0).sum()
        print(
            f"[Historical] {len(historical_df)} registros cargados | "
            f"{n_valid} con precio_unitario > 0"
        )
    except Exception as exc:
        print(f"[Historical] ERROR al cargar — {exc}")
        historical_df = None

# ── Regresión lineal simple (auxiliar, no bloquea) ────────────────────────────
reg_model = None
reg_numeric_cols: list = []
reg_path = os.path.join(DATA_DIR, 'regression.csv')
if os.path.exists(reg_path):
    try:
        reg_df = pd.read_csv(reg_path)
        possible = ['volume', 'year', 'ipc', 'value']
        numeric = [c for c in reg_df.columns if c in possible or np.issubdtype(reg_df[c].dtype, np.number)]
        if 'value' in reg_df.columns and len(numeric) > 0:
            reg_numeric_cols = [c for c in numeric if c != 'value']
            X = reg_df[reg_numeric_cols].fillna(0).values
            y = reg_df['value'].fillna(0).values
            reg_model = LinearRegression().fit(X, y)
    except Exception:
        reg_model = None

# ── XGBoost — persistencia + multi-modelo jerárquico ──────────────────────────
xgb_metricas: dict | None = None
xgb_cv_tipo: str = ""

# Intentar cargar desde caché (skip reentrenamiento si el CSV no cambió)
xgb_model = XGBoostValuationModel.cargar_si_valido(hist_path) or XGBoostValuationModel()

if xgb_model.is_fitted:
    # Modelo cargado — re-adjuntar embedder (no se serializa en el pkl)
    xgb_model._embedder = embedder
    xgb_metricas = xgb_model.metricas_globales
    xgb_cv_tipo  = xgb_model.cv_tipo
    m = xgb_metricas or {}
    r2_s = f"{m['r2']:.2f}" if m.get('r2') is not None else "—"
    print(
        f"[XGBoost] ACTIVO ✓ (caché) — "
        f"MAE=${m.get('mae', 0):,.0f}  MAPE={m.get('mape_pct', 0):.1f}%  R²={r2_s}"
    )
elif historical_df is not None and (historical_df['precio_unitario'] > 0).sum() >= 10:
    try:
        xgb_model.fit(historical_df, embedder, k_por_grupo=4)
        xgb_metricas = xgb_model.metricas_globales
        xgb_cv_tipo  = xgb_model.cv_tipo
        xgb_model.guardar(hist_path)   # ← persiste para próximo restart
    except Exception as exc:
        print(f"[XGBoost] ERROR al entrenar — {exc}")
else:
    causas = []
    if historical_df is None:
        causas.append("historical.csv no encontrado o vacío")
    else:
        causas.append(f"solo {(historical_df['precio_unitario'] > 0).sum()} filas con precio > 0 (mínimo 10)")
    print(f"[XGBoost] INACTIVO — {'; '.join(causas)}")


# ── Rutas ──────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/buscar', methods=['GET', 'POST'])
def buscar():
    question = ''
    licitacion = ''
    results = []
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        licitacion = request.form.get('licitacion', '').strip()
        if question:
            results = rag.search(question, top_k=8, licitacion=licitacion or None)
    return render_template(
        'buscar.html',
        question=question,
        licitacion=licitacion,
        results=results,
        licitaciones=rag.licitaciones(),
    )


@app.route('/valorizar', methods=['GET', 'POST'])
def valorizar():
    resultados: list = []
    error = ''
    licitacion = ''

    if request.method == 'POST':
        licitacion     = request.form.get('licitacion', '').strip()
        # checkbox marcado → envía 'true'; desmarcado → ausente → 'false'
        usar_gaussiano = request.form.get('usar_gaussiano', 'false').lower() == 'true'
        archivo_excel  = request.files.get('archivo_excel')
        archivo_pdf   = request.files.get('archivo_pdf')

        if not archivo_excel or not archivo_excel.filename:
            error = 'Se requiere el archivo Excel del itemizado.'
        elif not archivo_pdf or not archivo_pdf.filename:
            error = 'Se requiere el archivo PDF de bases técnicas.'
        else:
            try:
                # Excel → path temporal (openpyxl requiere ruta en disco)
                fd, tmp_xlsx = tempfile.mkstemp(suffix='.xlsx')
                try:
                    os.close(fd)
                    archivo_excel.save(tmp_xlsx)
                    all_rows = extract_itemizado_excel_rows(tmp_xlsx, licitacion or 'nuevo')
                finally:
                    os.unlink(tmp_xlsx)


                if not all_rows:
                    error = 'No se detectaron ítems en el Excel. Verifica el formato del archivo.'
                else:
                    # PDF → texto
                    bbtt_bytes = io.BytesIO(archivo_pdf.read())
                    with pdfplumber.open(bbtt_bytes) as pdf:
                        bbtt_texto = '\n'.join(
                            page.extract_text() or '' for page in pdf.pages
                        )
                        build_pdf_index(rag, bbtt_texto)

                    year_lic, mes_lic, ipc_actual = extraer_fecha_y_ipc(
                        bbtt_texto,
                        archivo_pdf.filename,
                        diccionario_ipc,
                    )
                    print(f"Contexto detectado → Año: {year_lic}, Mes: {mes_lic}, IPC: {ipc_actual:.2f}")

                    items_hoja = [
                        {
                            "item":      r["item"],
                            "nombre":    r["descripcion"],
                            "unidad":    r["unidad"],
                            "breadcrumb": r.get("breadcrumb", ""),
                            "cantidad":  float(r.get("cantidad") or 0.0),
                        }
                        for r in all_rows if r["tipo"] == "item"
                    ]

                    # ── Gaussiano (todos los ítems) ────────────────────────────
                    val_list   = vs.estimate_batch(
                        items=items_hoja,
                        bbtt_texto=bbtt_texto,
                        licitacion=licitacion or None,
                        ipc_actual=ipc_actual,
                        ipc_dict=diccionario_ipc,
                    )
                    val_by_item = {r["item_num"]: r for r in val_list}

                    n_items = sum(1 for r in all_rows if r.get("tipo") == "item")
                    print(
                        f"[valorizar] {n_items} ítems | "
                        f"XGBoost {'ACTIVO (fallback sin-refs)' if xgb_model.is_fitted else 'INACTIVO'}"
                    )

                    # ipc_factor para XGBoost — ratio respecto al IPC de referencia
                    ipc_factor_xgb = (ipc_actual / ipc_referencia) if ipc_referencia > 0 else 1.0

                    xgb_count   = 0
                    gauss_count = 0

                    for row in all_rows:
                        if row["tipo"] == "titulo":
                            resultados.append({
                                "tipo":        "titulo",
                                "item":        row["item"],
                                "descripcion": row["descripcion"],
                            })
                            continue

                        val      = val_by_item.get(row["item"], {})
                        pu_gauss = val.get("pu_estimado")
                        sin_refs = (pu_gauss is None or bool(val.get("error")))

                        # usar_gaussiano=True  → XGBoost solo como fallback (sin refs)
                        # usar_gaussiano=False → XGBoost para todos los ítems
                        usar_xgb = xgb_model.is_fitted and (sin_refs or not usar_gaussiano)

                        if usar_xgb:
                            try:
                                pred   = xgb_model.predict(
                                    descripcion=row.get('descripcion', ''),
                                    unidad=row.get('unidad', ''),
                                    cantidad=float(row.get('cantidad') or 1.0),
                                    anio=year_lic,
                                    mes=mes_lic,
                                    ipc_factor=ipc_factor_xgb,
                                )
                                pu_xgb = pred.get("pu_estimado")
                                if pu_xgb:
                                    fuente_tipo = "xgboost_fallback" if sin_refs else "xgboost_forzado"
                                    val["pu_estimado"]         = pu_xgb
                                    val["modelo_usado"]        = f"{fuente_tipo}|{pred.get('clave_modelo', '?')}"
                                    val["metricas_validacion"] = None
                                    val["metricas_xgb_global"] = xgb_metricas
                                    val["xgb_cv_tipo"]         = xgb_cv_tipo
                                    val["margen_error"]        = None
                                    val["error"]               = None
                                    xgb_count += 1
                                    print(
                                        f"  [{fuente_tipo}] ítem {row['item']:>10} | "
                                        f"${pu_xgb:,.0f} /u | "
                                        f"clave={pred.get('clave_modelo', '?')} | "
                                        f"fuente={pred.get('fuente', '?')}"
                                    )
                                elif pu_gauss:
                                    # XGBoost devolvió None pero Gaussiano tenía algo → usar Gaussiano
                                    val["modelo_usado"] = "gaussiano"
                                    gauss_count += 1
                                else:
                                    val["modelo_usado"] = "sin_estimacion"
                            except Exception as xgb_err:
                                print(f"  [XGB error] ítem {row['item']} — {xgb_err}")
                                val["modelo_usado"] = "gaussiano" if pu_gauss else "sin_estimacion"
                                if pu_gauss:
                                    gauss_count += 1
                        else:
                            val["modelo_usado"] = "gaussiano"
                            gauss_count += 1
                            print(
                                f"  [Gauss]        ítem {row['item']:>10} | "
                                f"${pu_gauss:,.0f} /u"
                            )

                        resultados.append({**row, **val})

                    print(
                        f"[valorizar] resumen → gaussiano: {gauss_count} | "
                        f"xgboost_fallback: {xgb_count} | "
                        f"sin estimación: {sum(1 for r in val_list if r.get('error') and r.get('pu_estimado') is None)}"
                    )

            except Exception as exc:
                error = f'Error procesando archivos: {exc}'

    # Contexto IPC para el template (solo disponible tras POST exitoso)
    ipc_contexto = None
    if request.method == 'POST' and not error and resultados:
        try:
            ipc_contexto = {
                "year": year_lic,
                "mes":  mes_lic,
                "ipc":  round(ipc_actual, 2),
            }
        except NameError:
            pass

    return render_template(
        'valorizar.html',
        resultados=resultados,
        error=error,
        licitacion=licitacion,
        licitaciones=rag.licitaciones(),
        ipc_contexto=ipc_contexto,
        xgb_activo=xgb_model.is_fitted,
        xgb_metricas=xgb_metricas,
        xgb_cv_tipo=xgb_cv_tipo,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
