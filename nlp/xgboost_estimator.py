"""xgboost_estimator.py — Multi-modelo jerárquico con taxonomía keyword + K-Means híbrido.

Arquitectura en dos niveles:
  Nivel 1: grupo_unidad  (mano_obra, hora_maquina, volumetrico, …)
  Nivel 2: sub_cluster   keyword match → subcategoría explícita (hormigon, excavacion_roca…)
                         K-Means solo si no hay match keyword

Persistencia:
  guardar(csv_path) → data/xgb_model.pkl  +  MD5 del CSV
  cargar_si_valido(csv_path) → modelo cacheado si el CSV no cambió, None si hay que reentrenar

Features (4, sin número de ítem ni empresa):
  log(cantidad), anio, mes, ipc_factor
Target: log1p(precio_unitario)
"""
from __future__ import annotations

import hashlib
import re as _re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Ruta del pkl relativa a este módulo → data/xgb_model.pkl
MODEL_PATH = Path(__file__).parent.parent / 'data' / 'xgb_model.pkl'


# ── Mapa de grupos de unidad ───────────────────────────────────────────────────

GRUPO_UNIDAD: dict[str, str] = {
    'hh': 'mano_obra',  'hr': 'mano_obra',  'hrs': 'mano_obra',
    'hm': 'hora_maquina',
    'dia': 'dia',        'día': 'dia',        'jornada': 'dia',
    'm3': 'volumetrico',
    'ml': 'lineal',      'lm': 'lineal',      'm': 'lineal',
    'm2': 'area',
    'kg': 'peso',        'ton': 'peso',
    'gl': 'global',      'glb': 'global',
    'un': 'unidad',      'cu': 'unidad',      'und': 'unidad',
}
GRUPO_DEFAULT = 'otros'


# ── Taxonomía keyword por grupo de unidad ──────────────────────────────────────
# Claves = valores de GRUPO_UNIDAD. Patrones evaluados en orden; primer match gana.

TAXONOMIA: dict[str, list[tuple[str, str]]] = {
    'volumetrico': [
        (r'hormig[oó]n|concreto|H\d{2}',                 'hormigon'),
        (r'excav.*(roca|no.rip|volad|dura)',               'excavacion_roca'),
        (r'excav|zanja|material.com[uú]n|tierra|suelo',   'excavacion_suelo'),
        (r'relleno|compact',                               'relleno'),
        (r'material.dren|drena|filtro|arena|grava',        'material_dren'),
        (r'escorias?|estéril|lastre|ripios?',              'esteril'),
    ],
    'mano_obra': [
        (r'supervisor|jefe|inspector|ingenier',            'profesional'),
        (r'maestro|técnic|calificad',                      'tecnico'),
        (r'soldad',                                        'soldador'),
        (r'operador',                                      'operador_maquinaria'),
        (r'operario|ayudante|cuadrilla|aseo',              'operario'),
    ],
    'hora_maquina': [
        (r'excavadora|retroexcav',                         'excavadora'),
        (r'buldó?zer|bulldozer|topadora|tractor\s*(?:d6|d7|d8|d9)',  'bulldozer'),
        (r'grúa|grua|pluma|alzahombre|manlift',            'grua'),
        (r'camión\s*tolva|camion\s*tolva',                 'camion_tolva'),
        (r'camión|camion|transporte',                      'camion_general'),
        (r'motonivelad|nivelad',                           'motoniveladora'),
        (r'compactor|rodillo',                             'compactador'),
        (r'cargador|pala\s*mecan',                         'cargador'),
    ],
    'dia': [
        (r'camión|camion|equipo.pesad|maquinaria',         'equipo_dia'),
        (r'cuadrilla|personal|trabajad',                   'personal_dia'),
    ],
    'lineal': [
        (r'HDPE|tubería|tuber[ií]a|cañería|PVC',           'tuberia'),
        (r'geomembrana',                                   'geomembrana'),
        (r'drén|dren|subdren',                             'dren_lineal'),
    ],
    'peso': [
        (r'acero|varilla|barra|fierro',                    'acero'),
        (r'HDPE|polietilen',                               'hdpe_kg'),
    ],
    'global': [
        (r'moviliz',                                       'movilizacion'),
        (r'desmoviliz',                                    'desmovilizacion'),
        (r'instalac.*faena|faena',                         'instalacion_faena'),
        (r'desmantel|retiro|demolicion',                   'desmantelamiento'),
    ],
}


def _asignar_subcategoria(descripcion: str, grupo_u: str) -> Optional[str]:
    """Keyword match → subcategoría.  None si no hay match (K-Means decide)."""
    desc_lower = descripcion.lower()
    for patron, subcat in TAXONOMIA.get(grupo_u, []):
        if _re.search(patron, desc_lower, _re.IGNORECASE):
            return subcat
    return None


# ── Detección CUDA ─────────────────────────────────────────────────────────────

def _detectar_device() -> str:
    try:
        p = xgb.XGBRegressor(device='cuda', n_estimators=1, verbosity=0)
        p.fit([[0, 0, 0, 0]], [0])
        print("[XGBoost] GPU (CUDA) detectada ✓")
        return 'cuda'
    except Exception:
        print("[XGBoost] GPU no disponible → usando CPU")
        return 'cpu'


DEVICE = _detectar_device()


# ── Features ───────────────────────────────────────────────────────────────────

def _features(df_g: pd.DataFrame) -> np.ndarray:
    """4 features. sub_cluster ya es implícito (determina qué modelo se usa)."""
    return np.column_stack([
        np.log1p(df_g['cantidad'].values.astype(float)),
        df_g['anio'].values.astype(float),
        df_g['mes'].values.astype(float),
        df_g['ipc_factor'].values.astype(float),
    ])


# ── Métricas ───────────────────────────────────────────────────────────────────

def _evaluar(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nz   = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else 0.0
    r2: Optional[float] = None
    if len(y_true) > 1:
        r2v = float(r2_score(y_true, y_pred))
        r2  = None if (r2v != r2v) else r2v
    return {'mae': mae, 'rmse': rmse, 'mape_pct': mape, 'r2': r2}


# ── Modelo ──────────────────────────────────────────────────────────────────────

class XGBoostValuationModel:
    """Multi-modelo jerárquico: grupo_unidad × sub_cluster (keyword o K-Means).

    self.modelos          : {"volumetrico_hormigon": XGB, "mano_obra_tecnico": XGB, …}
    self.kmeans_por_grupo : {"volumetrico": KMeans, …}  — solo para 'otros_*'
    self.stats            : {"volumetrico_hormigon": {tipo, n, mae, …}, …}
    """

    def __init__(self) -> None:
        self.modelos:           dict[str, xgb.XGBRegressor] = {}
        self.stats:             dict[str, dict]              = {}
        self.kmeans_por_grupo:  dict[str, KMeans]            = {}
        self._embedder  = None    # ref. al SentenceTransformer — NO se serializa
        self.is_fitted:  bool    = False
        self.metricas_por_grupo: dict[str, dict] = {}
        self.metricas_globales:  Optional[dict]  = None
        self.cv_tipo: str = ""
        self.device:  str = DEVICE

    # ── Persistencia ───────────────────────────────────────────────────────────

    @staticmethod
    def _hash_csv(csv_path: str) -> str:
        h = hashlib.md5()
        with open(csv_path, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()

    def guardar(self, csv_path: str) -> None:
        """Serializa el modelo + MD5 del CSV. El embedder queda excluido del pkl."""
        emb_bak = self._embedder
        self._embedder = None
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'modelo': self, 'csv_hash': self._hash_csv(csv_path)}, MODEL_PATH)
            print(f"[XGBoost] Modelo guardado → {MODEL_PATH}")
        finally:
            self._embedder = emb_bak

    @classmethod
    def cargar_si_valido(cls, csv_path: str) -> Optional["XGBoostValuationModel"]:
        """Carga el modelo cacheado si el CSV no cambió; None si hay que reentrenar."""
        import os
        if not os.path.exists(csv_path) or not MODEL_PATH.exists():
            return None
        try:
            payload = joblib.load(MODEL_PATH)
            if payload.get('csv_hash') == cls._hash_csv(csv_path):
                print(f"[XGBoost] Modelo cargado desde caché → skip reentrenamiento")
                return payload['modelo']
            print("[XGBoost] CSV modificado → reentrenando")
            return None
        except Exception as exc:
            print(f"[XGBoost] Caché inválida ({exc}) → reentrenando")
            return None

    # ── Entrenamiento ──────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        embedder,
        k_por_grupo:  int = 4,
        cv_folds:     int = 5,
        min_muestras: int = 15,
    ) -> "XGBoostValuationModel":
        """Entrena un modelo por (grupo_unidad, sub_cluster).

        Paso 1: keyword taxonomy → sub_cluster explícito
        Paso 2: K-Means solo para ítems sin match keyword
        Paso 3: XGBoost por clave compuesta (grupo_u_subcat)
        """
        df = df.copy()
        for col, default in [('anio', 2024), ('mes', 1), ('ipc_factor', 1.0),
                              ('cantidad', 1.0), ('descripcion', ''), ('unidad', ''),
                              ('precio_unitario', 0.0), ('peso_muestra', 1.0)]:
            if col not in df.columns:
                df[col] = default

        df = df[df['precio_unitario'] > 0].reset_index(drop=True)
        if df.empty:
            raise ValueError("No hay filas con precio_unitario > 0.")

        self.modelos.clear()
        self.stats.clear()
        self.kmeans_por_grupo.clear()
        self.metricas_por_grupo.clear()

        df['grupo_unidad'] = (
            df['unidad'].str.lower().str.strip()
            .map(GRUPO_UNIDAD).fillna(GRUPO_DEFAULT)
        )

        all_y_true: list[float] = []
        all_y_oof:  list[float] = []

        for grupo_u, df_u in df.groupby('grupo_unidad'):
            df_u = df_u.copy()

            # Paso 1 — keyword taxonomy
            df_u['subcat_kw'] = df_u['descripcion'].apply(
                lambda d: _asignar_subcategoria(str(d), grupo_u)
            )

            # Paso 2 — K-Means para los sin match
            sin_match = df_u['subcat_kw'].isna()
            n_sin     = int(sin_match.sum())

            if n_sin >= 4:
                descs_sm = df_u.loc[sin_match, 'descripcion'].astype(str).tolist()
                print(f"\n[XGBoost.fit] [{grupo_u}] K-Means sobre {n_sin} ítems sin keyword...")
                embs_sm = embedder.encode(
                    descs_sm,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype('float32')
                n_otros = max(2, min(k_por_grupo, df_u.loc[sin_match, 'descripcion'].nunique() // 3))
                km = KMeans(n_clusters=n_otros, random_state=42, n_init=10)
                km.fit(embs_sm)
                self.kmeans_por_grupo[grupo_u] = km
                df_u.loc[sin_match, 'subcat_kw'] = [f"otros_{l}" for l in km.labels_]
            else:
                df_u.loc[sin_match, 'subcat_kw'] = 'otros_0'

            df_u['sub_cluster'] = df_u['subcat_kw']

            # Paso 3 — XGBoost por (grupo_u, sub_cluster)
            for sub_cl, df_g in df_u.groupby('sub_cluster'):
                clave  = f"{grupo_u}_{sub_cl}"
                n      = len(df_g)
                y_raw  = df_g['precio_unitario'].values.astype(float)
                ejems  = df_g['descripcion'].value_counts().head(3).index.tolist()

                if n < min_muestras:
                    self.stats[clave] = {
                        'tipo':          'mediana',
                        'valor':         float(np.median(y_raw)),
                        'n':             n,
                        'grupo_u':       grupo_u,
                        'desc_ejemplos': ejems,
                    }
                    continue

                y = np.log1p(y_raw)
                X = _features(df_g)
                w = df_g['peso_muestra'].values.astype(float)

                n_est  = 400 if n >= 80 else 200
                modelo = xgb.XGBRegressor(
                    device           = self.device,
                    tree_method      = 'hist',
                    n_estimators     = n_est,
                    learning_rate    = 0.05,
                    max_depth        = 4,
                    subsample        = 0.8,
                    colsample_bytree = 0.8,
                    min_child_weight = max(3, n // 25),
                    objective        = 'reg:squarederror',
                    random_state     = 42,
                    verbosity        = 0,
                )

                if n >= cv_folds * 2:
                    kf    = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    y_oof = np.zeros_like(y)
                    for tr, va in kf.split(X):
                        mf = clone(modelo)

                        # NUEVO: Agregamos sample_weight=w[tr]
                        mf.fit(X[tr], y[tr], sample_weight=w[tr])

                        y_oof[va] = mf.predict(X[va])
                    metricas = _evaluar(y_raw, np.expm1(y_oof))
                    cv_label = f'{cv_folds}-fold'
                    all_y_true.extend(y_raw.tolist())
                    all_y_oof.extend(np.expm1(y_oof).tolist())
                else:
                    # NUEVO: Agregamos sample_weight=w
                    modelo.fit(X, y, sample_weight=w)
                    metricas = _evaluar(y_raw, np.expm1(modelo.predict(X)))
                    cv_label = 'in-sample'

                # NUEVO: Agregamos sample_weight=w al entrenamiento final definitivo
                modelo.fit(X, y, sample_weight=w)
                
                self.modelos[clave] = modelo
                stat = {
                    'tipo':          'xgboost',
                    'n':             n,
                    'cv':            cv_label,
                    'media_pu':      float(np.median(y_raw)),
                    'grupo_u':       grupo_u,
                    'desc_ejemplos': ejems,
                    **metricas,
                }
                self.stats[clave]             = stat
                self.metricas_por_grupo[clave] = {**metricas, 'n': n, 'cv': cv_label}

        self._embedder = embedder
        self.is_fitted = True
        self.cv_tipo   = f"{cv_folds}-fold CV jerárquico (unidad × cluster)"

        if all_y_true:
            self.metricas_globales = _evaluar(
                np.array(all_y_true), np.array(all_y_oof)
            )

        self._log_resumen()
        return self

    # ── Inferencia ─────────────────────────────────────────────────────────────

    def predict(
        self,
        descripcion: str,
        unidad:      str,
        cantidad:    float,
        anio:        int,
        mes:         int,
        ipc_factor:  float,
    ) -> dict:
        """Predice P.U. con fallback en cascada.

          1. Keyword match → XGBoost del sub_cluster explícito
          2. Keyword match → mediana del sub_cluster (pocos datos)
          3. K-Means → XGBoost / mediana del sub_cluster
          4. Mediana del grupo_unidad completo
        """
        if not self.is_fitted:
            raise RuntimeError("Llama fit() primero.")

        grupo_u = GRUPO_UNIDAD.get(str(unidad).lower().strip(), GRUPO_DEFAULT)
        qty     = max(float(cantidad), 0.0)

        # Nivel 1 — keyword taxonomy
        subcat = _asignar_subcategoria(str(descripcion), grupo_u)

        # Nivel 2 — K-Means si sin keyword
        if subcat is None:
            if grupo_u in self.kmeans_por_grupo and self._embedder is not None:
                emb    = self._embedder.encode(
                    [str(descripcion)],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype('float32')
                subcat = f"otros_{self.kmeans_por_grupo[grupo_u].predict(emb)[0]}"
            else:
                subcat = 'otros_0'

        clave = f"{grupo_u}_{subcat}"

        def _predict_xgb(clave_: str) -> Optional[float]:
            if clave_ not in self.modelos:
                return None
            x     = np.array([[np.log1p(qty), float(anio), float(mes), float(ipc_factor)]])
            y_log = float(self.modelos[clave_].predict(x)[0])
            return max(float(np.expm1(y_log)), 0.0)

        # Intento exacto
        pu = _predict_xgb(clave)
        if pu is not None:
            return {'pu_estimado': pu, 'total_estimado': pu * qty,
                    'grupo_unidad': grupo_u, 'sub_cluster': subcat,
                    'clave_modelo': clave, 'fuente': 'xgboost'}

        # Mediana del sub_cluster exacto
        if clave in self.stats and self.stats[clave].get('tipo') == 'mediana':
            pu = self.stats[clave]['valor']
            return {'pu_estimado': pu, 'total_estimado': pu * qty,
                    'grupo_unidad': grupo_u, 'sub_cluster': subcat,
                    'clave_modelo': clave, 'fuente': 'mediana_subcluster'}

        # Mediana del grupo_unidad completo
        vals = [
            s.get('media_pu', s.get('valor', 0.0))
            for s in self.stats.values()
            if s.get('grupo_u') == grupo_u and ('media_pu' in s or 'valor' in s)
        ]
        pu = float(np.median(vals)) if vals else 0.0
        return {'pu_estimado': pu or None, 'total_estimado': (pu * qty) if pu else None,
                'grupo_unidad': grupo_u, 'sub_cluster': subcat,
                'clave_modelo': clave, 'fuente': 'mediana_grupo'}

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_resumen(self) -> None:
        n_xgb = sum(1 for s in self.stats.values() if s['tipo'] == 'xgboost')
        n_med = sum(1 for s in self.stats.values() if s['tipo'] == 'mediana')
        m_g   = self.metricas_globales or {}
        r2_g  = f"{m_g['r2']:.2f}" if m_g.get('r2') is not None else "—"
        print(
            f"\n[XGBoost] ACTIVO ✓ — device={self.device} — "
            f"{n_xgb} claves XGBoost + {n_med} medianas\n"
            f"  Global: MAE=${m_g.get('mae', 0):,.0f}  "
            f"MAPE={m_g.get('mape_pct', 0):.1f}%  R²={r2_g}\n"
        )

        # Agrupar por grupo_u (guardado en stats)
        grupos: dict[str, list] = {}
        for clave, s in self.stats.items():
            grupos.setdefault(s.get('grupo_u', clave), []).append((clave, s))

        for gu, claves in sorted(grupos.items()):
            print(f"  [{gu}]")
            for clave, s in sorted(claves):
                ejems = ', '.join(s.get('desc_ejemplos', [])[:2])
                if s['tipo'] == 'mediana':
                    print(f"    {clave:<30} (n={s['n']:>3}, mediana=${s['valor']:,.0f})")
                else:
                    r2_s = f"{s.get('r2', 0):.2f}" if s.get('r2') is not None else "—"
                    print(
                        f"    {clave:<30} (n={s['n']:>3}, {s.get('cv', '')}) "
                        f"MAE=${s.get('mae', 0):,.0f}  "
                        f"MAPE={s.get('mape_pct', 0):.1f}%  R²={r2_s}"
                    )
                if ejems:
                    print(f"          ↳ {ejems}")

    # ── Compatibilidad ─────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_model(y_true: list, y_pred: list) -> dict:
        return _evaluar(np.asarray(y_true, float), np.asarray(y_pred, float))
