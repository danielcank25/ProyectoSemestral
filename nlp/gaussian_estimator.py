from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class GaussianValuationModel:
    """Estimador de precio unitario basado en distribución Gaussiana ponderada.

    Los pesos corresponden a los scores de similitud coseno del RAG,
    de modo que referencias más cercanas semánticamente influyen más en μ.
    """

    def __init__(self) -> None:
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None

    def fit(self, prices: list[float], weights: list[float]) -> "GaussianValuationModel":
        """Ajusta la distribución a partir de precios históricos y pesos de similitud."""
        p = np.asarray(prices, dtype=float)
        w = np.asarray(weights, dtype=float)

        if p.size == 0:
            raise ValueError("Se requiere al menos un precio histórico.")
        if p.size != w.size:
            raise ValueError("prices y weights deben tener la misma longitud.")

        w = np.clip(w, 0, None)
        w_sum = w.sum()
        if w_sum == 0:
            w = np.ones_like(w)
            w_sum = float(w.size)

        self.mu = float(np.dot(w, p) / w_sum)
        variance = float(np.dot(w, (p - self.mu) ** 2) / w_sum)
        self.sigma = float(np.sqrt(variance)) if variance > 0 else 0.0
        return self

    def predict(self) -> dict[str, float]:
        """Retorna μ y σ del ajuste actual."""
        if self.mu is None:
            raise RuntimeError("Llama a fit() antes de predict().")
        return {"pu_estimado": self.mu, "margen_error": self.sigma}

    @staticmethod
    def evaluate_model(
        y_true: list[float], y_pred: list[float]
    ) -> dict[str, float]:
        """Calcula MAE, RMSE, MAPE y R² entre valores reales y predichos."""
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        if yt.size == 0 or yt.size != yp.size:
            raise ValueError("y_true y y_pred deben tener la misma longitud y al menos un elemento.")

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        nonzero = yt != 0
        mape = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100) if nonzero.any() else 0.0

        r2: Optional[float] = None
        if yt.size > 1:
            r2_val = float(r2_score(yt, yp))
            r2 = None if (r2_val != r2_val) else r2_val  # descarta NaN

        return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}
