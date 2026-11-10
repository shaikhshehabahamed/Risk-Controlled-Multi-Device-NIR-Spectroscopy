"""
gpu_models.py
=============
GPU-accelerated drop-in replacements for the sklearn components used
throughout the NIR sorting pipeline.

Requirements (Windows 11 + CUDA):
    pip install cupy-cuda12x          # match your CUDA version
    # cupyx.scipy.signal is bundled with CuPy – no extra install needed

All public methods accept NumPy arrays as input and return NumPy arrays
as output so that sklearn utilities (Pipeline, CalibratedClassifierCV,
StratifiedGroupKFold, …) continue to work without modification.
Every heavy computation is executed on the GPU via CuPy.

Classes
-------
GPUStandardScaler   – fit/transform on GPU, sklearn-compatible
GPUPCA              – truncated SVD PCA on GPU, sklearn-compatible
GPURidgeClassifier  – weighted multi-class Ridge on GPU, sklearn-compatible

Factory helpers
---------------
build_gpu_ridge_pipeline(alpha)
    Returns sklearn Pipeline([GPUStandardScaler, GPURidgeClassifier])

build_gpu_pca_ridge_pipeline(n_components, alpha)
    Returns sklearn Pipeline([GPUStandardScaler, GPUPCA, GPURidgeClassifier])
"""

from __future__ import annotations

from typing import Any, Union

import cupy as cp
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline


def asnumpy(x) -> np.ndarray:
    """Transfer a CuPy array to a NumPy array on the host."""
    return cp.asnumpy(x)




# ── _GPUClassifierPipeline ────────────────────────────────────────────────────

class _GPUClassifierPipeline(Pipeline):
    """
    Pipeline subclass that explicitly advertises itself as a classifier via both
    the legacy ``_estimator_type`` attribute and the sklearn >= 1.6 tag system.

    Why this is needed
    ------------------
    In sklearn >= 1.6, ``is_classifier()`` reads ``__sklearn_tags__().estimator_type``
    rather than the legacy ``_estimator_type`` property.  ``FrozenEstimator`` delegates
    its own ``__sklearn_tags__`` to the wrapped estimator (Pipeline), but
    ``Pipeline.__sklearn_tags__`` copies only ``target_tags / classifier_tags /
    regressor_tags`` from the final step — it does **not** copy ``estimator_type``.
    The field therefore stays at its default value of ``None``, so
    ``is_classifier(FrozenEstimator(pipeline))`` returns ``False``, causing
    ``CalibratedClassifierCV.predict_proba`` to raise::

        ValueError: FrozenEstimator should either be a classifier …
        Got a regressor with response_method=['decision_function', 'predict_proba']

    By overriding ``__sklearn_tags__`` here we guarantee that ``estimator_type`` is
    set to ``"classifier"`` regardless of how the Pipeline propagation works.
    """

    # Legacy attribute — keeps ``is_classifier()`` working on sklearn < 1.6.
    _estimator_type = "classifier"

    def __sklearn_tags__(self):          # sklearn >= 1.6 tag protocol
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_gpu(X) -> Any:
    """Convert any array-like to the configured array backend (CuPy or NumPy)."""
    return cp.asarray(np.asarray(X), dtype=cp.float64)


def _to_cpu(X_gpu: Any) -> np.ndarray:
    """Transfer a backend array back to a NumPy array on the host."""
    return asnumpy(X_gpu)


# ── GPUStandardScaler ─────────────────────────────────────────────────────────

class GPUStandardScaler(BaseEstimator, TransformerMixin):
    """
    Feature-wise zero-mean / unit-variance scaling computed on GPU.

    Attributes (after fit)
    ----------------------
    mean_  : np.ndarray  shape (n_features,)
    scale_ : np.ndarray  shape (n_features,)
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        X_gpu = _to_gpu(X)
        self._mean_gpu  = X_gpu.mean(axis=0)
        scale           = X_gpu.std(axis=0)
        self._scale_gpu = cp.where(scale < 1e-12, cp.ones_like(scale), scale)
        # Expose numpy copies so sklearn introspection works
        self.mean_  = _to_cpu(self._mean_gpu)
        self.scale_ = _to_cpu(self._scale_gpu)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        X_gpu = _to_gpu(X)
        return _to_cpu((X_gpu - self._mean_gpu) / self._scale_gpu)

    def inverse_transform(self, X) -> np.ndarray:
        X_gpu = _to_gpu(X)
        return _to_cpu(X_gpu * self._scale_gpu + self._mean_gpu)


# ── GPUPCA ────────────────────────────────────────────────────────────────────

class GPUPCA(BaseEstimator, TransformerMixin):
    """
    Principal Component Analysis using full GPU SVD (via CuPy).

    Parameters
    ----------
    n_components : int | float | None
        Integer  → keep exactly that many components.
        Float in (0, 1) → keep enough components to explain that fraction
                          of total variance.
        None     → keep min(n_samples, n_features) components.
    random_state : int  (unused, kept for API parity with sklearn PCA)

    Attributes (after fit)
    ----------------------
    components_               : np.ndarray  shape (n_components_, n_features)
    explained_variance_        : np.ndarray  shape (n_components_,)
    explained_variance_ratio_  : np.ndarray  shape (n_components_,)
    n_components_              : int
    mean_                      : np.ndarray  shape (n_features,)
    """

    def __init__(
        self,
        n_components: Union[int, float, None] = None,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X_gpu = _to_gpu(X)
        n_samples, n_features = X_gpu.shape

        self._mean_gpu = X_gpu.mean(axis=0)
        X_c = X_gpu - self._mean_gpu

        # Full SVD on GPU
        U, S, Vt = cp.linalg.svd(X_c, full_matrices=False)

        # Determine how many components to retain
        total_var = float((S ** 2).sum())
        max_comp  = min(n_samples, n_features)

        if self.n_components is None:
            n_comp = max_comp

        elif isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            # Variance-explained threshold
            cumvar = float(0.0)
            n_comp = max_comp
            threshold = float(self.n_components) * total_var
            for k, s in enumerate(asnumpy(S)):
                cumvar += float(s) ** 2
                if cumvar >= threshold:
                    n_comp = k + 1
                    break

        else:
            n_comp = max(1, min(int(self.n_components), max_comp))

        self._components_gpu = Vt[:n_comp]           # (n_comp, n_features)
        ev = (S[:n_comp] ** 2) / max(n_samples - 1, 1)

        # Store as GPU tensors for fast transform; also expose numpy for sklearn
        self._ev_gpu = ev
        self.n_components_             = int(n_comp)
        self.components_               = _to_cpu(self._components_gpu)
        self.explained_variance_       = _to_cpu(ev)
        self.explained_variance_ratio_ = _to_cpu(ev / max(total_var / max(n_samples - 1, 1), 1e-300))
        self.mean_                     = _to_cpu(self._mean_gpu)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        X_gpu = _to_gpu(X)
        X_c   = X_gpu - self._mean_gpu
        return _to_cpu(X_c @ self._components_gpu.T)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


# ── GPURidgeClassifier ────────────────────────────────────────────────────────

class GPURidgeClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-class Ridge classifier (solves weighted normal equations on GPU).

    The standard-form ridge problem is solved as:
        (Xw^T Xw + α I) coef = Xw^T Yw
    where Xw and Yw are row-wise sqrt-weighted versions of the centred
    design matrix and one-hot target matrix.  The intercept is recovered
    by de-centring.

    Parameters
    ----------
    alpha        : float  L2 regularisation strength (default 1.0)
    class_weight : 'balanced' | None

    Attributes (after fit)
    ----------------------
    classes_    : np.ndarray  sorted unique class labels
    coef_       : np.ndarray  shape (n_classes, n_features)  (sklearn convention)
    intercept_  : np.ndarray  shape (n_classes,)
    """

    def __init__(self, alpha: float = 1.0, class_weight: str | None = "balanced") -> None:
        self.alpha        = alpha
        self.class_weight = class_weight

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        y_arr            = np.asarray(y)
        self.classes_    = np.array(sorted(set(y_arr.tolist())), dtype=y_arr.dtype)
        n_classes        = len(self.classes_)
        label_to_idx     = {lbl: i for i, lbl in enumerate(self.classes_)}
        y_idx            = np.array([label_to_idx[lbl] for lbl in y_arr], dtype=np.int32)

        n_samples, n_features = np.asarray(X).shape

        # Sample weights
        if self.class_weight == "balanced":
            counts   = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
            w_class  = np.where(counts > 0, n_samples / (n_classes * counts), 1.0)
            w_sample = w_class[y_idx]
        else:
            w_sample = np.ones(n_samples, dtype=np.float64)

        X_gpu  = _to_gpu(X)
        w_gpu  = cp.asarray(w_sample, dtype=cp.float64)

        # One-hot encode target on GPU
        Y_gpu  = cp.zeros((n_samples, n_classes), dtype=cp.float64)
        rows   = cp.arange(n_samples)
        cols   = cp.asarray(y_idx, dtype=cp.int32)
        Y_gpu[rows, cols] = 1.0

        # Weighted centring (handles intercept correctly)
        w_sum      = w_gpu.sum()
        X_wmean    = (w_gpu[:, None] * X_gpu).sum(axis=0) / w_sum
        Y_wmean    = (w_gpu[:, None] * Y_gpu).sum(axis=0) / w_sum
        X_c        = X_gpu - X_wmean
        Y_c        = Y_gpu - Y_wmean

        # Scale rows by sqrt(w)
        sqrt_w     = cp.sqrt(w_gpu)
        Xw         = sqrt_w[:, None] * X_c
        Yw         = sqrt_w[:, None] * Y_c

        # Normal equations: (Xw^T Xw + α I) coef = Xw^T Yw
        alpha_gpu  = cp.float64(self.alpha)
        A          = Xw.T @ Xw + alpha_gpu * cp.eye(n_features, dtype=cp.float64)
        B          = Xw.T @ Yw                             # (n_features, n_classes)
        coef_gpu   = cp.linalg.solve(A, B)                 # (n_features, n_classes)
        intcp_gpu  = Y_wmean - X_wmean @ coef_gpu          # (n_classes,)

        # Store in sklearn convention: coef_ shape = (n_classes, n_features)
        self._coef_gpu    = coef_gpu.T
        self._intcp_gpu   = intcp_gpu
        self.coef_        = _to_cpu(self._coef_gpu)
        self.intercept_   = _to_cpu(self._intcp_gpu)
        return self

    # ── decision_function ────────────────────────────────────────────────────

    def decision_function(self, X) -> np.ndarray:
        """Returns raw scores.

        Shape conventions (matching sklearn):
          n_classes == 1  →  (n_samples,)  all zeros (trivial single-class model)
          n_classes == 2  →  (n_samples,)  score of positive class minus negative class
          n_classes  > 2  →  (n_samples, n_classes)
        """
        X_gpu    = _to_gpu(X)
        scores   = X_gpu @ self._coef_gpu.T + self._intcp_gpu   # (n_samples, n_classes)
        scores_np = _to_cpu(scores)
        if len(self.classes_) == 1:
            # Trivial single-class model: always predicts the sole class.
            # Return a zero 1-D array so downstream code sees ndim==1 consistently.
            return np.zeros(scores_np.shape[0], dtype=float)
        if len(self.classes_) == 2:
            return scores_np[:, 1] - scores_np[:, 0]
        return scores_np

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[scores.argmax(axis=1)]


# ── Factory helpers ───────────────────────────────────────────────────────────

def build_gpu_ridge_pipeline(alpha: float = 1.0) -> _GPUClassifierPipeline:
    """
    sklearn Pipeline:  GPUStandardScaler  →  GPURidgeClassifier
    (used in group_split_baseline – no PCA stage)
    """
    return _GPUClassifierPipeline([
        ("scaler", GPUStandardScaler()),
        ("clf",    GPURidgeClassifier(alpha=alpha, class_weight="balanced")),
    ])


def build_gpu_pca_ridge_pipeline(
    n_components: Union[int, float] = 50,
    alpha: float = 1.0,
) -> _GPUClassifierPipeline:
    """
    sklearn Pipeline:  GPUStandardScaler  →  GPUPCA  →  GPURidgeClassifier
    (used in pca_baseline, reliability, freeze_champion_model, …)
    """
    return _GPUClassifierPipeline([
        ("scaler", GPUStandardScaler()),
        ("pca",    GPUPCA(n_components=n_components)),
        ("clf",    GPURidgeClassifier(alpha=alpha, class_weight="balanced")),
    ])
