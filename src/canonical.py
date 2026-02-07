from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Canonical feature set (base-paper). DO NOT EDIT.
CANONICAL_FEATURES: list[str] = ['hemo', 'sc', 'al', 'htn', 'age', 'dm']
CANONICAL_NUMERIC: list[str] = ['hemo', 'sc', 'al', 'age']
CANONICAL_FLAGS: list[str] = ['htn', 'dm']


def assert_canonical_schema(X: pd.DataFrame) -> None:
    """Hard fail if columns deviate from canonical schema."""
    assert list(X.columns) == CANONICAL_FEATURES, (
        'Schema mismatch: expected canonical features/order.\n'
        f'Expected: {CANONICAL_FEATURES}\n'
        f'Got     : {list(X.columns)}'
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _coerce_binary_flag(series: pd.Series) -> pd.Series:
    """Coerce common CKD dataset encodings to {0,1} with NaNs preserved.

    Accepts numeric 0/1, strings like 'yes'/'no', '1'/'0', 'true'/'false'.
    Any other value becomes NaN and will be imputed by the preprocessor.
    """

    if pd.api.types.is_bool_dtype(series):
        out = series.astype('Int64')
    elif pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors='coerce')
    else:
        s = series.astype(str).str.strip().str.lower()
        s = s.replace({'?': np.nan, 'nan': np.nan, 'none': np.nan, '': np.nan})
        mapped = s.map({
            'yes': 1,
            'no': 0,
            'y': 1,
            'n': 0,
            'true': 1,
            'false': 0,
            '1': 1,
            '0': 0,
            '1.0': 1,
            '0.0': 0,
        })
        out = pd.to_numeric(mapped, errors='coerce')

    # Anything not in {0,1} becomes NaN
    out = out.where(out.isin([0, 1]), np.nan)
    return out


@dataclass
class CanonicalPreprocessor:
    """Canonical preprocessing for base-paper feature space.

    - Selects ONLY canonical features
    - Binary-encodes clinical flags: dm, htn ∈ {0,1}
    - Imputes missing: numeric -> median, flags -> mode (fallback 0)
    - Preserves numeric scale (no StandardScaler)
    - Enforces strict column order
    """

    numeric_fill_: dict[str, float] | None = None
    flag_fill_: dict[str, int] | None = None

    def fit(self, X_raw: pd.DataFrame) -> 'CanonicalPreprocessor':
        X = self._select_and_coerce(X_raw, allow_extra=True)

        numeric_fill: dict[str, float] = {}
        for col in CANONICAL_NUMERIC:
            med = float(np.nanmedian(X[col].to_numpy(dtype=float))) if col in X.columns else float('nan')
            if np.isnan(med):
                med = 0.0
            numeric_fill[col] = med

        flag_fill: dict[str, int] = {}
        for col in CANONICAL_FLAGS:
            s = X[col]
            mode = s.dropna().mode()
            if mode.empty:
                flag_fill[col] = 0
            else:
                v = int(mode.iloc[0])
                flag_fill[col] = 1 if v == 1 else 0

        self.numeric_fill_ = numeric_fill
        self.flag_fill_ = flag_fill
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_fill_ is None or self.flag_fill_ is None:
            raise RuntimeError('CanonicalPreprocessor not fitted. Call fit() first.')

        X = self._select_and_coerce(X_raw, allow_extra=True)

        # Impute numeric
        for col, fill in self.numeric_fill_.items():
            X[col] = _coerce_numeric(X[col]).fillna(fill).astype(float)

        # Impute flags and force {0,1} ints
        for col, fill in self.flag_fill_.items():
            X[col] = _coerce_binary_flag(X[col]).fillna(fill)
            X[col] = X[col].astype(int)
            bad = set(pd.unique(X[col])) - {0, 1}
            if bad:
                raise ValueError(f"{col} has non-binary values after preprocessing: {sorted(bad)}")

        # Strict order and strict schema
        X = X[CANONICAL_FEATURES].copy()
        assert_canonical_schema(X)
        return X

    def fit_transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_raw).transform(X_raw)

    @staticmethod
    def _select_and_coerce(X_raw: pd.DataFrame, allow_extra: bool = True) -> pd.DataFrame:
        if not isinstance(X_raw, pd.DataFrame):
            raise TypeError('X_raw must be a pandas DataFrame')

        missing = [c for c in CANONICAL_FEATURES if c not in X_raw.columns]
        if missing:
            raise ValueError(f'Missing canonical columns: {missing}')

        X = X_raw.copy()

        # Coerce numerics first to prevent get_dummies/OHE-like expansions elsewhere.
        for col in CANONICAL_NUMERIC:
            X[col] = _coerce_numeric(X[col]).astype(float)

        for col in CANONICAL_FLAGS:
            X[col] = _coerce_binary_flag(X[col])

        # Drop any non-canonical columns explicitly
        if allow_extra:
            X = X[CANONICAL_FEATURES].copy()
        else:
            # still enforce strict
            assert_canonical_schema(X)

        return X


def ensure_canonical_dataframe(
    X_raw: pd.DataFrame,
    preprocessor: CanonicalPreprocessor | None,
) -> tuple[pd.DataFrame, CanonicalPreprocessor]:
    """Utility: fit if needed, then return canonical X with strict schema."""
    if preprocessor is None:
        preprocessor = CanonicalPreprocessor().fit(X_raw)
    X = preprocessor.transform(X_raw)
    return X, preprocessor


def forbid_onehot_residuals(columns: list[str]) -> None:
    """Guardrail: reject common one-hot artifacts for dm/htn."""
    banned = {
        'dm_0.0', 'dm_1.0', 'dm_0', 'dm_1',
        'htn_0.0', 'htn_1.0', 'htn_0', 'htn_1',
    }
    bad = [c for c in columns if c in banned]
    if bad:
        raise ValueError(f'Found forbidden one-hot encoded columns: {bad}')
