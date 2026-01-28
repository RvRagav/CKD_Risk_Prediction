from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Allow running as: python src/cleaning.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CAT_COLS,
    CLINICAL_BOUNDS,
    CONT_COLS,
    ORD_COLS,
    NORMAL_ABNORMAL_COLS,
    PRESENT_COLS,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    RANDOM_STATE,
    SPLIT_DIR,
    TARGET_COL,
    TARGET_RAW_COL,
    TEST_SIZE,
    YESNO_COLS,
    CLEANED_DATA_PATH,
    PREPROC_DIR,
)
from src.utils import ensure_dir, save_json  # noqa: E402


def model_based_impute(train_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Feature-wise regression imputation (paper-aligned).

    Trains per-feature regressors on *train_df only* and imputes missing values
    in *full_df*. Categorical predictors are one-hot encoded using categories
    learned from train_df to avoid leakage.
    """

    imputed = full_df.copy()

    def _make_design_matrices(
        X_train_raw: pd.DataFrame,
        X_apply_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # One-hot encode categoricals using train-only learned columns.
        # Keep NaNs as their own category so missingness can be predictive.
        train_dum = pd.get_dummies(X_train_raw, dummy_na=True)
        apply_dum = pd.get_dummies(X_apply_raw, dummy_na=True)
        apply_dum = apply_dum.reindex(columns=train_dum.columns, fill_value=0)

        # Any remaining NaNs (from numeric predictors) are filled using train-only medians.
        medians = train_dum.median(numeric_only=True)
        train_dum = train_dum.fillna(medians)
        apply_dum = apply_dum.fillna(medians)
        return train_dum, apply_dum

    for col in CONT_COLS + ORD_COLS:
        if col not in imputed.columns:
            continue

        missing_mask = imputed[col].isna()
        if int(missing_mask.sum()) == 0:
            continue

        predictors = [p for p in imputed.columns if p != col]

        train_rows = train_df.dropna(subset=[col])
        if train_rows.empty:
            continue

        X_train_raw = train_rows[predictors]
        y_train = pd.to_numeric(train_rows[col], errors='coerce')

        # If y_train has NaNs after coercion, drop them.
        keep = y_train.notna()
        if not bool(keep.all()):
            X_train_raw = X_train_raw.loc[keep]
            y_train = y_train.loc[keep]
        if y_train.empty:
            continue

        X_missing_raw = imputed.loc[missing_mask, predictors]

        X_train, X_missing = _make_design_matrices(X_train_raw, X_missing_raw)

        model = LinearRegression()
        model.fit(X_train, y_train)
        imputed.loc[missing_mask, col] = model.predict(X_missing)

    return imputed


def apply_clinical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip key numeric columns to simple physiologic bounds.

    This reduces tail explosions and keeps synthesis/training more stable.
    """

    out = df.copy()
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').clip(lower=lo, upper=hi)
    return out


def _strip_and_nan(s: pd.Series) -> pd.Series:
    # Strip whitespace/tabs; convert '?' and string 'nan' into NaN
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({'?': np.nan, 'nan': np.nan, 'None': np.nan, '': np.nan})
    return s2


def load_and_tidy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop id if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Strip whitespace/tabs in object columns & replace '?' with NaN
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = _strip_and_nan(df[c])

    # Fix target stray tab/whitespace
    if TARGET_RAW_COL in df.columns:
        df[TARGET_RAW_COL] = df[TARGET_RAW_COL].astype(str).str.strip()

    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = CONT_COLS + ORD_COLS
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Yes/No columns
    for c in YESNO_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.lower()
            # handle common tab/space variants even after strip
            s = s.replace({'\tyes': 'yes', '\tno': 'no', 'yes ': 'yes', 'no ': 'no'})
            df[c] = pd.to_numeric(s.map({'yes': 1, 'no': 0}), errors='coerce')

    # rbc and pc: normal/abnormal (locked mapping)
    for c in NORMAL_ABNORMAL_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.lower()
            s = s.replace({'?': np.nan, 'nan': np.nan})
            df[c] = pd.to_numeric(s.map({'normal': 0, 'abnormal': 1}), errors='coerce')

    # pcc and ba: present/notpresent
    for c in PRESENT_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.lower()
            s = s.replace({'?': np.nan, 'nan': np.nan})
            df[c] = pd.to_numeric(s.map({'notpresent': 0, 'present': 1}), errors='coerce')

    # appet: normalize whitespace/case (kept as categorical for one-hot)
    if 'appet' in df.columns:
        df['appet'] = df['appet'].astype(str).str.strip().str.lower().replace({'?': np.nan, 'nan': np.nan})

    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_RAW_COL not in df.columns:
        raise ValueError(f"Expected '{TARGET_RAW_COL}' column")

    # IMPORTANT: do NOT use substring 'ckd' in 'notckd'
    y_raw = df[TARGET_RAW_COL].astype(str).str.strip().str.lower()
    y = y_raw.map({'ckd': 1, 'notckd': 0})

    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    df[TARGET_COL] = y.loc[mask].astype(int)

    df = df.drop(columns=[TARGET_RAW_COL])
    return df


def build_preprocessor(cont_cols: list[str], ord_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    cont_pipe = Pipeline([
        ('scaler', StandardScaler()),
    ])

    # sklearn changed arg name from sparse->sparse_output
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe),
    ])

    used_num_cols = [c for c in (cont_cols + ord_cols) if c is not None]
    used_cat_cols = [c for c in cat_cols if c is not None]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', cont_pipe, used_num_cols),
            ('cat', cat_pipe, used_cat_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    if hasattr(preprocessor, 'get_feature_names_out'):
        return list(preprocessor.get_feature_names_out())

    # Fallback for very old sklearn
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder' and trans == 'drop':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            try:
                out = trans.get_feature_names_out(cols)
            except TypeError:
                out = trans.get_feature_names_out()
            names.extend(list(out))
        else:
            names.extend(list(cols))
    return names


def _to_dense(matrix) -> np.ndarray:
    # ColumnTransformer/OneHotEncoder can produce sparse matrices depending on sklearn version.
    if hasattr(matrix, 'toarray'):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description='CKD cleaning + preprocessing (locked test set).')
    parser.add_argument('--raw-path', type=str, default=RAW_DATA_PATH)
    parser.add_argument('--test-size', type=float, default=TEST_SIZE)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = load_and_tidy(args.raw_path)
    df = coerce_numeric(df)
    df = normalize_categoricals(df)
    df = apply_clinical_bounds(df)
    df = map_target(df)

    ensure_dir(Path(CLEANED_DATA_PATH).parent)
    df.to_csv(CLEANED_DATA_PATH, index=False)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    # Save raw splits
    split_dir = ensure_dir(SPLIT_DIR)
    X_train.to_csv(split_dir / 'X_train_raw.csv', index=False)
    X_test.to_csv(split_dir / 'X_test_raw.csv', index=False)
    y_train.to_csv(split_dir / 'y_train.csv', index=False, header=True)
    y_test.to_csv(split_dir / 'y_test.csv', index=False, header=True)

    # Impute AFTER split (train-only learned), on raw feature space.
    # Learn imputers from the *observed* training split to avoid leakage.
    X_train_learn = X_train.copy()
    X_train = model_based_impute(X_train_learn, X_train)
    X_test = model_based_impute(X_train_learn, X_test)

    # Optional: clip again after imputation to ensure physiologic plausibility.
    X_train = apply_clinical_bounds(X_train)
    X_test = apply_clinical_bounds(X_test)

    # Save raw-imputed splits for counterfactual / clinical interpretability work
    X_train.to_csv(split_dir / 'X_train_imputed_raw.csv', index=False)
    X_test.to_csv(split_dir / 'X_test_imputed_raw.csv', index=False)

    # Enforce no remaining missing for numeric model features.
    numeric_cols = [c for c in (CONT_COLS + ORD_COLS) if c in X_train.columns]
    if X_train[numeric_cols].isna().any().any() or X_test[numeric_cols].isna().any().any():
        raise ValueError(
            'Model-based imputation left missing values in numeric columns. '
            'Check columns with all-missing values or incompatible dtypes.'
        )

    # Build + fit preprocessor on TRAIN only
    cont_cols = [c for c in CONT_COLS if c in X_train.columns]
    ord_cols = [c for c in ORD_COLS if c in X_train.columns]
    cat_cols = [c for c in CAT_COLS if c in X_train.columns]

    preprocessor = build_preprocessor(cont_cols, ord_cols, cat_cols)
    preprocessor.fit(X_train)

    # Transform
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_proc = _to_dense(X_train_proc)
    X_test_proc = _to_dense(X_test_proc)

    feature_names = get_feature_names(preprocessor)

    preproc_dir = ensure_dir(PREPROC_DIR)
    pd.DataFrame(X_train_proc, columns=feature_names).to_csv(preproc_dir / 'X_train_preproc.csv', index=False)
    pd.DataFrame(X_test_proc, columns=feature_names).to_csv(preproc_dir / 'X_test_preproc.csv', index=False)
    save_json(feature_names, preproc_dir / 'feature_names.json')

    # Save fitted preprocessor
    ensure_dir(Path(PREPROCESSOR_PATH).parent)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print('Saved cleaned dataset:', CLEANED_DATA_PATH)
    print('Saved raw splits to:', split_dir)
    print('Saved preprocessed splits to:', preproc_dir)
    print('Saved preprocessor to:', PREPROCESSOR_PATH)


if __name__ == '__main__':
    main()
