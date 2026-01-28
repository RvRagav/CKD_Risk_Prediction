from __future__ import annotations

# Locked semantic groupings (derived from dataset content)
CONT_COLS = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
ORD_COLS = ['sg', 'al', 'su']
CAT_COLS = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
TARGET_RAW_COL = 'classification'
TARGET_COL = 'target'

YESNO_COLS = ['htn', 'dm', 'cad', 'pe', 'ane']
NORMAL_ABNORMAL_COLS = ['rbc', 'pc']
PRESENT_COLS = ['pcc', 'ba']

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Paths (relative to repo root)
RAW_DATA_PATH = 'dataset/kidney_disease.csv'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

CLEANED_DATA_PATH = 'data/processed/kidney_disease_cleaned.csv'
SPLIT_DIR = 'data/processed/splits'
PREPROC_DIR = 'data/processed/preprocessed'
SYNTH_DIR = 'data/synthetic'
RESULTS_DIR = 'results'

# Simple physiologic bounds (used for clipping, not as medical advice)
CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
	'sg': (1.005, 1.030),
	'bp': (50.0, 200.0),
	'sc': (0.4, 15.0),
	'sod': (110.0, 170.0),
	'pot': (2.5, 7.5),
}

# Continuous/ordinal features that tend to dominate realism + calibration.
HIGH_RISK_NUMERIC = ['sc', 'sod', 'sg', 'bp', 'al', 'su']

# Counterfactual constraints (paper-aligned)
# Features that should never be suggested to change.
NON_ACTIONABLE_COLS = ['age', 'cad']

# Monotonic directions used for realistic counterfactual constraints.
# Values: 'increase' | 'decrease'
MONOTONIC_COLS = {
	'sc': 'decrease',
	'hemo': 'increase',
	'bp': 'decrease',
}
