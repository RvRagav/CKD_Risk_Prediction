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
TEST_SIZE = 0.20

# Paths (relative to repo root)
RAW_DATA_PATH = 'dataset/kidney_disease.csv'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

CLEANED_DATA_PATH = 'data/processed/kidney_disease_cleaned.csv'
SPLIT_DIR = 'data/processed/splits'
PREPROC_DIR = 'data/processed/preprocessed'
SYNTH_DIR = 'data/synthetic'
RESULTS_DIR = 'results'
