import socket

# Determine the hostname
hostname = socket.gethostname()
if hostname == "Khais-MacBook-Pro.local" or hostname == "Khais-MBP.attlocal.net":  
    from boston_housing.config_mac import *  
else:
    from boston_housing.config_cuda import * 

import os

BOSTON_RAW_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'raw.txt')
BOSTON_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'boston_data.pkl')
DATASET_SELECTION = "boston"

EVAL_FUNC_METRIC = 'accuracy' #'f1' # 'accuracy' 
EVAL_MODELS = [
                # 'default',
                'MPL',
                'CNN', 
                'LSTM', 
                'bi-LSTM',
                'conv-LSTM', 
                #'seg-gru',
                ]




from pathlib import Path
def set_output_dir(path):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    return path
# Get the root project directory (the parent directory of boston_housing)
project_root = Path(__file__).resolve().parent.parent
# Define the output directory path relative to the project root
OUTPUT_DIR_A3 = project_root / 'outputs' / DATASET_SELECTION
DRAFT_VER_A3 = 1
# Set the directories using set_output_dir
AGGREGATED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/aggregated_graphs')
Y_PRED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/y_pred_graphs')
CV_LOSSES_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/pkl_cv')
TXT_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/txt_stats')
OUTPUT_DIR_RAW_DATA_A3 =set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/raw_data_assessments')


#ML PARAMS
K_FOLD_CV = 5