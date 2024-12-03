from boston_housing.config import *
from src.models import *
from boston_housing.utils import *
from boston_housing.tests import *

import boston_housing.etl as etl
import boston_housing.plots as plots
import boston_housing.hypotheses as hypotheses

import pickle
import glob
import re
import time
import os
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pickle
import random
from copy import deepcopy
import math
import itertools

from scipy.stats import ttest_1samp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score, log_loss, \
                       confusion_matrix, ConfusionMatrixDisplay,\
                          roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN,Birch,MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterSampler

#import dimension reduction modules
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset

import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_metrics_in_context(y_true, y_pred, model_name, file_path=f"{TXT_OUTDIR}/dt_model_results.txt"):
    # Calculate MSE, MAE, and R²
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate average price for relative error calculations
    avg_price = np.mean(y_true)
    
    # Calculate relative MSE and MAE
    relative_mse = (mse / avg_price) * 100
    relative_mae = (mae / avg_price) * 100
    
    # Print results in context
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    print(f"Relative MSE (% of avg price): {relative_mse:.2f}%")
    print(f"Relative MAE (% of avg price): {relative_mae:.2f}%")
    
    # Append the results to the file
    with open(file_path, "a") as log_file:
        log_file.write(f"\nFor model {model_name}:\n")
        log_file.write(f"Mean Squared Error (MSE): {mse}\n")
        log_file.write(f"Mean Absolute Error (MAE): {mae}\n")
        log_file.write(f"R² Score: {r2}\n")
        log_file.write(f"Relative MSE (% of avg price): {relative_mse:.2f}%\n")
        log_file.write(f"Relative MAE (% of avg price): {relative_mae:.2f}%\n")
        log_file.write("\n" + "="*50 + "\n")
    
    # return mse, mae, r2, relative_mse, relative_mae

# Function to train and evaluate the Decision Tree Regressor with different configurations
def train_and_evaluate_dt(X_train, y_train, X_test, y_test):
    # Initialize models
    dt = DecisionTreeRegressor(random_state=GT_ID)
    bagging = BaggingRegressor(estimator =dt, n_estimators=50, random_state=GT_ID)
    boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=GT_ID)
    
    # GridSearchCV for tuning Decision Tree
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    # Fit models
    models = {
        "Default Decision Tree": dt,
        "Tuned Decision Tree (GridSearch)": grid_search,
        "Bagging with Decision Tree": bagging,
        "Boosting with Decision Tree": boosting
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"For model {model_name}:")
        evaluate_metrics_in_context(y_test, y_pred, model_name)
        plots.plot_predictions_vs_actuals(y_test, y_pred, 
                                          model_name, f"{AGGREGATED_OUTDIR}/pred_actual_diff_{model_name}.png" )
        
        # Calculate losses
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }
    
    return results

# Function to save results to a pickle file
def save_results(results,filename ):

    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def check_etl():
    X, y = etl.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=GT_ID)
    test_data_etl_input_check(X,y,X_train, X_test, y_train, y_test, show = False)
    etl.graph_raw_data(X, y)

    print("======> Data verification complete")
    return X,y,X_train, X_test, y_train, y_test 

###############
def main(): 
    np.random.seed(GT_ID)
    X,y,X_train, X_test, y_train, y_test  = check_etl()
    check_data_info(X, y, X_train, X_test, y_train, y_test, show = False)

    ######
    result_save_file = f"{Y_PRED_OUTDIR}/dt_results.pkl"
    if not os.path.exists(result_save_file):
        results = train_and_evaluate_dt(X_train, y_train, X_test, y_test)
        print("Model Evaluation Results:", results)
        save_results(results, f"{Y_PRED_OUTDIR}/dt_results.pkl")
    else:
        with open(result_save_file, 'rb') as f:
            results = pickle.load(f)
    res_vis_png_path = f"{AGGREGATED_OUTDIR}/dt_results.png" 
    plots.plot_dt_results(results, res_vis_png_path)




if __name__ == "__main__":
    ###################
    print("PyTorch mps check: ",torch.backends.mps.is_available())
    print("PyTorch cuda check: ",torch.cuda.is_available())
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=K_FOLD_CV, shuffle=True, random_state=GT_ID)
    print(f"Torch will be running on {device}")
    ####################
    main()
    