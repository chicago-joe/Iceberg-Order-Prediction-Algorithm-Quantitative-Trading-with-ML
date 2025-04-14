import os, optuna, pendulum, neptune, shap, flaml
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError
from multiprocessing import freeze_support

from optuna.samplers import TPESampler

os.environ['NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZjNlODVlNC04NzYwLTQ5ZWItYTUyYy1kY2EwNjBmYjFjODcifQ=="
os.environ['NEPTUNE_PROJECT'] = "sbuser/SX3M"

from flaml.automl import AutoML
import neptune_lightgbm as nptlgbm_utils
from joblib import dump, load
# from optuna_fast_fanova import FanovaImportanceEvaluator
import neptune_xgboost as nptxgb_utils
from prefect import flow, task, get_run_logger
from pathlib2 import Path
import pandas as pd
from sxmpy.connections import create_mongodb_query
import seaborn as sns
import matplotlib.pyplot as plt
from neptune.utils import stringify_unsupported, StringifyValue
from datetime import timedelta
from neptune.types import File
import numpy as np
import neptune.integrations.sklearn as npt_utils
from tqdm.auto import tqdm, trange
from dask.distributed import Client
from lightgbm.dask import DaskLGBMClassifier
from xgboost.dask import DaskXGBClassifier, DaskXGBRFClassifier
# from xgboost_ray import RayXGBClassifier, RayXGBRFClassifier
from datetime import time

# np.random.Generator()
random_state = np.random.RandomState(seed=42)
import neptune.integrations.optuna as nptopt_utils
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

#todo: implement plotting
# from lightgbm.plotting import create_tree_digraph, plot_tree, plot_importance
# from xgboost.plotting import plot_tree, plot_importance, to_graphviz
# from xgboost.sklearn import XGBClassifier, XGBRFClassifier
# from lightgbm.sklearn import LGBMClassifier

from random import randint
import joblib
from itertools import product
from typing import List
import numpy as np

from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sxmpy.common import setPandas
import warnings
from sxmpy.common import convert_nanoseconds_to_dttm

setPandas(3)
sns.set_style('darkgrid')
idx = pd.IndexSlice
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------------------------------------------------

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )

# ------------------------------------------------------------------------------------------------------------------------------------------
# HyperparameterTuner class

class HyperparameterTuner:

    hyperparameter_set_pct_size = None
    hyperparameter_set_dates = None
    hyperopt_X_dataset, hyperopt_y_dataset = [], []

    def __init__(self, model_evaluator, hyperparameter_set_pct_size):
        self.model_evaluator = model_evaluator
        self.run = model_evaluator.run
        self.hyperparameter_set_pct_size = hyperparameter_set_pct_size

        self.hyperopt_X_train_agg = {name: pd.DataFrame() for name in self.model_evaluator.model_names}
        self.hyperopt_y_train_agg = {name: [] for name in self.model_evaluator.model_names}
        self.hyperopt_X_test_agg = {name: pd.DataFrame() for name in self.model_evaluator.model_names}
        self.hyperopt_y_test_agg = {name: [] for name in self.model_evaluator.model_names}
        self.hyperopt_y_pred_agg = {name: [] for name in self.model_evaluator.model_names}

        # get unique dates only used for hyperopt
        self._get_hyperparameter_set_dates()
        
        
    def _get_hyperparameter_set_dates(self):
        self.hyperparameter_set_dates = self.model_evaluator.unique_split_dates[:round(len(self.model_evaluator.unique_split_dates) * self.hyperparameter_set_pct_size)]

        # get total hyperopt_X_dataset and hyperopt_y_dataset
        self.hyperopt_X_dataset = self.model_evaluator.X_dataset.query("tradeDate.isin(@self.hyperparameter_set_dates)")
        self.hyperopt_y_dataset = self.model_evaluator.y_dataset.to_frame().query("tradeDate.isin(@self.hyperparameter_set_dates)").T.stack(-1).reset_index(
                level=0, drop=True, name='mdExec').rename('mdExec')

        # log train test for entire run to neptune metadata
        self.run['hyperparameters_set_pct_size'] = self.hyperparameter_set_pct_size
        self.run["ndays_hyperparameter_set"] = len(self.hyperparameter_set_dates)
        self.run["ndays_validation_set"] = len(self.model_evaluator.validation_set_dates)
        self.run['hyperparameter_set_dates'] = self.hyperparameter_set_dates
        self.run["validation_set_dates"] = self.model_evaluator.validation_set_dates

        return

    def _save_data(trial_number, x_test, x_y, x_preds):
    # You could also save to a database, write to a file, etc.
        joblib.dump((x_test, x_y, x_preds), f"trial_data_{trial_number}.pkl")

    def get_model_hyperparameters(self, trial, model_name):
        # Define hyperparameters for the given model

        if model_name == "Random Forest":
            return {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 250, 500, 1000]),
                'criterion': 'log_loss',
                'max_features':None,
                'max_depth': trial.suggest_int('max_depth', 2, 4, step=1),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 10, step=1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 5, step=1),
                'bootstrap': True,
                'oob_score': False
            }
        if model_name == "XGBoost":
            return {
                'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'error@0.7', 'error@0.5']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, step=0.01),
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 250, 500, 1000]),
                'max_depth': trial.suggest_int('max_depth', 3, 5, step=1),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 10, step=1),
                'gamma': trial.suggest_float('gamma', 0.1, 0.2, step=0.05),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.2, step=0.1),
                'reg_lambda': trial.suggest_int('reg_lambda', 1, 3, step=1)
            }
        if model_name == 'XGBoost RF':
            return {
                'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'error@0.7', 'error@0.5']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, step=0.01),
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 250, 500, 1000]),
                'max_depth': trial.suggest_int('max_depth', 3, 5, step=1),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 10, step=1),
                'gamma': trial.suggest_float('gamma', 0.1, 0.2, step=0.05),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.8, 1.0, step=0.1),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.2, step=0.1),
                'reg_lambda': trial.suggest_int('reg_lambda', 1, 3, step=1),
                'num_parallel_tree': trial.suggest_categorical('num_parallel_tree', [50, 100])
            }
        if model_name == 'LightGBM':
            return {
                'objective': trial.suggest_categorical('objective', ['binary','regression']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, step=0.01),
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 250, 500, 1000]),
                'max_depth': trial.suggest_int('max_depth', 3, 5, step=1),
                'num_leaves': trial.suggest_categorical('num_leaves', [2,3,7,15,31]),
                'min_sum_hessian_in_leaf': trial.suggest_categorical('min_sum_hessian_in_leaf', [0.001, 0.01, 0.1, 1, 10]),
                'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 25, 100, step=25),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6,1.0,step=0.2),  # Also known as colsample_bytree in some models
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6,1.0,step=0.2),
                'bagging_freq': trial.suggest_categorical('bagging_freq',[0, 5, 10]),
                'lambda_l1': trial.suggest_categorical('lambda_l1',[0, 1e-1, 1, 2]),
                'lambda_l2': trial.suggest_categorical('lambda_l2',[0, 1e-1, 1, 2]),
                'min_gain_to_split': trial.suggest_categorical('min_gain_to_split', [0, 0.1, 0.5]),
                # Add other parameters here
            }
        if model_name == "Logistic Regression":
                return {
                    'penalty': trial.suggest_categorical('penalty',['l1', 'l2', 'elasticnet']),
                    'C': trial.suggest_categorical('C', [0.01, 0.1, 1, 10, 100]),
                    'solver': trial.suggest_categorical('solver', ['saga']),  # 'saga' supports L1, L2, and Elastic-Net penalties
                    'max_iter': trial.suggest_categorical('max_iter', [100, 500, 1000]),
                    'l1_ratio': trial.suggest_categorical('l1_ratio', [0, 0.5, 1])  # Only relevant if penalty is 'elasticnet'
                }



    # Optimize model hyperparameters and train size
    def objective(self, trial, model, model_name):

        model_params = self.get_model_hyperparameters(trial, model_name)
        model.set_params(**model_params)

        self.hyperopt_y_pred_agg[model_name] = []
        self.hyperopt_y_test_agg[model_name] = []

        train_size = trial.suggest_categorical('train_size', [2, 3, 4, 5, 6, 7, 8, 9, 10])
        logging_df = pd.DataFrame(columns=["train_dates","test_dates"]).assign(train_size=train_size)

        # for train_dates, test_dates in tqdm(self.model_evaluator.generate_splits(train_size)):
        n = 0
        self.run["model"][model_name]["tuning"]["trials"][str(trial.number)][f"split_{str(n)}"] = {}
        self.run["model"][model_name]["tuning"]["trials"][str(trial.number)][f"split_{str(n)}/train_dates"] = []
        self.run["model"][model_name]["tuning"]["trials"][str(trial.number)][f"split_{str(n)}/test_dates"] = []

        for train_dates, test_dates in tqdm(self.model_evaluator.generate_splits([train_size], self.hyperparameter_set_dates)):

            self.run["model"][model_name]["tuning"]["trials"][str(trial.number)][f"split_{str(n)}/train_dates"].append(train_dates)
            self.run["model"][model_name]["tuning"]["trials"][str(trial.number)][f"split_{str(n)}/test_dates"].append(test_dates)
            n += 1

            if self.model_evaluator.test_size < 2:
                q = "tradeDate==@test_dates"
            else:
                q = "tradeDate.isin(@test_dates)"

            hyperopt_X_train, hyperopt_y_train = (
                self.hyperopt_X_dataset.query("tradeDate.isin(@train_dates)"),
                self.hyperopt_y_dataset.to_frame().query(f"tradeDate.isin(@train_dates)").T.stack(-1).reset_index(
                level=0, drop=True, name='mdExec').rename('mdExec')
            )
            hyperopt_X_test, hyperopt_y_test = (
                self.hyperopt_X_dataset.query("tradeDate.isin(@test_dates)"),
                self.hyperopt_y_dataset.to_frame().query(f"tradeDate.isin(@test_dates)").T.stack(-1).reset_index(
                level=0, drop=True, name='mdExec').rename('mdExec')
            )

            # Train and validate the model using model_params
            model.fit(hyperopt_X_train, hyperopt_y_train)
            hyperopt_y_pred = model.predict(hyperopt_X_test)

            self.hyperopt_y_test_agg[model_name] += hyperopt_y_test.tolist()
            self.hyperopt_y_pred_agg[model_name] += hyperopt_y_pred.tolist()


        # calculate and accumulate the score
        score = self.model_evaluator.max_precision_optimal_recall_score(self.hyperopt_y_test_agg[model_name], self.hyperopt_y_pred_agg[model_name])
        return score


    def tune(self, n_trials, seed=None):
        for model, model_name in zip(self.model_evaluator.models, self.model_evaluator.model_names):
            if model_name == "Dummy":
                pass
            else:
                neptune_callback = nptopt_utils.NeptuneCallback(run=self.run["model"][model_name]["tuning"], plots_update_freq=1)

                # Make the sampler behave in a deterministic way.
                if seed is not None:
                    sampler = TPESampler(seed=seed)
                    study = optuna.create_study(direction="maximize", sampler=sampler)
                else:
                    study = optuna.create_study(direction="maximize")

                study.optimize(lambda trial: self.objective(trial, model, model_name),
                               n_trials=n_trials,
                               callbacks=[neptune_callback, logging_callback],
                               show_progress_bar=True
                               )

                # Save and log best parameters
                self.model_evaluator.best_params[model_name] = study.best_params
                self.run["model"][model_name]["hyperoptimized_best_params"] = study.best_params

                # importance = optuna.importance.get_param_importances(
                # study, evaluator=FanovaImportanceEvaluator()
                # )
                # print(importance)
        self.run.wait()

# ------------------------------------------------------------------------------------------------------------------------------------------
# ModelEvaluator class

class ModelEvaluator:
    dataset = None
    test_size = 1
    feature_names = []
    unique_split_dates = []
    
    X_dataset, y_dataset = [], []
    X_train, y_train, X_test, y_test = [], [], [], []
    validation_set_dates = []

    hyperparameter_set_pct_size = 0.5
    hyperparameter_set_dates = []

    def train(self):
        for model, name in zip(self.models, self.model_names):
            if name == "Dummy":  # Skip dummy model or any specific model if necessary
                continue

            # Logging the training process
            print(f"Training model: {name}")

            # Train the model on the aggregated training data
            
            # Splitting the dataset for dates greater than 2023-09-01
            mask = self.dataset.index > '2023-09-01'
            X_train, X_test = self.dataset.loc[~mask], self.dataset.loc[mask]
            y_train, y_test = self.y_dataset.loc[~mask], self.y_dataset.loc[mask]
            self.X_train_agg[name] = X_train
            self.y_train_agg[name] = y_train
            self.X_test_agg[name] = X_test
            self.y_test_agg[name] = y_test

            self.y_train_agg[name] = np.concatenate(self.y_train_agg[name], axis=0)  # Concatenate all training labels

            model.set_params(**self.best_params[name])
            model.fit(self.X_train_agg[name], self.y_train_agg[name])  # Training the model

            # Predict on the aggregated test data
            self.X_test_agg[name] = pd.concat(self.X_test_agg[name], axis=0)  # Concatenate all test data
            self.y_test_agg[name] = np.concatenate(self.y_test_agg[name], axis=0)  # Concatenate all test labels

            y_pred = model.predict(self.X_test_agg[name])

            # Store the predictions and actual values for later evaluation
            self.y_pred_agg[name] = y_pred

            score = self.max_precision_optimal_recall_score(self.y_test_agg[name], self.y_pred_agg[name])
            print(f'Score for {name}: {score}')
            
            self.evaluate_model(name)  # Assuming you have a method that evaluates each model individually

            # Optionally, log and visualize the model's performance
            # self.log_model_performance(name)  # You can create a method to log the performance to Neptune or elsewhere
            # self.visualize_model_performance(name)  # Similarly, create a method for visualization

        print("Training complete.")

    def __init__(self, models, model_names, random_state):
        model_keys = {
            "Dummy": "DUM",
            "Logistic Regression": "LR",
            "Random Forest": "RF",
            "XGBoost": "XG",
            "XGBoost RF": "XGRF",
            "LightGBM": "LGBM",
        }

        self.random_state = random_state
        self.models = models
        self.model_names = model_names
        self.models_metadata = {}  # Store model metadata

        # to initialize storage for feature importances
        self.feature_importances = {name: [] for name in model_names if name != 'Dummy'}  # Added this line
        self.mda_importances = {name: [] for name in model_names[1:]}  # Store MDA importances, excluding the dummy model
        self.shap_values = {name: [] for name in model_names[1:]}  # Store SHAP values, excluding the dummy model

        self.X_train_agg = {name: pd.DataFrame() for name in model_names}
        self.y_train_agg = {name: [] for name in model_names}
        self.X_test_agg = {name: pd.DataFrame() for name in model_names}
        self.y_test_agg = {name: [] for name in model_names}
        self.y_pred_agg = {name: [] for name in model_names}

        self.best_params = {name: {} for name in model_names}  # Store the best parameters for each model
        self.tuned_models = {name: None for name in model_names}  # Store the tuned models
        self.partial_dependences = {name: [] for name in model_names}

        # initialize new neptune run
        self.run = neptune.init_run(
            capture_stdout=True,
            capture_stderr=True,
            capture_hardware_metrics=True,
            source_files=['./refactored.py'],
            mode='sync'  # comment this out when done debugging
        )
        # ------------------------------------------------------------------------------------------------------------------------------------------
        # instantiate neptune run for new model versions

        for estimator, name in zip(self.models, self.model_names):
            mkey = model_keys.get(name)

            # Check if the model already exists in Neptune
            try:
                model = neptune.init_model(key=mkey)
                print("Creating a new model version...")
                model_version = neptune.init_model_version(model=model["sys/id"].fetch())

            except NeptuneModelKeyAlreadyExistsError:
                print(f"A model with the provided key {mkey} already exists in this project.")
                print("Creating a new model version...")
                model_version = neptune.init_model_version(
                    model=neptune.Project().fetch_models_table().to_pandas().query("`sys/id`.str.split('-',expand=True)[1]==@mkey")[
                        "sys/id"].values[0]
                )

            model_version.wait()
            string_params = stringify_unsupported(npt_utils.get_estimator_params(estimator))

            if "missing" in string_params.keys():
                string_params.pop("missing")

            model_version['estimator/params'] = string_params
            model_version['estimator/class'] = str(estimator.__class__)
            model_version.wait()
            model_version.sync()

            self.run[f"model/{name}"] = model_version.fetch()
            self.run.wait()

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # UTILITY FUNCTIONS

    @staticmethod
    def max_precision_optimal_recall_score(y_true, y_pred):
        """
        This is a custom scoring function that maximizes precision while optimizing to the best possible recall.
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        min_recall = 0.5
        score = 0 if recall < min_recall else precision
        return score

    def _create_time_series_splits(self, train_size, dates):
        splits = []
        n = len(dates)

        for i in range(n):
            if i + train_size < n:
                train_dates = dates[i:i + train_size]
                test_dates = [dates[i + train_size]]
                splits.append((train_dates, test_dates))

        return splits

    def generate_splits(self, train_sizes, dates):
        all_splits = []

        for size in train_sizes:
            splits = self._create_time_series_splits(train_size=size, dates=dates)
            all_splits.extend(splits)

        return all_splits

    def ingest_dataset(self, df=None, target_variable='mdExec', split_on_variable="tradeDate"):
        """
        Ingests the dataset and splits it into train and test sets.

        Parameters:
            df: The dataset to be ingested.
            target_variable: The target variable to be predicted.
            split_on_variable: The variable to be used for splitting the dataset into train and test sets.

        Returns:
            The train and test sets.
        """

        self.dataset = df.copy()
        if split_on_variable in self.dataset.columns:
            self.dataset.set_index(split_on_variable, inplace=True)
        else:
            self.feature_names = [col for col in self.dataset.columns if col != target_variable]
            self.unique_split_dates = sorted(self.dataset.index.unique().tolist())

            # Split the dataset into train and test sets
            self.X_dataset = self.dataset[self.feature_names]
            self.y_dataset = self.dataset[target_variable]
        
        return

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # HYPEROPT FUNCTIONS

    def tune_models(self, n_trials=25, hyperparameter_set_pct_size=0.5, seed = None):
        self.hyperparameter_set_pct_size = hyperparameter_set_pct_size
        tuner = HyperparameterTuner(self, self.hyperparameter_set_pct_size)

        # set validation dates (testing of final best parameters from hyperparameter opts)
        self.hyperparameter_set_dates = sorted(tuner.hyperparameter_set_dates)
        self.validation_set_dates = sorted(list(set(self.unique_split_dates) - set(self.hyperparameter_set_dates)))

        tuner.tune(n_trials, seed=seed)
        return

    def update_models_with_best_params(self, best_params):
        for name in self.model_names:
            if name in best_params:
                # Reinitialize the model with the best parameters
                if name == 'Random Forest':
                    self.models[self.model_names.index(name)] = RandomForestClassifier(**best_params[name])
                elif name == 'XGBoost':
                    self.models[self.model_names.index(name)] = XGBClassifier(**best_params[name])
                # Add other models as needed
        return

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUATION  FUNCTIONS
    def evaluate_models_custom_scorer(self):
        custom_scorer_function = make_scorer(self.max_precision_optimal_recall_score, greater_is_better=True)
        custom_scores = {}

        for model, name in zip(self.models, self.model_names):
            y_test = self.y_test_agg[name]
            y_pred = model.predict(self.X_test_agg[name])
            custom_score = custom_scorer_function(model, y_test, y_pred)
            custom_scores[name] = custom_score

        return custom_scores

    def train(self):
        for model, name in zip(self.models, self.model_names):
            if name == "Dummy":  # Skip dummy model or any specific model if necessary
                continue

            # Logging the training process
            print(f"Training model: {name}")

            # Train the model on the aggregated training data
            
            # Splitting the dataset for dates greater than 2023-09-01
            mask = self.dataset.index > '2023-09-01'
            X_train, X_test = self.dataset.loc[~mask], self.dataset.loc[mask]
            y_train, y_test = self.y_dataset.loc[~mask], self.y_dataset.loc[mask]
            self.X_train_agg[name] = X_train
            self.y_train_agg[name] = y_train
            self.X_test_agg[name] = X_test
            self.y_test_agg[name] = y_test

            self.y_train_agg[name] = np.concatenate(self.y_train_agg[name], axis=0)  # Concatenate all training labels

            model.set_params(**self.best_params[name])
            model.fit(self.X_train_agg[name], self.y_train_agg[name])  # Training the model

            # Predict on the aggregated test data
            self.X_test_agg[name] = pd.concat(self.X_test_agg[name], axis=0)  # Concatenate all test data
            self.y_test_agg[name] = np.concatenate(self.y_test_agg[name], axis=0)  # Concatenate all test labels

            y_pred = model.predict(self.X_test_agg[name])

            # Store the predictions and actual values for later evaluation
            self.y_pred_agg[name] = y_pred

            score = self.max_precision_optimal_recall_score(self.y_test_agg[name], self.y_pred_agg[name])
            print(f'Score for {name}: {score}')
            
            self.evaluate_model(name)  # Assuming you have a method that evaluates each model individually

            # Optionally, log and visualize the model's performance
            # self.log_model_performance(name)  # You can create a method to log the performance to Neptune or elsewhere
            # self.visualize_model_performance(name)  # Similarly, create a method for visualization

        print("Training complete.")

    # Add other methods as necessary, e.g., for evaluation, logging, and visualization

# Rest of the existing code...
    def predict_and_aggregate(self, X_test, y_test):
        for model, name in zip(self.models, self.model_names):
            y_pred = model.predict(X_test)

            self.y_test_agg[name] += y_test.tolist()
            self.X_test_agg[name] = pd.concat([self.X_test_agg[name], X_test], axis=0)
            self.y_pred_agg[name] += y_pred.tolist()

            if name != 'Dummy':
                self.feature_importances[name].append(model.feature_importances_)  # Store feature importances in each iteration

                mda_importances = self.calculate_mda(model, X_test, y_test)
                self.mda_importances[name].append(mda_importances)

                self.calculate_shap(model, X_test, name)

                #todo: calculate partial dependence
                # selected_features = [f for f, _ in sorted(self.mda_importances[name][-1].items(), key=lambda x: x[1], reverse=True)[:3]]
                # self.calculate_and_store_partial_dependence(model, X_test, name, selected_features)
        return


    def evaluate_model(self):
        WTF = {}

        for model, name in zip(self.models, self.model_names):
            # create and log classification summary
            if name != "Logistic Regression":
                self.run[f"cls_summary/{name}"] = stringify_unsupported(npt_utils.create_classifier_summary(
                    model, self.X_train_agg[name], self.X_test_agg[name], pd.Series(self.y_train_agg[name]), pd.Series(self.y_test_agg[name]),
                    log_charts=True
                ))

            WTF[name] = {
                'Accuracy': accuracy_score(self.y_test_agg[name], self.y_pred_agg[name]),
                'F1 Score': f1_score(self.y_test_agg[name], self.y_pred_agg[name]),
                'Precision': precision_score(self.y_test_agg[name], self.y_pred_agg[name]),
                'Recall': recall_score(self.y_test_agg[name], self.y_pred_agg[name]),
                'Confusion Matrix': confusion_matrix(self.y_test_agg[name], self.y_pred_agg[name])
            }
            # Log metrics to Neptune
            for metric, value in WTF[name].items():
                self.run[f"results/{name}/{metric}"] = value
                self.run.wait()
        return

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # STATISTICAL  FUNCTIONS

    def calculate_mda(self, model, X_test, y_test, scoring=f1_score):
        """
        MDA (Mean Decrease Accuracy):
        This is a technique where the importance of a feature is evaluated by permuting the values of the feature
        and measuring the decrease in model performance.
        The idea is that permuting the values of an important feature should lead to a significant drop in model performance,
        indicating the feature's importance.
        """
        base_score = scoring(y_test, model.predict(X_test))
        feature_importances = {}

        for feature in X_test.columns:
            X_copy = X_test.copy()
            X_copy[feature] = np.random.permutation(X_copy[feature].values)
            new_score = scoring(y_test, model.predict(X_copy))
            feature_importances[feature] = base_score - new_score

        return feature_importances


    def get_mean_feature_importances(self):
        mean_feature_importances = {
            name: np.mean(self.feature_importances[name], axis=0) for name in self.feature_importances
        }
        return mean_feature_importances, self.feature_names  # Modified to return feature names as well


    def calculate_shap(self, model, X_test, model_name):
        """
        SHAP (SHapley Additive exPlanations):
        This is another technique for interpreting machine learning models, and it is based on game theory.
        SHAP values provide a unified measure of feature importance that is fairly allocated among features,
        ensuring consistent and accurate interpretations.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        self.shap_values[model_name].append(shap_values)


    def calculate_and_store_partial_dependence(self, model, X_test, model_name, features):
        pd_results = partial_dependence(model, X_test, features, grid_resolution=50,
                                        kind='average',
                                        # method='brute'
                                        )
        if model_name not in self.partial_dependences:
            self.partial_dependences[model_name] = []
        self.partial_dependences[model_name].append(pd_results)


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # PLOTTING / VISUAL FUNCTIONS

    def plot_aggregated_shap(self):
        for model_name, shap_values_list in self.shap_values.items():
            aggregated_shap_values = np.mean(shap_values_list, axis=0)

            shap.summary_plot(aggregated_shap_values, feature_names=self.feature_names, plot_type="bar", max_display=12, show=False)
            plt.title(f'Aggregated SHAP Feature Importances for {model_name}')
            plt.savefig(f"aggregated_shap_importances_{model_name}.png")
            plt.show()

            self.run[f"aggregated_shap_importances/{model_name}"] = File.from_path(f"aggregated_shap_importances_{model_name}.png")


    def plot_aggregated_partial_dependence(self):
        """
        Partial Dependence Plots:
        These plots show the effect of a single feature on the predicted outcome of a machine learning model,
        holding all other features constant.
        It helps to understand the relationship between the response variable and a feature.
        """
        for model_name, pd_results_list in self.partial_dependences.items():
            # Calculate mean partial dependence
            averaged_pd_results = np.mean([result.average for result in pd_results_list], axis=0)

            fig, ax = plt.subplots(figsize=(10, 8))
            # plot = PartialDependenceDisplay(
            #     pd_results_list[0].display.axes_,
            #     averaged_pd_results,
            #     pd_results_list[0].features
            # )
            display = PartialDependenceDisplay.from_results(
                pd_results_list[0].features,
                averaged_pd_results,
                pd_results_list[0].feature_names,
                ax=ax  # Pass the axes object to the display
            )

            fig.suptitle(f'Aggregated Partial Dependence Plots for {model_name}', y=1.02)
            fig.subplots_adjust(top=0.9)  # Adjust title position

            self.run[f"partial_dependency/{model_name}/aggregated_partial_dependence_plot"] = self.run.upload(fig)

            plt.show()
            return


    def plot_feature_importances(self):
        mean_feature_importances, feature_names = self.get_mean_feature_importances()

        # Plot the top 12 features
        for model_name, importances in mean_feature_importances.items():
            sorted_indices = np.argsort(importances)[::-1][:12]

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.title(f"Top 12 Feature Importances for {model_name}")
            ax.bar(range(12), importances[sorted_indices], align="center")
            ax.xticks(range(12), [feature_names[i] for i in sorted_indices], rotation=45)  # Use the feature names for x-ticks
            fig.tight_layout()  # Adjust layout for better visibility

            # self.run["model"][model_name]["mean_feature_importances"] = mean_feature_importances[model_name]
            # self.run[f"mean_feature_importances/{model_name}"] = File.from_path(f'mean_feature_importances_{model_name}.png')

            # upload to neptune
            self.run[f"feature_importance/{model_name}/mean_feature_importance"] = mean_feature_importances[model_name]
            self.run[f"feature_importance/{model_name}/mean_feature_importance_plot"] = self.run.upload(fig)

            plt.show()
            return

    def plot_mda_importances(self, top_n=12):
        for model_name, importances_list in self.mda_importances.items():
            # Calculate mean MDA importances across all iterations
            mean_mda_importances = {
                feature: np.mean([imp[feature] for imp in importances_list])
                for feature in self.feature_names
            }

            # Sort features by importance
            sorted_features = sorted(mean_mda_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(features)), importances, align='center')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Reverse the order for better readability

            ax.xlabel('Mean Decrease Accuracy')
            ax.title(f'MDA Feature Importances for {model_name}')
            fig.tight_layout()  # Adjust layout for better visibility

            # upload to neptune
            self.run[f"mda_importances/{model_name}/mean_mda_importances"] = mean_mda_importances
            self.run[f"mda_importances/{model_name}/mda_importance_plot"] = self.run.upload(fig)

            plt.show()
            return

    def plot_roc_curves(self):
        """
        The plot_roc_curves method computes the ROC curve and ROC area for each class and then plots them.
        The roc_curve function computes the false positive rate and true positive rate, which are then used to plot the ROC curve.
        The auc function computes the area under the ROC curve.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for name in self.model_names:
            fpr, tpr, _ = roc_curve(self.y_test_agg[name],
                                    self.models[self.model_names.index(name)].predict_proba(self.X_test_agg[name])[:, 1])

            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

            # Log ROC AUC to Neptune
            # self.run[f"metrics/{name}/ROC_AUC"] = roc_auc
            # self.run["model"][name]["results/ROC_AUC"] = roc_auc
            self.run[f"metrics/{name}/ROC_AUC"] = roc_auc

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")

        # plt.savefig('roc_curve.png')
        # self.run["compare"]["roc_curve"] = File.from_path('roc_curve.png')

        self.run[f"metrics/roc_auc_plot"] = self.run.upload(fig)
        plt.show()

        return

    def plot_precision_recall_curves(self):
        """
        The plot_precision_recall_curves method computes the precision-recall curve for each class and then plots them.
        The precision_recall_curve function computes the precision and recall, which are then used to plot the precision-recall curve.
        The average_precision_score function computes the average precision, a summary metric of the precision-recall curve.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for name in self.model_names:
            precision, recall, _ = precision_recall_curve(self.y_test_agg[name],
                                                          self.models[self.model_names.index(name)].predict_proba(self.X_test_agg[name])[:,
                                                          1])
            average_precision = average_precision_score(self.y_test_agg[name],
                                                        self.models[self.model_names.index(name)].predict_proba(self.X_test_agg[name])[:,
                                                        1])
            ax.plot(recall, precision, label=f'{name} (average precision = {average_precision:.2f})')

            self.run[f"metrics/{name}/average_precision_score"] = average_precision

        ax.xlabel('Recall')
        ax.ylabel('Precision')
        ax.title('Precision-Recall Curve')
        ax.legend(loc="upper right")

        self.run[f"metrics/precision_recall_curve"] = self.run.upload(fig)

        plt.show()
        return

    def stop_neptune_run(self):
        self.run.stop()
        for model_name, metadata in self.models_metadata.items():          # stop models as well
            metadata["model"].stop()
        return

# ------------------------------------------------------------------------------------------------------------------------------------------
# preprocess data
def preprocess_data(df=None):
    # todo: tmp fix for ifus data
    # update_dates = df.tradeDate.unique()[-10:]
    # for dt in update_dates:
    #     df.loc[df.tradeDate == dt, 'tradeDate'] = pd.to_datetime(
    #         pendulum.from_timestamp(pd.to_datetime(dt).timestamp()).subtract(days=23).timestamp(), unit="s").date()

    df = df.set_index('tradeDate').sort_index()
    df.drop(columns=['symbol'], inplace=True)
    df = df.applymap(pd.to_numeric)

    # have to make this 0 for xgb
    target = 'mdExec'
    df[target].where(df[target] != -1, 0, inplace=True)

    y = df[target]
    class_names = y.unique().astype(str)
    X = df.drop([c for c in df.columns if c.startswith('mdExec')], axis=1)

    #todo: add these back??
    # drop unused feature cols for optimal performance
    colsdrop1, colsdrop2 = X.filter(like='waitingToOrder').columns, X.filter(like='orderPlaced').columns

    X.drop(columns=colsdrop1, inplace=True)
    X.drop(columns=colsdrop2, inplace=True)

    # return X, y, class_names
    return pd.concat([X.sort_index(axis=1), y], axis=1)

# ------------------------------------------------------------------------------------------------------------------------------------------
# flatten json to columns
def flatten_to_columns(df=None, flatten_cols=['waitingToOrder','orderPlaced','oneStateBeforeFill','orderFilled']):
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # FLATTEN COLUMNS
    lstFlattened = []

    for col in flatten_cols:

        dicts = df[col].to_list()

        # ------------------------------------------------------------------------------------------------------------------------------------------
        # TradeImbalance

        # Now we can expand each dictionary into its own DataFrame
        expanded = [pd.json_normalize(d['TradeImbalance']) for d in dicts]
        flattened_df = pd.concat([df.T.stack() for df in expanded], axis=1).T
        # flattened_df = pd.concat([df.T.stack().reset_index(level=1,drop=True) for df in expanded], axis=1).T
        flattened_df = flattened_df.reset_index(drop=True)

        # Define custom mapping according to your requirement.
        flattened_df.columns=flattened_df.columns.set_levels(levels=["90sec","100msg","60sec","30sec","50msg","300sec","1000msg"], level=1)
        flattened_df=flattened_df.stack().T.unstack().unstack(1).unstack()

        # Rename a level of multi-index by joining tuple components
        # flattened_df.columns = flattened_df.columns.map('_'.join)
        flattened_df.columns=flattened_df.columns.map("_".join).to_series().apply(lambda x:f"{col}_"+x).tolist()
        # print(flattened_df.head())
        filtered_df=flattened_df.drop(pd.concat([flattened_df.filter(like="_isTimeBased"),flattened_df.filter(like="_period"),flattened_df.filter(like="timeInterval")],axis=1).columns,axis=1)
        # print(filtered_df)

        # filtered_df.to_csv("TradeImbalance.csv",index=False)

        # ------------------------------------------------------------------------------------------------------------------------------------------
        # Indicators - Qty

        expandedQty = [pd.json_normalize(d['Qty']) for d in dicts]
        flattened_dfQty = pd.concat([df.T.stack().reset_index(level=1,drop=True) for df in expandedQty], axis=1).T
        flattened_dfQty = flattened_dfQty.reset_index(drop=True)

        flattened_dfQty.columns=flattened_dfQty.columns.to_series().apply(lambda x:f"{col}_"+x).tolist()
        # print(flattened_dfQty.head())

        # filtered_dfQty=flattened_dfQty.drop(pd.concat([flattened_dfQty.filter(like="_useQty"),flattened_dfQty.filter(like="EventTime"),flattened_dfQty.filter(like="ExchangeTime")],axis=1).columns,axis=1)
        filtered_dfQty=flattened_dfQty.drop(pd.concat([flattened_dfQty.filter(like="_useQty")],axis=1).columns,axis=1)
        filtered_dfQty.drop(pd.concat([
            filtered_dfQty.filter(like="_highLowRange"),
            filtered_dfQty.filter(like="_numReloads"),
            filtered_dfQty.filter(like="_fillToDisplayRatio"),
            filtered_dfQty.filter(like="_deltaHighLowRange"),
            filtered_dfQty.filter(like="_deltaNumReloads"),
            filtered_dfQty.filter(like="_deltaFillToDisplayRatio")
        ], axis=1).columns, axis=1, inplace=True)
        filtered_dfQty.columns=filtered_dfQty.columns.to_series().apply(lambda x:x+"Qty").tolist()

        # ------------------------------------------------------------------------------------------------------------------------------------------
        # Indicators - NumOrders

        expandedNumOrders = [pd.json_normalize(d['NumOrders']) for d in dicts]
        flattened_dfNumOrders = pd.concat([df.T.stack().reset_index(level=1, drop=True) for df in expandedNumOrders], axis=1).T
        flattened_dfNumOrders = flattened_dfNumOrders.reset_index(drop=True)

        flattened_dfNumOrders.columns = flattened_dfNumOrders.columns.to_series().apply(lambda x: f"{col}_" + x).tolist()

        filtered_dfNumOrders = flattened_dfNumOrders.drop(pd.concat([flattened_dfNumOrders.filter(like="_useQty")], axis=1).columns, axis=1)
        for c in ["_highLowRange", "_numReloads", "_fillToDisplayRatio"]:
            filtered_df[f"{col}{c}"] = filtered_dfNumOrders[f"{col}{c}"].astype(float)
            filtered_dfNumOrders.drop(columns=[f"{col}{c}"], inplace=True)
        filtered_dfNumOrders.drop(pd.concat(
            [filtered_dfNumOrders.filter(like="_deltaHighLowRange"), filtered_dfNumOrders.filter(like="_deltaNumReloads"),
             filtered_dfNumOrders.filter(like="_deltaFillToDisplayRatio")], axis=1).columns, axis=1, inplace=True)
        filtered_dfNumOrders.columns = filtered_dfNumOrders.columns.to_series().apply(lambda x: x + "NumOrders").tolist()

        # append to list
        lstFlattened.append(pd.concat([filtered_df, filtered_dfQty, filtered_dfNumOrders], axis=1))

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Final Table for ML
    # assemble final table for ML

    dftable = pd.concat([df.loc[:, 'symbol':'oneStateBeforeFill'].drop(columns=flatten_cols), pd.concat(lstFlattened,axis=1)], axis=1)
    return dftable

# calculate months to expiry
def months_to_exp(dttm=None):
    # get 3rd week of month
    expDttm = pendulum.parse(dttm).add(weeks=3)
    periodInterval = pendulum.period(expDttm, pendulum.today()).as_interval()
    months_to_expiry = periodInterval.total_weeks()*-1/4
    return months_to_expiry


@flow
def get_data_from_mongodb():

    result = create_mongodb_query("sxmdb",
                                "mktSim_analytics.icebergsimulation.IcebergSimulation",
                                {
#                                    "archiveDate":{"$gte":"20230811"},
                                    "orderPlaced":{"$exists":True},"symbol":{"$nin":['HO','CL','GC','SI','HG']},"archiveDate":{"$lt":"20230901"}
                                    # "orderFilled":{"$exists":True}
                                })
    result.drop(columns=['_id','hostId','hostName','archiveDate','icebergId','price','volume','showSize','filledSizeEnd','simFinishedBecauseOfTrade',
                                                                    'icebergGoneFromPriceChange','icebergDetectedFromPriceChange','throwAway','missedCancel','tickSize','initialBookQty','currentQueuePosition','initialQueuePosition',
                                                                   'orderFilled',], inplace=True)
    df = flatten_to_columns(result, flatten_cols=["oneStateBeforeFill","waitingToOrder","orderPlaced"])

    # 1 column ticksFromLow if bid and ticksFromHigh if offer and column 2 = ticksFromLow if offer and ticksFromHigh if bid
    df['ticksFromSupportLevel'] = np.where(df.isBid==True,df['ticksFromLow'],df['ticksFromHigh'])
    df['ticksFromResistanceLevel'] = np.where(df.isBid!=True,df['ticksFromLow'],df['ticksFromHigh'])
    df.drop(columns=['ticksFromHigh','ticksFromLow'],inplace=True)

    # ### convert bidAsk imbalances to side and 1-side
    bidImbalanceCols= df.filter(like="bidImbalance").columns

    for col in bidImbalanceCols:
        df[col.replace("bid","sameSide")] = np.where(df.isBid==True,df[col], 1-df[col])

    bidaskImbalances = pd.concat([df.filter(like="bidImbalance"),df.filter(like="askImbalance")],axis=1)

    df.drop(columns=bidaskImbalances.columns,inplace=True)

    # calculate months to expiry
    df['monthsToExpiry'] = df['expirationMonthYear'].apply(months_to_exp)

    # convert nanoseconds to datetime
    df['eventTime']=df.eventTime.apply(convert_nanoseconds_to_dttm)

    # Extract the date from the "eventTime" column
    df['tradeDate'] = df['eventTime'].dt.date

    # Extract just the time from the "eventTime" column
    df['eventTime_only_time'] = df['eventTime'].dt.time
    # Define the time 4:30 PM for comparison
    time_430pm = time(16, 30, 0)

    # Create a mask where the event time is past 4:30 PM
    mask_past_430pm = df['eventTime_only_time'] > time_430pm

    # Increase the "tradeDate" by 1 day where the event time is past 4:30 PM
    df.loc[mask_past_430pm, 'tradeDate'] = (pd.to_datetime(df.loc[mask_past_430pm, 'tradeDate']) + timedelta(days=1)).dt.date

    # Remove the temporary column used for time comparison
    df.drop(columns=['eventTime_only_time'], inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(columns=['expirationMonthYear','isBid','highLowRange','eventTime'],inplace=True)

    with pd.HDFStore(f"/hpdata.h5", "w") as f:
        f.put("/processed",df)

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# run script
@flow()
def HyperParameters_flow():
    setPandas(3)
    logger = get_run_logger()

    #todo: change this when done testing
    random_state = None

    ## if using dask:
    # freeze_support()
    # client = Client(n_workers=16, threads_per_worker=2, memory_limit="2GB")

    get_data_from_mongodb()

    # ---------------------------------------------------------------------------------
    # Load the dataset

    with pd.HDFStore('/hpdata.h5', 'r') as store:
        df = store.get("processed").dropna()

    dataset = preprocess_data(df)

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize models

    dummy_clf = DummyClassifier(
        strategy='stratified',
        random_state=random_state
    )

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        criterion='log_loss',
        max_features=None,
        max_depth=10,  # Decreased
        min_samples_split=10,  # Increased
        min_samples_leaf=5,  # Increased
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    # XGBoost
    xgb_clf = XGBClassifier(
        enable_categorical=True,
        max_depth=4,  # Decreased
        learning_rate=0.05,  # Decreased
        n_estimators=500,
        objective='binary:logistic',
        booster='gbtree',
        # eval_metric='logloss',
        eval_metric='error@0.7',
        n_jobs=-1,
        gamma=0.2,  # Increased
        min_child_weight=10,  # Increased
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,  # Increased
        reg_lambda=2,  # Increased
        random_state=random_state,
        verbose=0
    )

    # XGBoost RF
    xgbrf_clf = XGBRFClassifier(
        enable_categorical=True,
        colsample_bynode=0.8,
        learning_rate=0.1,  # Decreased
        subsample=0.8,
        n_estimators=500,
        objective='binary:logistic',
        base_score=None,
        booster="gbtree",
        colsample_bytree=1.0,
        # eval_metric="logloss",
        eval_metric="error@0.7",
        gamma=0.2,  # Increased
        max_depth=4,  # Decreased
        min_child_weight=10,  # Increased
        n_jobs=-1,
        num_parallel_tree=100,  # Be careful with this
        random_state=random_state,
        reg_alpha=0.2,  # Increased
        reg_lambda=2,  # Increased
        verbosity=0
    )

    lgbm_clf = LGBMClassifier(
        random_state=random_state, deterministic=True, force_col_wise=True, #change when ready
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        # min_data_in_leaf=
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=1,
        lambda_l2=1,
        min_gain_to_split=0.1,
        num_leaves=31,
        verbose=0,
    )
    lr_clf = LogisticRegression(
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # init ModelEvaluator

    evaluator = ModelEvaluator(
        models=[dummy_clf, lr_clf, rf_clf, xgb_clf, xgbrf_clf, lgbm_clf],
        model_names=['Dummy', "Logistic Regression", 'Random Forest', 'XGBoost', "LightGBM",],
        random_state=random_state
    )
    evaluator.ingest_dataset(df=dataset, target_variable='mdExec', split_on_variable="tradeDate")
    # todo: remove seed when done testing
    evaluator.tune_models(n_trials=50, hyperparameter_set_pct_size=1, seed=random_state)
    evaluator.stop_neptune_run()
@flow()
def HyperParameters_flow():
    setPandas(3)
    logger = get_run_logger()

    #todo: change this when done testing
    random_state = None

    ## if using dask:
    # freeze_support()
    # client = Client(n_workers=16, threads_per_worker=2, memory_limit="2GB")

    get_data_from_mongodb()

    # ---------------------------------------------------------------------------------
    # Load the dataset

    with pd.HDFStore('/hpdata.h5', 'r') as store:
        df = store.get("processed").dropna()

    dataset = preprocess_data(df)

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize models

    dummy_clf = DummyClassifier(
        strategy='stratified',
        random_state=random_state
    )

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        criterion='log_loss',
        max_features=None,
        max_depth=10,  # Decreased
        min_samples_split=10,  # Increased
        min_samples_leaf=5,  # Increased
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    # XGBoost
    xgb_clf = XGBClassifier(
        enable_categorical=True,
        max_depth=4,  # Decreased
        learning_rate=0.05,  # Decreased
        n_estimators=500,
        objective='binary:logistic',
        booster='gbtree',
        # eval_metric='logloss',
        eval_metric='error@0.7',
        n_jobs=-1,
        gamma=0.2,  # Increased
        min_child_weight=10,  # Increased
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,  # Increased
        reg_lambda=2,  # Increased
        random_state=random_state,
        verbose=0
    )

    # XGBoost RF
    xgbrf_clf = XGBRFClassifier(
        enable_categorical=True,
        colsample_bynode=0.8,
        learning_rate=0.1,  # Decreased
        subsample=0.8,
        n_estimators=500,
        objective='binary:logistic',
        base_score=None,
        booster="gbtree",
        colsample_bytree=1.0,
        # eval_metric="logloss",
        eval_metric="error@0.7",
        gamma=0.2,  # Increased
        max_depth=4,  # Decreased
        min_child_weight=10,  # Increased
        n_jobs=-1,
        num_parallel_tree=100,  # Be careful with this
        random_state=random_state,
        reg_alpha=0.2,  # Increased
        reg_lambda=2,  # Increased
        verbosity=0
    )

    lgbm_clf = LGBMClassifier(
        random_state=random_state, deterministic=True, force_col_wise=True, #change when ready
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        # min_data_in_leaf=
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=1,
        lambda_l2=1,
        min_gain_to_split=0.1,
        num_leaves=31,
        verbose=0,
    )
    lr_clf = LogisticRegression(
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # init ModelEvaluator

    evaluator = ModelEvaluator(
        models=[dummy_clf, lr_clf, rf_clf, xgb_clf, xgbrf_clf, lgbm_clf],
        model_names=['Dummy', "Logistic Regression", 'Random Forest', 'XGBoost', "LightGBM",],
        random_state=random_state
    )
    evaluator.ingest_dataset(df=dataset, target_variable='mdExec', split_on_variable="tradeDate")
    # todo: remove seed when done testing
    evaluator.tune_models(n_trials=50, hyperparameter_set_pct_size=1, seed=random_state)
    evaluator.stop_neptune_run()


@flow(log_prints=True)
def IcebergLearn():
    setPandas(3)
    logger = get_run_logger()

    #todo: change this when done testing
    random_state = None

    ## if using dask:
    # freeze_support()
    # client = Client(n_workers=16, threads_per_worker=2, memory_limit="2GB")

    get_data_from_mongodb()

    # ---------------------------------------------------------------------------------
    # Load the dataset

    with pd.HDFStore('/hpdata.h5', 'r') as store:
        df = store.get("processed").dropna()

    dataset = preprocess_data(df)

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize models

    dummy_clf = DummyClassifier(
        strategy='stratified',
        random_state=random_state
    )

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        criterion='log_loss',
        max_features=None,
        max_depth=10,  # Decreased
        min_samples_split=10,  # Increased
        min_samples_leaf=5,  # Increased
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    # XGBoost
    xgb_clf = XGBClassifier(
        enable_categorical=True,
        max_depth=4,  # Decreased
        learning_rate=0.05,  # Decreased
        n_estimators=500,
        objective='binary:logistic',
        booster='gbtree',
        # eval_metric='logloss',
        eval_metric='error@0.7',
        n_jobs=-1,
        gamma=0.2,  # Increased
        min_child_weight=10,  # Increased
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,  # Increased
        reg_lambda=2,  # Increased
        random_state=random_state,
        verbose=0
    )

    lgbm_clf = LGBMClassifier(
        random_state=random_state, deterministic=True, force_col_wise=True, #change when ready
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=4,
        # min_data_in_leaf=
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=1,
        lambda_l2=1,
        min_gain_to_split=0.1,
        num_leaves=31,
        verbose=0,
    )
    lr_clf = LogisticRegression(
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # init ModelEvaluator

    evaluator = ModelEvaluator(
        models=[dummy_clf, lr_clf, rf_clf, xgb_clf, lgbm_clf],
        model_names=['Dummy', "Logistic Regression", 'Random Forest', 'XGBoost', "LightGBM",],
        random_state=random_state
    )
    evaluator.ingest_dataset(df=dataset, target_variable='mdExec', split_on_variable="tradeDate")
    # todo: remove seed when done testing
    evaluator.tune_models(n_trials=50, hyperparameter_set_pct_size=1, seed=random_state)
    evaluator.stop_neptune_run()


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # WORK IN PROGRESS

    # for train, test in tqdm(splits):
    #     print(f"Training Dates: {train}, Testing Date: {test}")
    #     if test_size < 2:
    #         q = "tradeDate==@test"
    #     else:
    #         q = "tradeDate.isin(@test)"
    #
    #     X_train, y_train = X.query("tradeDate.isin(@train)"), y.to_frame().query(f"tradeDate.isin(@train)").T.stack(-1).reset_index(
    #         level=0, drop=True, name='mdExec').rename('mdExec')
    #
    #     X_test, y_test = X.query(q), y.to_frame().query(q).T.stack(-1).reset_index(
    #         level=0, drop=True, name='mdExec').rename('mdExec')
    #
    #     # Train models
    #     evaluator.fit(X_train, y_train, hyperparameter_tuning=True)
    #     evaluator.predict_and_aggregate(X_test, y_test)

    # evaluator.evaluate_model()

    # feature_importances = evaluator.get_mean_feature_importances()
    # feature_names = selected_features
    # evaluator.plot_roc_curves()
    # evaluator.plot_precision_recall_curves()
    # evaluator.plot_feature_importances()
    # evaluator.plot_mda_importances()
    # # todo:
    # # evaluator.plot_aggregated_partial_dependence()
    # evaluator.plot_aggregated_shap()
    #
    # evaluator.stop_neptune_run()
if __name__ == "__main__":
    HyperParameters_flow()