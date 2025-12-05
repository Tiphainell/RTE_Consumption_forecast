import argparse
from utils.config import TrainingConfig
import pandas as pd
from utils.data_processing import prepare_data_set_for_training, create_Y_matrix
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import numpy as np
import os

def train(cfg: TrainingConfig):
    #dataset
    data_power = pd.read_csv(cfg.source_path)
    X_train, y_train, X_test, y_test=prepare_data_set_for_training(data_power,cfg.column_power,cfg.column_time_stamp,cfg.hourly_horizon,cfg.step_per_hour,cfg.split_date)

    #Xg Boost training
    # voir si je change le nombre d'hyperparamètres
    model = xgb.XGBRegressor(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth)
    # Entraîner avec early stopping
    trained_model = MultiOutputRegressor(model).fit(X_train, y_train)

    #enregistrement du modèle
    joblib.dump(trained_model, "../../model/xgb_model.pkl")

    #calcul des prédictions
    y_pred_xgboost_24h = trained_model.predict(X_test)


    #calcul de predictions d'un modèle naif (même valeur il y a une semaine)
    lag_model_naif=672
    _, Y_model_naif = create_Y_matrix(X_test, f"average_imported_power_kw_lag_{lag_model_naif}", cfg.hourly_horizon, cfg.step_per_hour)

    # Créer un dictionnaire pour tout sauvegarder
    predictions_dict = {
        "X_test": X_test,  # DataFrame complet avec colonnes
        "y_pred_model": y_pred_xgboost_24h,  # NumPy array
        "y_pred_naive": Y_model_naif,  # NumPy array
        "y_test": y_test  # DataFrame ou array
    }

    # Sauvegarde avec joblib
    joblib.dump(predictions_dict, "../../predictions/predictions.pkl")

    mean_error_naif = mean_absolute_error(Y_model_naif[:-lag_model_naif, :], y_test.iloc[:-lag_model_naif, :])
    print(f"erreur modèle naif sur tous les horizons {mean_error_naif}")

    mean_error = mean_absolute_error(y_pred_xgboost_24h[:-lag_model_naif, :], y_test.iloc[:-lag_model_naif, :])
    print(f"erreur modèle xgboost sur tous les horizons {mean_error}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="24h prediction curve")


    args = parser.parse_args()

    # YAML loading
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    config_path = project_root / "config" / "config.yaml"

    cfg = TrainingConfig(config_path)



    train(cfg)