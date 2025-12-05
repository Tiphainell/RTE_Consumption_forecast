import pandas as pd
from pandas import DataFrame
import numpy as np
from .features_engineering import add_temporal_features


def split_train_test(df: DataFrame, split_date):
    """
        Ajoute une colonne 'set' pour séparer train/test.

        #ici on a 28 mois d'historique, nous gardons 4 mois de tests (derniers mois de 2025) et deux années complètes pour entrainer
        #le modèle (test set ~15%)
        #je ne fais pas de validation set parce que je ne vais pas tester plusieurs modèles

        - Train : les 24 premiers mois
        - Test : les 4 derniers mois (ici à partir du 1er septembre 2025)
        """
    df = df.copy()



    # Créer la colonne 'set' : test à partir du 1er septembre 2025, train sinon
    df["set"] = "train"
    df.loc[df["start_date"] >= split_date, "set"] = "test"

    return df


def create_Y_matrix(df, col: str = "average_imported_power_kw",
                    horizon_hour: int = 24, step_per_hour: int = 4):
    """
    df      : DataFrame avec index temporel
    col     : nom de la colonne contenant les valeurs (ex: 'power')
    horizon_hour : nombre d'heures à prédire
    step_per_hour: nombre de pas par heure (ex: 4 pour 15 min)

    Retourne :
        - df avec de nouvelles colonnes y_1, y_2, ..., y_horizon
        - Y_matrix : np.ndarray shape (n_samples, horizon)
    """
    df = df.copy()
    values = df[col].values
    horizon = horizon_hour * step_per_hour
    n_samples = len(df)

    # Créer la matrice Y initialisée avec NaN
    Y_matrix = np.full((n_samples, horizon), np.nan)

    for i in range(n_samples - horizon):
        Y_matrix[i, :] = values[i + 1: i + 1 + horizon]

    # Ajouter les colonnes y_1, y_2, ... au DataFrame (optionnel)
    for j in range(horizon):
        df[f"y_{j + 1}"] = Y_matrix[:, j]

    return df, Y_matrix


def prepare_data_set_for_training(df: pd.DataFrame,column_power, column_timestamp, horizon, step_per_hour,split_date):
    # Create Y for training
    data_power, Y = create_Y_matrix(df, column_power, horizon, step_per_hour)

    # add temporal features
    data_power = add_temporal_features(data_power, column_timestamp)

    # split train/test
    data_power = split_train_test(data_power,split_date)

    #definition of train and testing set
    train, test = data_power[data_power["set"] == "train"], data_power[data_power["set"] == "test"]

    columns_to_exclude = ["start_date", "set","day_name"]
    columns_y = [c for c in data_power.columns if c.lower().startswith("y_")]
    columns_x = [c for c in data_power.columns if c not in columns_y and c not in columns_to_exclude]

    X_train = train[columns_x]
    y_train = train[columns_y]

    X_test = test[columns_x]
    y_test = test[columns_y]

    return X_train, y_train, X_test, y_test





def prediction_in_production(matrice_pred, window):
    """
    matrice_pred : matrice NxH
    window : nb d'horizons que tu consommes (ex : 4)
    step : pas de réactualisation (ex : 4)
    """

    # On prend les lignes : 0, 4, 8, 12, ...
    rows = np.arange(0, matrice_pred.shape[0], window)

    # On extrait les window premières prédictions
    selected = matrice_pred[rows, :window]  # shape = (len(rows), window)

    # On "aplatit" en 1D
    return selected.reshape(-1)
