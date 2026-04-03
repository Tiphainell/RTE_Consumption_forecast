import pandas as pd
from pandas import DataFrame
import numpy as np
from .features_engineering import add_temporal_features
from glob import glob
import os

def format_rte_files(file_path: str)->DataFrame:
    data_power = pd.read_csv(file_path, sep=",", header=None)
    #create date column
    block_size = 100
    n = len(data_power) // block_size

    data_power = create_date_column(data_power,n,block_size)
    data_power.rename(columns={0: "Heure", 1: "Previsions J-1", 2: "Previsions J", 3: "Consommation"}, inplace=True)
    #removing firt lines of each block :
    # 1 day -> 1 block : 100 lines

    to_remove = np.concatenate([np.array([0, 1, 98, 99]) + i * block_size for i in range(n)])
    data_power = data_power.drop(to_remove).reset_index(drop=True)
    # create a column timestamps
    data_power["datetime_str"] = data_power["Date"] + " " + data_power["Heure"]

    # 2️⃣ Convertir en datetime
    data_power["timestamp"] = pd.to_datetime(
        data_power["datetime_str"], format="%d/%m/%Y %H:%M"
    )

    # 3️⃣ Optionnel : supprimer la colonne temporaire
    data_power.drop(columns=["datetime_str"], inplace=True)

    #enlever les deux dernières lignes
    data_power=data_power.iloc[:-2]

    #forcer le typage de la colonne Consommation
    data_power["Consommation"] = pd.to_numeric(data_power["Consommation"], errors="coerce")

    #remplace les données manquantes :
    data_power=treat_missing_consumption(data_power)

    return data_power

def treat_missing_consumption(df):
    df['Consommation'] = df['Consommation'].fillna(df['Consommation'].shift(672))

    return df





def concatenate_and_format_rte_files(directory_path: str) -> pd.DataFrame:
    """
    Parcourt tous les fichiers CSV dans le dossier spécifié,
    applique la fonction `format_rte_files` à chacun,
    et concatène tous les résultats dans un seul DataFrame.

    Args:
        directory_path (str): Chemin du dossier contenant les fichiers CSV.

    Returns:
        pd.DataFrame: DataFrame final combinant tous les CSV formatés.
    """
    df_final = pd.DataFrame()  # Initialisation du DataFrame final

    # Récupère tous les fichiers CSV dans le dossier
    csv_files = glob(os.path.join(directory_path, "*.csv"))

    for file in csv_files:
        # Lecture et formatage du fichier CSV
        df = format_rte_files(file)
        # Concaténation verticale avec le DataFrame final
        df_final = pd.concat([df_final, df], ignore_index=True)

    # Tri par la colonne timestamp (assurez-vous que c'est de type datetime)
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
    df_final = df_final.sort_values(by='timestamp').reset_index(drop=True)

    return df_final



def create_date_column(data_power:DataFrame,n,block_size=100)-> DataFrame:
    # Assurez-vous que la colonne 4 existe
    data_power["Date"] = None

    block_size = 100
    n = len(data_power) // block_size  # nombre de blocs

    # 1️⃣ récupérer les valeurs sources (10 derniers caractères)
    vals = data_power.iloc[0:n * block_size:block_size, 0].str[-10:].values

    # 2️⃣ créer un array qui répète chaque valeur pour 97 lignes
    # (on voulait index+2 à index+99 → 97 lignes par bloc)
    vals_repeated = np.repeat(vals, 96)

    # 3️⃣ créer les indices correspondants dans data_power
    # start indices pour chaque bloc
    starts = np.arange(n) * block_size + 2
    # pour chaque bloc, ajouter 0→96
    indices = (starts[:, None] + np.arange(96)).flatten()

    # 4️⃣ assigner directement dans la colonne 4
    data_power.loc[indices, "Date"] = vals_repeated

    return data_power


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
    df.loc[df["timestamp"] >= split_date, "set"] = "test"

    return df


def create_Y_matrix(df, col: str = "Consumption",
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
    data_power = add_temporal_features(data_power, column_timestamp, column_consommation=column_power)

    # split train/test
    data_power = split_train_test(data_power,split_date)

    #definition of train and testing set
    train, test = data_power[data_power["set"] == "train"], data_power[data_power["set"] == "test"]

    columns_to_exclude = ["start_date", "set","day_name","Heure", "Previsions J-1","Previsions J","Date","timestamp"]
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
