import pandas as pd
from pandas import DataFrame
import numpy as np

def assign_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def cyclic_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day"] / 365)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df



def add_lags(df, target_col, lags=None, rolling_windows=None):
    """
    Ajoute des lags et des rolling means pour un modèle de forecast.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec une colonne target.
    target_col : str
        Nom de la colonne cible.
    lags : list[int], optional
        Liste des lags en nombre de périodes (ex: [1,4,96])
    rolling_windows : list[int], optional
        Liste des fenêtres pour rolling mean (ex: [4,96])

    Returns
    -------
    df_new : pd.DataFrame
        DataFrame avec les colonnes lag_x et rolling_y ajoutées.
    """
    df_new = df.copy()

    if lags is None:
        lags = [1, 4, 96, 672]  # lags par défaut
    if rolling_windows is None:
        rolling_windows = [4, 96]  # rolling par défaut

    # Ajouter les lags
    for lag in lags:
        df_new[f"{target_col}_lag_{lag}"] = df_new[target_col].shift(lag)

    # Ajouter les rolling means
    for window in rolling_windows:
        df_new[f"{target_col}_rollmean_{window}"] = df_new[target_col].shift(1).rolling(window=window).mean()
        # shift(1) pour ne pas inclure la valeur courante dans la moyenne

    return df_new


def add_temporal_features(df: DataFrame, column_timestamp: str) -> DataFrame:
    df = df.copy()
    df[column_timestamp] = pd.to_datetime(df[column_timestamp])
    df["year"] = df[column_timestamp].dt.year
    df["month"] = df[column_timestamp].dt.month
    df["day"] = df[column_timestamp].dt.day
    df["hour"] = df[column_timestamp].dt.hour
    df["minute"] = df[column_timestamp].dt.minute
    df["week"] = df[column_timestamp].dt.isocalendar().week

    df['week'] = df['start_date'].dt.isocalendar().week
    df['day_name'] = df['start_date'].dt.day_name()
    df["day_of_week"] = df["start_date"].dt.dayofweek

    # ajout de si week-end ou pas :
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Ajout des vacances d'hiver
    df['is_winter_holiday'] = (
            ((df['start_date'].dt.month == 12) & (df['start_date'].dt.day >= 20)) |
            ((df['start_date'].dt.month == 1) & (df['start_date'].dt.day <= 5))
    )

    # Ajout des vacances d'été
    df['is_summer_holiday'] = (df['start_date'].dt.month == 8)


    # Ajout 15 août
    df['15_august'] = (
        ((df['start_date'].dt.month == 8) & (df['start_date'].dt.day == 15))
    )
    #Season added
    df["season"] = df["month"].apply(assign_season)


    #Night shift
    h = df["start_date"].dt.hour
    df["night_shift"] = ((h >= 18) | (h <= 3)).astype(int)

    #lags added and rolling mean average
    df=add_lags(df,"average_imported_power_kw")
    df=cyclic_features(df)

    return df


def compute_mean_freq_dynamic(df: DataFrame, column: str = "average_imported_power_kw") -> DataFrame:
    """
    Calcule des moyennes glissantes selon différentes fréquences temporelles
    et les affecte à chaque ligne du DataFrame.

    Args:
        df : DataFrame avec une colonne datetime 'timestamp' et la colonne à moyenner
        column : nom de la colonne sur laquelle calculer les moyennes

    Returns:
        DataFrame avec de nouvelles colonnes 'imported_power_kw_<freq>_avg'
    """
    df = df.copy()

    # Dictionnaire : clé = nouvelle colonne, valeur = colonnes pour le groupby
    freq_groups = {
        'year': ['year'],
        'month': ['year', 'month'],
        'day': ['year', 'month', 'day'],
        'hour': ['year', 'month', 'day', 'hour'],
        'minute': ['year', 'month', 'day', 'hour', 'minute'],
        'week': ['year', 'week']
    }

    # Boucle dynamique sur les fréquences
    for freq, group_cols in freq_groups.items():
        new_col = f"{column}_{freq}_avg"
        df[new_col] = df.groupby(group_cols)[column].transform('mean')

    return df
