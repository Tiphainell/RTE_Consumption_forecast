import pandas as pd
from pandas import DataFrame


def compute_average_mean(df: DataFrame, window_frame: str) -> DataFrame:
    # traitement selon freq
    if window_frame == "daily":
        window = 96
        pass
    elif window_frame == "weekly":
        window = 672
        pass
    elif window_frame == "monthly":
        # code pour monthly
        window = 2688
        pass
    df.copy()
    df[f"average_mean_{window_frame}"] = df["average_imported_power_kw"].rolling(window=window, min_periods=1).mean()

    return df


import pandas as pd
from pandas import DataFrame


def compute_average_mean_plus(df: DataFrame, column: str, window: int = None, freq: str = None) -> DataFrame:
    """
    Calcule une moyenne glissante sur une colonne d'un DataFrame.

    Args:
        df : DataFrame contenant la colonne et éventuellement une colonne timestamp.
        column : nom de la colonne sur laquelle calculer la moyenne.
        window : taille de la fenêtre en nombre de lignes (si freq=None).
        freq : fréquence temporelle ('D', 'M', 'H', ...) si on veut une moyenne resample.

    Returns:
        DataFrame avec la colonne 'average'.
    """
    df = df.copy()

    if freq:
        # Assure que le DataFrame a une colonne datetime appelée 'timestamp'
        if 'start_date' not in df.columns:
            raise ValueError("La colonne 'start_date' doit exister pour le calcul par fréquence")

        # Resample selon la fréquence et calcul de la moyenne
        df_resampled = df.set_index('start_date').resample(freq)[column].mean().reset_index()
        df_resampled.rename(columns={column: f"{column}_average"}, inplace=True)
        return df_resampled

    elif window:
        # Moyenne glissante par nombre de lignes
        df[f"{column}_average"] = df[column].rolling(window=window, min_periods=1).mean()
        return df

    else:
        raise ValueError("Il faut spécifier au moins window ou freq")
