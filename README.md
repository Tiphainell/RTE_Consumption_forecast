# README - RTE consumption Forecast Time series

## Project de prédiction de la consommation de puissance électrique sur 24 h à partir des données publiées par RTE
   Ce projet évalue la performance d'un modèle XGboost utilisé pour faire de la prédiction de times series : un modèle xg boost pour chaque pas de 15 min 
   sur 24h par rapport à un modèle naif ( même consommation que la semaine précédente à la même heure).
   Les données de consommation sont prises parmi les données publiées par RTE au lien suivant :
   https://www.services-rte.com/fr/telechargez-les-donnees-publiees-par-rte.html?activation_key%3D7b7add15-3796-496a-acae-0ce60ef1c969%26activation_type%3Dpublic=&category=consumption&type=short_term
   Ce repo donne donc dans le script utils/data_processing des fonctions pour gérer le format des fichiers RTE.
   Le modèle XGboost est entrainé sur les données de janvier 2021 à 30 avril 2023. Le modèle est testé sur les données de mai 2023 à décembre 2023 et comparé au modèle naif que la base de la MAE par horizon de temps sur 24h ainsi que globallement.


## Installation

Cloner le repo et installer les dépendances :

```bash
git clone ggit@github.com:Tiphainell/RTE_Consumption_forecast.git
cd RTE_Consumption_forecast
python3 -m venv .venv
source .venv/bin/activate
pip install .
pip install jupyter
```

## Resultat sur le test set mai 2023 à décembre 2023 : 7 mois

MAE en production (backtest) : 47 kW (≈ 8 % d'erreur relative / la puissance moyenne sur la période),
contre un MAE de 161 kw pour un modèle naïf utilisant la valeur observée une semaine plus tôt (P(t) = P(t−7 jours)).

→ Le modèle XGBoost réduit donc l’erreur d’environ 70 % par rapport à ce modèle naif.

MAE par horizon (96 horizons → 24h) :
![img.png](img.png)

Comparaison des prédictions en production (modèle naïf vs XGBoost vs observé) sur 3 jours :
![img_1.png](img_1.png)

## À lire pour comprendre le raisonnement

1. `Notebooks/Exploration.ipynb`  
   - Analyse exploratoire des données, identification des patterns saisonniers et hebdomadaires, création des features, et justification du choix du modèle.  

2. `Notebooks/Validation.ipynb`  
   - Évaluation des performances du modèle, notamment en backtest et comparaison avec un modèle naïf.



# Structure du projet 
## Notebooks

- **Exploration** : analyse graphique des puissances moyennes à différentes fréquences temporelles. Sert à identifier les features pertinentes et à justifier le choix de XGBoost.  
- **Validation** : évaluation quantitative et graphique du modèle sur le test set, incluant backtest et métriques (MAE globale et par horizon). 


## Dossier `src`

Code source du projet :

train.py : script d’entraînement du modèle XGBoost (paramétrable via config/config.yaml)

utils/ : fonctions utilitaires (feature engineering, preprocessing, etc.)

Le réentraînement n’est pas obligatoire : les prédictions sont déjà stockées dans predictions/.

> Remarque : Lancer `train.py` n’est pas obligatoire si vous voulez simplement consulter les résultats (stockés dans predictions), mais c’est utile pour réentraîner le modèle avec d’autres hyperparamètres.  

---

## Dossier `config`

Contient les configurations utilisées pour l'entraînement (config.yaml).

## Dossier `model`

Modèle XGBoost entraîné et sauvegardé.

## Dossier `predictions`

Matrices de prédiction utilisées pour la validation et les graphiques.


## Dossier `Questions_partie_2`
 
Réponses aux questions de la partie 2 (données additionnelles, optimisation batterie, MLOps).







