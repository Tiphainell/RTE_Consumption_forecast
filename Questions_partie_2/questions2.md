# Partie 1 

`Ce que j'aurais fait en plus si j'avais eu le temps` :

- Analyser les features les plus importantes - tester une régularization des features
- Tester d'autres hyperparamètres pour XG boost ( Séparer la base en train, val, et test pour le faire pour choisir les meilleurs hyperparamètres sur le val)
- Tester d'autres modèles (SARIMA, LSTM) et comparer les performances sur le val

- L'entraînement du modèle XGBoost (`train.py`)  avec la configuration à entrer dans le fichier config/config.yaml
- Les utilitaires utilisés par les notebooks et scripts (`utils/`)  


# Partie 2

### 2.a. Additional data

Je pense que la courbe donnée donne la courbe de puissance de consommation dans des immeubles de bureaux. 

données pour améliorer la performance : 
- données de température extérieure (données Météorologiques) /liées consommation de chauffage
- Consigne de temperature dans le bâtiment par horaire
- "emplois du temps" du bâtiment (courbe de fréquentation du bâtiment, timeline d'équipement consommateur)

### 2.b. Downstream optimization

Si c'est couplé à un optomiseur de batterie, il faut plutôt donnée des outputs probabilistes, 
avec des scénarios de différentes probabilités. 


### 2.c. Orchestration & monitoring

Je ne suis pas encore très familière avec les outils MLOps, mais je suis motivée pour apprendre.
Pour la mise en production, il faudrait :

- Stocker et versionner le modèle, par exemple avec MLflow.

- Orchestrer les jobs horaires, avec un outil comme Airflow, pour recalculer les prévisions.

- Mettre en place un tableau de bord de visualisation et un système d’alertes pour suivre la qualité des prévisions et détecter les problèmes.





