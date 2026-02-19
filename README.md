# Pipeline MLOps pour l'Analyse des Données d'Accidents

**Formation :** Machine Learning Engineer - Expert en ingénierie de l'intelligence artificielle  
**Institution :** Liora / Mines ParisTech Executive Education

**Auteurs :**
- Julie PINTO
- Diego KLOPFENSTEIN
- Yasser BELAIDI
- Yves BRU

**Responsable de cohorte :** Maria DOMENZAIN ACEVEDO  
**Mentor de projet :** Antoine Fradin

---

## Aperçu

Ce projet fournit un pipeline MLOps complet pour predire la gravite des accidents routiers dans un perimetre cible. Il alimente une interface graphique utilisee par une caserne de pompiers situee a Bassens (33, Gironde) afin d'identifier les routes les plus a risque dans les prochaines heures et de visualiser une timeline de risque sur 24h. Le systeme s'appuie sur des donnees BAAC (Bases de données annuelles des accidents corporels de la circulation routière) de 2019 a 2024, nettoyees et encodees, puis entraine un modele XGBoost pour produire des predictions contextualisees.
Les prédictions sont calculées à partir d'input combinant des variables structurelles (vitesse autorisee, largeur, infrastructure, etc.) definies et mises a jour par l'utilisateur via l'interface, avec des donnees temps reel (https://openweathermap.org/) et un contexte temporel (date/heure).

## Objectifs

- Fournir a la caserne de Bassens une prediction du risque d'accident grave dans les prochaines heures, localisee par route.
- Remonter dans l'interface graphique le top 5 des routes avec la probabilite la plus elevee d'accident grave.
- Generer une timeline des risques sur 24h pour aider a la planification des moyens.
- Combiner des variables structurelles (vitesse autorisee, largeur, infrastructure, etc.) definies et mises a jour par l'utilisateur via l'interface, avec des donnees temps reel (meteo) et un contexte temporel (date/heure) pour affiner les predictions.

## Prérequis

- **Python 3.9+**
- **Environnement virtuel** (venv/conda)
- **Certificats SSL Nginx**
  - `deployments/nginx/certs/nginx.crt`
  - `deployments/nginx/certs/nginx.key`
- **Création des utilisateurs**

### 2) Initialisation du Projet

A la première utilisation, lancez l'initialisation complète :

```bash
make init
```

Cette commande configure automatiquement :
- Les répertoires de base (`data/`, `models/`, `logs/`, `metrics/`)
- Le fichier `.env` avec les variables d'environnement nécessaires
- Les permissions Airflow sur systèmes Linux/WSL
- La base de données utilisateurs (`users_db.json`)

Le script détecte votre système et applique les configurations appropriées (Linux natif, WSL2, macOS).

**Note** : Lors de la première exécution, vous devrez créer un utilisateur administrateur en fournissant un nom d'utilisateur et un mot de passe.



## Démarrage Rapide



### 1) Permissions Docker (WSL)

Si `docker ps` retourne une erreur de permission :
```bash
sudo chmod 666 /var/run/docker.sock
```

### Installation
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Exécution (Docker)
```bash
# Démarrer le projet (build + containers)
make start-project

# Arrêter le projet
make stop-project

# Reset complet (supprime containers orphelins, rebuild)
make reset-project

```

### Pipeline
```bash
# Pipeline complet (hors Docker)
make pipeline

# Ou étapes individuelles
python src/pipeline/01_data_import/main.py
python src/pipeline/02_data_clean/main.py
python src/pipeline/03_merge/main.py
python src/pipeline/04_encodage/main.py
python src/pipeline/05_data_transformation/main.py
python src/pipeline/06_resampling/main.py
python src/pipeline/07_model_trainer/main.py
python src/pipeline/08_model_evaluation/main.py
python src/pipeline/09_shap_explicability/main.py
```

### Tests unitaires
```bash
# Tests API (sortie verbose avec prints)
make unit-test

# Tests API (sortie verbose sans captures stdout)
make unit-test-debug
```

## Test d'intégration
```bash
# Tests API (sortie verbose avec prints)
make int-test

# Tests API (sortie verbose sans captures stdout)
make int-test-debug
```

Vérifier les logs :
```bash
cat logs/logs.log
```

## Caractéristiques Principales

- **Pipeline Piloté par Configuration** : Configuration centralisée via `config.yaml`
- **Validation de Schéma Stricte** : Vérifications de qualité aux étapes critiques via `schema.yaml`
- **Journalisation Centralisée** : Logger personnalisé avec répertoire dédié `logs/logs.log`
- **Architecture Modulaire** : Séparation entre `src/` et `pipeline/`

## 🛡️ Architecture de Validation des Données

### Schema (src/data_processing/schema.yaml)

Schéma de données centralisé définissant type, description et utilisation de chaque colonne :

```yaml
COLUMNS:
  Num_Acc:
    type: int64
    description: "Numéro d'identifiant de l'accident"
    normalized: False
    use_for_fit: False # Exclut l'ID des calculs
  an:
    type: int64
    description: "Année de l'accident"
    normalized: True
    use_for_fit: True  # Utilisé pour l'entraînement
```

### Rapport d'Exécution (data/status.txt)

Le fichier est reinitialise au demarrage du pipeline et complete a chaque etape. En cas de divergence, le pipeline s'arrete (fail-fast) :

```
AGGREGATE:Validation status: ✅ True
```

Ou en cas d'erreur :

```
Missing columns in DataFrame: {"catu","place"} 
Extra columns in DataFrame: {"Couleur"} 
MERGE:Validation status: ❌ False
```

### Fonctionnalités Clés

1. **Contrôle d'Intégrité** : Vérification des types et colonnes obligatoires
2. **Configuration du Preprocessing** : Identification des colonnes à normaliser
3. **Sélection Dynamique des Features** : Ajustement des variables d'entraînement via YAML

## Modèle et Artefacts

### Parametres de Model Training

Le `param_grid` XGBoost est definie dans `src/modeling/params.yaml` et chargee par le pipeline au moment du training.

### Artefacts attendus

Au demarrage, l'API charge le modele et la liste des features. Si un artefact manque, l'API echoue au startup.
Artefacts requis (configures dans `config.yaml`) :

- `models/best_model.joblib`
- `models/features.joblib`
- `models/one_hot_encoder.joblib`
- `models/shap_explainer.joblib`

## API S.A.V.E.R. (saver_app)

L'interface web `saver_app.html` est servie à la racine via Nginx et consomme l'API de prédiction V2.

### Endpoints utilisés par l'app

**GET /**
- Sert l'interface S.A.V.E.R. (`saver_app.html`).

**GET /api/v1/health**
- Healthcheck de l'API.
- Réponse exemple :
```json
{
  "status": "ok",
  "model_loaded": true,
  "n_features": 123
}
```

**POST /api/v2/predict**
- Endpoint principal appelé par l'interface.
- Body JSON :
```json
{
  "cities": ["Bassens", "Sainte-Eulalie"],
  "timestamp": "2026-02-10T22:00:00Z"
}
```
- Le champ `cities` doit etre dans la liste `secteur_insee` du `config.yaml`.
- `timestamp` est attendu en UTC au format `YYYY-MM-DDTHH:00:00Z`.
- Réponse exemple :
```json
{
  "status": "success",
  "top_k": 5,
  "data": [
    {
      "commune": "Bassens",
      "adresse": "avenue de la république",
      "facteurs": "...",
      "prediction": 61.2345,
      "risk_level": 6,
      "risk_label": "MODÉRÉ"
    }
  ]
}
```

**POST /api/risk-timeline**
- Retourne une timeline de risque basee sur les villes de `secteur_insee` et l'heure courante UTC.
- Reponse type :
```json
{
  "status": "success",
  "data": [
    {
      "timestamp": 1770760800,
      "risk_index": 42.1,
      "risk_level": 3,
      "risk_label": "Moderate",
      "temperature_c": 12.3,
      "description": "clear sky",
      "daylight": true
    }
  ]
}
```

## Structure du Projet
```
SEP25-BMLE-MLOPS-ACCIDENTS/
├── src/                           
│   ├── data_processing/           # Modules de gestion des données
│   │   ├── clean/                 # Nettoyage par type de données
│   │   ├── engineering/           # Feature engineering
│   │   ├── check_structure.py     # Verifs fichiers/colonnes
│   │   ├── data_import.py         # Import des donnees brutes
│   │   ├── data_clean.py          # Nettoyage des donnees
│   │   ├── data_merge.py          # Fusion des tables
│   │   ├── data_encoding.py       # Encodage des variables
│   │   ├── data_transformation.py # Split/normalisation/features
│   │   ├── schema.yaml            # Schema de reference
│   │   └── __init__.py
│   ├── modeling/                  # Modules ML et prédictions
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── params.yaml
│   │   └── __init__.py
│   ├── pipeline/                  # Orchestration du pipeline
│   │   ├── 01_data_import
│   │   ├── 02_data_clean
│   │   ├── 03_merge
│   │   ├── 04_encodage
│   │   ├── 05_data_transformation
│   │   ├── 06_resampling
│   │   ├── 07_model_trainer
│   │   ├── 08_model_evaluation
│   │   ├── 09_shap_explicability
│   │   └── dags
│   │       └── pipeline_dag.py    # Airflow dags
│   ├── api/                       # API FastAPI
│   │   ├── feature_encoder.py     # Encodage des features
│   │   ├── feature_time.py        # Features temporelles
│   │   ├── feature_weather.py     # Features meteo
│   │   ├── inference_engine.py    # Orchestration inference
│   │   ├── main.py                # FastAPI app
│   │   ├── prediction.py          # Logique de prediction
│   │   └── weather.py             # Client meteo
│   ├── config.yaml                # Configuration des step de la pipeline
│   ├── config.py                  # Configuration des paramètres du projet
│   ├── config_manager.py          # Gestionnaire de configuration
│   ├── custom_logger.py           # Logger personnalisé pour la journalisation
│   ├── entity.py                  # Définition des entités de données
│   └── common_utils.py            # Fonctions utilitaires communes
├── data/                          # Répertoire des données
│   ├── 00_raw/                    # Données brutes (2019-2024)
│   ├── 01_processed/              # Données nettoyées
│   ├── 02_processed_merged/       # Données fusionnees et encodees
│   ├── 03_features_selected/      # Ensembles train/test et features selectionnees
│   ├── 04_resampled/              # Donnees re-equilibrees
│   └── status.txt
├── logs/                          # Journaux d'exécution
├── models/                        # Artefacts de modèle
├── metrics/                       # Métriques d'évaluation
├── notebooks/                     # Analyses exploratoires
├── templates/                     # Templates application
├── requirements.txt               
├── .gitignore                     
└── README.md                      
```