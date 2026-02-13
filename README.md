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

Pipeline ML prêt pour la production pour l'analyse des données d'accidents, suivant une architecture modulaire avec orchestration pilotée par configuration, validation de schéma stricte et surveillance centralisée.

## Objectifs

Prédire les zones de danger au sein d'un secteur géographique restreint (périmètre d'intervention d'une caserne de pompiers) avec analyse temporelle configurable. Le système enrichit les données BAAC avec des informations en temps réel (conditions météorologiques, luminosité) et caractéristiques des infrastructures routières locales.

## Caractéristiques Principales

- **Pipeline Piloté par Configuration** : Configuration centralisée via `config.yaml`
- **Validation de Schéma Stricte** : Vérifications de qualité aux étapes critiques via `schema.yaml`
- **Journalisation Centralisée** : Logger personnalisé avec répertoire dédié `logs/logs.log`
- **Architecture Modulaire** : Séparation entre `src/` et `pipeline/`

## Prérequis
- Python 3.9+
- Environnement virtuel (venv/conda)

## Démarrage Rapide

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Exécution
```bash
# Pipeline complet
python .\src\pipeline\00_pipeline.py

# Ou étapes individuelles
python .\src\pipeline\01_data_import.py
python .\src\pipeline\02_data_clean.py
python .\src\pipeline\03_merge.py
python .\src\pipeline\04_encodage.py
python .\src\pipeline\05_data_transformation.py
```

Vérifier les logs :
```bash
cat logs/logs.log
```

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
- Réponse exemple :
```json
{
  "status": "success",
  "data": [
    {
      "commune": "Bassens",
      "adresse": "avenue de la république",
      "facteurs": "...",
      "prediction": 61.2345,
      "stabilite": "➡️ Stable"
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
│   │   ├── check_structure.py
│   │   ├── data_import.py
│   │   ├── data_clean.py
│   │   ├── data_merge.py
│   │   ├── data_encoding.py
│   │   ├── data_transformation.py
│   │   ├── schema.yaml
│   │   └── __init__.py
│   ├── models/                    # Modules ML et prédictions
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── params.yaml
│   │   └── __init__.py
│   ├── pipeline/                  # Orchestration du pipeline
│   │   ├── 00_pipeline.py         # Pipeline complet
│   │   ├── 01_data_import.py
│   │   ├── 02_data_clean.py
│   │   ├── 03_merge.py
│   │   ├── 04_encodage.py
│   │   ├── 05_data_transformation.py
│   │   └── __init__.py
│   ├── app/                       # Application
│   ├── config.yaml                # Configuration des step de la pipeline
│   ├── config.py                  # Configuration des paramètres du projet
│   ├── config_manager.py          # Gestionnaire de configuration
│   ├── custom_logger.py           # Logger personnalisé pour la journalisation
│   ├── entity.py                  # Définition des entités de données
│   └── common_utils.py            # Fonctions utilitaires communes
├── data/                          # Répertoire des données
│   ├── raw/                       # Données brutes (2019-2024)
│   ├── processed/                 # Données nettoyées
│   ├── processed_merged/          # Données fusionnées et encodées
│   ├── train_test/                # Ensembles d'entraînement/test
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

Chaque étape compare le DataFrame au schéma. En cas de divergence, le pipeline s'arrête (fail-fast) :

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

