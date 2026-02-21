from pathlib import Path
import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List

from inference_engine import model_prediction, timeline_prediction
from src.common_utils import read_yaml
from src.custom_logger import logger

from datetime import datetime, timezone
from basicauth import authenticate

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

def is_docker():
    return Path('/.dockerenv').exists() or os.getenv('DOCKER_CONTAINER') == 'true'
path_prefix = "" if is_docker() else "."

CONFIG = read_yaml(Path("src/config.yaml"))
API_CFG = CONFIG.api_inference

MODEL_PATH = Path(path_prefix+API_CFG.model_path)
EXPLAINER_PATH = Path(path_prefix+API_CFG.explainer_path)
FEATURES_PATH = Path(path_prefix+API_CFG.features_path)
TEMPLATE_PATH = Path(path_prefix+API_CFG.template_path)

# City list and INSEE mapping
secteur = API_CFG.secteur_insee
data = list(secteur.keys())

# -------------------------------------------------------------------
# Load the model and feature list
# -------------------------------------------------------------------

def load_prediction_model():
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model from %s", MODEL_PATH)
        raise RuntimeError(f"Error loading model: {e}")
    return model

def load_shap_explainer():
    logger.info("Loading SHAP explainer from %s", EXPLAINER_PATH)
    try:
        shap_explainer = joblib.load(EXPLAINER_PATH)
    except Exception as e:
        logger.exception("Failed to load SHAP explainer from %s", EXPLAINER_PATH)
        raise RuntimeError(f"Error loading explainer: {e}")
    return shap_explainer

def load_feature_names():
    logger.info("Loading feature names from %s", FEATURES_PATH)
    try:
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        logger.exception("Failed to load feature names from %s", FEATURES_PATH)
        raise RuntimeError(f"Error loading feature names: {e}")
    return feature_names

# -------------------------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

INFERENCE_TIME = Histogram(
    "ml_inference_duration_seconds",
    "Time spent in ML inference",
    ["endpoint"]
)

INFERENCE_ERRORS = Counter(
    "ml_inference_errors_total",
    "Total ML inference errors",
    ["endpoint"]
)

# -------------------------------------------------------------------
# FastAPI initialization
# -------------------------------------------------------------------

app = FastAPI(
    title="Fast API",
    description="API de prédiction, timeline des risques, routes et monitoring",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# -------------------------------------------------------------------
# Input schema: feature dictionary
# -------------------------------------------------------------------

class AccidentFeatures(BaseModel):
    features: Dict[str, Any]

class PredictionInputV2(BaseModel):
    cities: List[str]
    timestamp: str


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/api/health")
def health_check(auth: None = Depends(authenticate)):
    start_time = time.time()
    endpoint = "/api/login"

    logger.info(f">>>>> Call /api/health called <<<<<")
 
    try:
        inference_start = time.time()
        REQUEST_COUNT.labels("GET", endpoint, "200").inc()
        return {
            "status": "ok",
            "model_loaded": True,
            "n_features": len(load_feature_names())
        }

    except ValueError as exc:
        REQUEST_COUNT.labels("GET", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=400, detail=str(exc))
    
    except Exception as exc:
        import traceback
        logger.error(f"traceback error: {traceback.format_exc()}")
        REQUEST_COUNT.labels("GET", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=500, detail=f"Login error: {exc}")
    
    finally:
        REQUEST_LATENCY.labels("GET", endpoint).observe(
            time.time() - start_time
        )
    INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)





# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/api/v1/predict",
    summary="Test inférence à partir des features pré traitées",
    description="""
    **Description détaillée :**
    Prédit la probabilité d'un accident à partir des features fournies.

    **Exemple de payload :**
    ```json
    {
        "features": {
            "feature1": 1,
            "feature2": 0,
            ...
        }
    }
    ```
    **Réponses possibles :**
    - 200 : Prédiction réussie ; retourne la prédiction
    - 400 : Features manquantes ou invalides.
    - 500 : Erreur interne lors de la prédiction.
    """,
    response_description="Un dictionnaire contenant la prédiction et éventuellement les probabilités.",
    tags=["Inférence_Sandbox"]
)
def predict(payload: AccidentFeatures, auth: None = Depends(authenticate)):    
    start_time = time.time()
    endpoint = "/api/v1/predict"
    model = load_prediction_model()

    try:
        input_dict = payload.features

        # Check for missing features
        feature_names = load_feature_names()
        missing_features = set(feature_names) - set(input_dict.keys())
        if missing_features:
            REQUEST_COUNT.labels("POST", endpoint, "400").inc()
            INFERENCE_ERRORS.labels(endpoint).inc()
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {sorted(list(missing_features))}"
            )

        # Create the DataFrame in the expected order
        X = pd.DataFrame([input_dict], columns=feature_names)

        # Convert bool -> int if needed
        X = X.replace({True: 1, False: 0})

        # Inference timing start
        inference_start = time.time()

        # Prediction
        prediction = model.predict(X)

        # Inference timing end
        INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)

        response = {
            "prediction": float(prediction[0])
        }

        # Probabilities if available
        if hasattr(model, "predict_proba"):
            response["probabilites"] = model.predict_proba(X)[0].tolist()

        REQUEST_COUNT.labels("POST", endpoint, "200").inc()

        return response

    except HTTPException:
        raise
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint).inc()
        REQUEST_COUNT.labels("POST", endpoint, "500").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e}"
        )
    finally:
        REQUEST_LATENCY.labels("POST", endpoint).observe(
            time.time() - start_time
        )
    
@app.post("/api/v2/predict",
    summary="Prédire la probabilité d'un accident + explications SHAP",
    description="""
    **Description détaillée :**
    Prédit la probabilité d'un accident à partir de la date fournie, des villes sélectionnées et des informations météorologiques.
    Retourne les routes les plus à risque avec leur niveau de risque et une explication basée sur le modèle SHAP.

    **Exemple de payload :**
    ```json
    {
       "cities": ["City1", "City2"],
       "timestamp": "2026-02-18T12:00:00Z"
    }
    ```
    **Réponses possibles :**
    - 200 : Prédiction réussie avec routes à risque identifiées
    - 400 : Données d'entrée invalides
    - 500 : Erreur interne lors de la prédiction
    """,
    response_description="Dictionnaire contenant les prédictions et explications SHAP pour chaque route.",
    tags=["Inférence_Production"]
)
async def predict_v2(payload: PredictionInputV2, auth: None = Depends(authenticate)):
    """
    Endpoint V2 : reçoit les paramètres métier et orchestre la transformation avant l'inférence.
    """
    start_time = time.time()
    endpoint = "/api/v2/predict"
    model = load_prediction_model()
    feature_names = load_feature_names()
    shap_explainer = load_shap_explainer()

    logger.info(f">>>>> Call /api/v2/predict called <<<<<")
    # 1. Prepare the payload
    inputs = {
        "cities": payload.cities,
        "timestamp": payload.timestamp
    }
    try:
        inference_start = time.time()
        preds = model_prediction(inputs, model, feature_names, shap_explainer)
        INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)
        REQUEST_COUNT.labels("POST", endpoint, "200").inc()
    except ValueError as exc:
        REQUEST_COUNT.labels("POST", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        logger.error(f"Value error during prediction: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        REQUEST_COUNT.labels("POST", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    finally:
        REQUEST_LATENCY.labels("POST", endpoint).observe(
            time.time() - start_time
        )
    logger.info(f">>>>> Endpoint /api/v2/predict completed <<<<<\n\nx=======x")
    return preds


@app.post("/api/risk-timeline",
    summary="Chronologie des risques maximaux d'accidents sur 24h",
    description="""
    **Description détaillée :**
    Récupère la chronologie des risques d'accidents pour les heures suivantes.
    Utilise l'heure actuelle comme point de départ et les prévisions météorologiques
    pour retourner, par créneau horaire, le niveau de risque maximal.

    **Réponses possibles :**
    - 200 : Timeline calculée avec succès
    - 400 : Données d'entrée invalides
    - 500 : Erreur interne lors du calcul
    """,
    response_description="Dictionnaire contenant la chronologie des risques par ville et par heure.",
    tags=["Inférence_Production"]
)
async def risk_timeline(auth: None = Depends(authenticate)):
    """
    Retourne la chronologie des risques selon les règles métier.
    """
    start_time = time.time()
    endpoint = "/api/risk-timeline"
    model = load_prediction_model()
    feature_names = load_feature_names()

    logger.info(f">>>>> Call /api/risk-timeline called <<<<<")
    
    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamp_iso = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    inputs = {
        "cities": data,
        "timestamp": timestamp_iso
    }

    try:
        inference_start = time.time()
        timeline_data = timeline_prediction(inputs, model, feature_names)
        INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)
        REQUEST_COUNT.labels("POST", endpoint, "200").inc()
    except ValueError as exc:
        REQUEST_COUNT.labels("POST", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        logger.error(f"Value error during timeline prediction: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        logger.error(f"traceback error: {traceback.format_exc()}")
        REQUEST_COUNT.labels("POST", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        logger.exception("Unexpected timeline prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Timeline error: {exc}")
    finally:
        REQUEST_LATENCY.labels("POST", endpoint).observe(
            time.time() - start_time
        )
    logger.info(f"Timeline data: {timeline_data}")
    logger.info(f">>>>> Endpoint /api/risk-timeline completed <<<<<\n\nx=======x")
    return timeline_data


@app.get("/api/roads",
    summary="Charge la base de données des routes (infrastructure)",
    description="""
    **Description détaillée :**
    Retourne la base de données des routes et secteurs utilisée pour les prédictions.

    **Réponses possibles :**
    - 200 : Liste des routes/secteurs retournée avec succès.
    - 500 : Erreur interne lors du chargement des données.
    """,
    response_description="Liste des routes/secteurs au format JSON.",
    tags=["Typologie (Data Management)"]
)
def get_roads(auth: None = Depends(authenticate)):
    start_time = time.time()
    endpoint = "/api/roads"

    try:
        inference_start = time.time()
        logger.info(f">>>>> Call GET /api/roads called <<<<<")
        load_file_path = path_prefix+API_CFG.road_secteur_path
        if not Path(load_file_path).exists():
            load_file_path = path_prefix+API_CFG.road_secteur_path.replace("_current", "_ref")

        logger.info(f"Loading roads data from {load_file_path}")
        df = pd.read_csv(load_file_path, sep=";", encoding="utf-8", dtype=str)
        REQUEST_COUNT.labels("GET", endpoint, "200").inc()
        logger.info(f">>>>> Endpoint GET /api/roads completed <<<<<\n\nx=======x")
        
        return df.to_dict(orient="records")

    except ValueError as exc:
        REQUEST_COUNT.labels("GET", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        import traceback
        logger.error(f"traceback error: {traceback.format_exc()}")
        REQUEST_COUNT.labels("GET", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=500, detail=f"Get Roads failed with error: {exc}")
    
    finally:
        REQUEST_LATENCY.labels("GET", endpoint).observe(
            time.time() - start_time
        )
    INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)

@app.put("/api/roads",
    summary="Met à jour la base de données des routes (infrastructure)",
    description="""
    **Description détaillée :**
    Remplace le contenu du fichier de routes/secteurs par les lignes fournies.

    **Exemple de payload :**
    ```json
    [
      {
        "nom_voie": "Avenue de la République",
        "commune": "Bassens"
      }
    ]
    ```
    **Réponses possibles :**
    - 200 : Mise à jour effectuée avec succès.
    - 400 : Données d'entrée invalides.
    - 500 : Erreur interne lors de l'écriture du fichier.
    """,
    response_description="Statut de confirmation de la mise à jour.",
    tags=["Typologie (Data Management)"]
)
def put_roads(rows: list[dict], auth: None = Depends(authenticate)):
    start_time = time.time()
    endpoint = "/api/roads"

    logger.info(f">>>>> Call PUT /api/roads called <<<<<")
    
    try:
        inference_start = time.time()
        df = pd.DataFrame(rows)
        df.to_csv(path_prefix+API_CFG.road_secteur_path, sep=";", index=False, encoding="utf-8")

        logger.info(f"CSV file {API_CFG.road_secteur_path} updated successfully with {len(rows)} rows.")
        logger.info(f">>>>> Endpoint PUT /api/roads completed <<<<<\n\nx=======x")
        REQUEST_COUNT.labels("PUT", endpoint, "200").inc()
        
        return {"status": "ok"}
            
    except ValueError as exc:
        REQUEST_COUNT.labels("PUT", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=400, detail=str(exc))
    
    except Exception as exc:
        import traceback
        logger.error(f"traceback error: {traceback.format_exc()}")
        REQUEST_COUNT.labels("PUT", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=500, detail=f"Timeline error: {exc}")
    
    finally:
        REQUEST_LATENCY.labels("PUT", endpoint).observe(
            time.time() - start_time
        )
    
    INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

@app.get("/",
    summary="Retourne la page d'accueil de l'API",
    description="""
    **Description détaillée :**
    Retourne la page HTML d'accueil de l'application API.

    **Réponses possibles :**
    - 200 : Page HTML servie avec succès.
    - 500 : Erreur interne lors du chargement du template.
    """,
    response_description="Page HTML d'accueil.",
    tags=["Système"]
)
def health_check_api(auth: None = Depends(authenticate)):
    return FileResponse(TEMPLATE_PATH)

# -------------------------------------------------------------------
# Endpoint Authentication
# -------------------------------------------------------------------
@app.get("/api/login",
    summary="Valide les identifiants Basic Auth",
    description="""
    **Description détaillée :**
    Vérifie les identifiants transmis via l'en-tête Authorization (Basic Auth).
    Retourne les informations utilisateur si l'authentification est valide.

    **Réponses possibles :**
    - 200 : Utilisateur authentifié.
    - 401 : Identifiants invalides ou absents.
    """,
    response_description="Statut d'authentification et informations utilisateur.",
    tags=["Authentification"]
)
def login(current_user: dict = Depends(authenticate)):
    """
    Valide username/password via Basic Auth.
    """
    start_time = time.time()
    endpoint = "/api/login"

    logger.info(f">>>>> Call /api/login called <<<<<")
 
    try:
        inference_start = time.time()
        REQUEST_COUNT.labels("GET", endpoint, "200").inc()
        return {"status": "authenticated", "user": current_user}

    except ValueError as exc:
        REQUEST_COUNT.labels("GET", endpoint, "400").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=400, detail=str(exc))
    
    except Exception as exc:
        import traceback
        logger.error(f"traceback error: {traceback.format_exc()}")
        REQUEST_COUNT.labels("GET", endpoint, "500").inc()
        INFERENCE_ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=500, detail=f"Login error: {exc}")
    
    finally:
        REQUEST_LATENCY.labels("GET", endpoint).observe(
            time.time() - start_time
        )
    INFERENCE_TIME.labels(endpoint).observe(time.time() - inference_start)
    

# -------------------------------------------------------------------
# Endpoint metrics
# -------------------------------------------------------------------
@app.get("/api/metrics",
    summary="Expose les métriques Prometheus",
    description="""
    **Description détaillée :**
    Expose les métriques applicatives au format Prometheus pour la supervision.

    **Réponses possibles :**
    - 200 : Métriques retournées avec succès.
    """,
    response_description="Flux texte des métriques Prometheus.",
    tags=["Monitoring"]
)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)   
    
