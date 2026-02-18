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

try:
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model from %s", MODEL_PATH)
    raise RuntimeError(f"Error loading model: {e}")

try:
    logger.info("Loading SHAP explainer from %s", EXPLAINER_PATH)
    shap_explainer = joblib.load(EXPLAINER_PATH)
except Exception as e:
    logger.exception("Failed to load SHAP explainer from %s", EXPLAINER_PATH)
    raise RuntimeError(f"Error loading explainer: {e}")

try:
    logger.info("Loading feature names from %s", FEATURES_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    logger.exception("Failed to load feature names from %s", FEATURES_PATH)
    raise RuntimeError(f"Error loading feature names: {e}")


# -------------------------------------------------------------------
# FastAPI initialization
# -------------------------------------------------------------------

app = FastAPI(
    title="Accidents Severity Prediction API",
    description="API de prédiction de la gravité des accidents routiers",
    version="2.0.0",
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

@app.get("/api/v1/health")
def health_check(auth: None = Depends(authenticate)):
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(feature_names)
    }



# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/api/v1/predict")
def predict(payload: AccidentFeatures, auth: None = Depends(authenticate)):
    try:
        input_dict = payload.features

        # Check for missing features
        missing_features = set(feature_names) - set(input_dict.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {sorted(list(missing_features))}"
            )

        # Create the DataFrame in the expected order
        X = pd.DataFrame([input_dict], columns=feature_names)

        # Convert bool -> int if needed
        X = X.replace({True: 1, False: 0})

        # Prediction
        prediction = model.predict(X)

        response = {
            "prediction": float(prediction[0])
        }

        # Probabilities if available
        if hasattr(model, "predict_proba"):
            response["probabilites"] = model.predict_proba(X)[0].tolist()

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e}"
        )
    
@app.post("/api/v2/predict")
async def predict_v2(payload: PredictionInputV2, auth: None = Depends(authenticate)):
    """
    V2 Endpoint: Receives business requirements and orchestrates transformation before inference.
    """
    logger.info(f">>>>> Call /api/v2/predict called <<<<<")
    # 1. Prepare the payload
    inputs = {
        "cities": payload.cities,
        "timestamp": payload.timestamp
    }
    try:
        preds = model_prediction(inputs, model, feature_names,shap_explainer)
    except ValueError as exc:
        logger.error(f"Value error during prediction: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    logger.info(f">>>>> Endpoint /api/v2/predict completed <<<<<\n\nx=======x")
    return preds

@app.post("/api/risk-timeline")
async def risk_timeline(auth: None = Depends(authenticate)):
    """
    Endpoint to retrieve the risk timeline from business requirements.
    """
    logger.info(f">>>>> Call /api/risk-timeline called <<<<<")
    
    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamp_iso = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    inputs = {
        "cities": data,
        "timestamp": timestamp_iso
    }

    try:
        timeline_data = timeline_prediction(inputs, model, feature_names)
    except ValueError as exc:
        logger.error(f"Value error during timeline prediction: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected timeline prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Timeline error: {exc}")
    logger.info(f"Timeline data: {timeline_data}")
    logger.info(f">>>>> Endpoint /api/risk-timeline completed <<<<<\n\nx=======x")
    return timeline_data

@app.get("/api/roads")
def get_roads(auth: None = Depends(authenticate)):
    logger.info(f">>>>> Call GET /api/roads called <<<<<")
    load_file_path = path_prefix+API_CFG.road_secteur_path
    if not Path(load_file_path).exists():
        load_file_path = path_prefix+API_CFG.road_secteur_path.replace("_current", "_ref")

    logger.info(f"Loading roads data from {load_file_path}")
    df = pd.read_csv(load_file_path, sep=";", encoding="utf-8", dtype=str)
    logger.info(f">>>>> Endpoint GET /api/roads completed <<<<<\n\nx=======x")
    return df.to_dict(orient="records")

@app.put("/api/roads")
def put_roads(rows: list[dict], auth: None = Depends(authenticate)):
    logger.info(f">>>>> Call PUT /api/roads called <<<<<")
    df = pd.DataFrame(rows)
    df.to_csv(path_prefix+API_CFG.road_secteur_path, sep=";", index=False, encoding="utf-8")
    logger.info(f"CSV file {API_CFG.road_secteur_path} updated successfully with {len(rows)} rows.")
    logger.info(f">>>>> Endpoint PUT /api/roads completed <<<<<\n\nx=======x")
    return {"status": "ok"}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

@app.get("/")
def health_check_api(auth: None = Depends(authenticate)):
    return FileResponse(TEMPLATE_PATH)

# Authentication endpoint
@app.post("/login")
def login_endpoint(current_user: None = Depends(authenticate)):
    """
    Valide username/password via Basic Auth.
    """
    return JSONResponse(content={"message": f"Connexion réussie {current_user}"}, status_code=200)