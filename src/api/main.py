import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../model/random_forest_regressor.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "../../model/features.joblib")


# -------------------------------------------------------------------
# Chargement du modèle et des features
# -------------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

try:
    feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading feature names: {e}")


# -------------------------------------------------------------------
# Initialisation FastAPI
# -------------------------------------------------------------------

app = FastAPI(
    title="Accidents Severity Prediction API",
    description="API de prédiction de la gravité des accidents routiers",
    version="2.0.0",
)


# -------------------------------------------------------------------
# Schéma d'entrée : dictionnaire de features
# -------------------------------------------------------------------

class AccidentFeatures(BaseModel):
    features: Dict[str, Any]


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(feature_names)
    }


# -------------------------------------------------------------------
# Endpoint de prédiction
# -------------------------------------------------------------------

@app.post("/api/v1/predict")
def predict(payload: AccidentFeatures):
    try:
        input_dict = payload.features

        # Vérification des features manquantes
        missing_features = set(feature_names) - set(input_dict.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {sorted(list(missing_features))}"
            )

        # Création du DataFrame dans le BON ordre
        X = pd.DataFrame([input_dict], columns=feature_names)

        # Conversion bool -> int si nécessaire
        X = X.replace({True: 1, False: 0})

        # Prédiction
        prediction = model.predict(X)

        response = {
            "prediction": float(prediction[0])
        }

        # Probabilités si dispo
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

