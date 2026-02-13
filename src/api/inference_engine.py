import os
from pathlib import Path

import pandas as pd
import joblib

from src.common_utils import read_yaml
from src.custom_logger import logger
from feature_weather import encode_meteorological_features
from feature_encoder import encode_categorical_values
from prediction import build_top_predictions, make_predictions, score_to_risk_level

# -------------------------------------------------------------------
# TODO
# -------------------------------------------------------------------
# - Replace placeholder factors with data-driven explanations.
# - Add connection with password

def is_docker():
    return os.environ.get('IS_DOCKER', 'false').lower() == 'true'
path_prefix = "" if is_docker() else "."

# Load configuration
CONFIG = read_yaml(Path("src/config.yaml"))
API_CFG = CONFIG.api_inference
MODEL_PATH = Path(path_prefix + API_CFG.model_path)
FEATURES_PATH = Path(path_prefix + API_CFG.features_path)
ENCODER_PATH = Path(path_prefix + API_CFG.encoder_path)
ENCODED_COLS = CONFIG.data_encodage.encode_columns
DEFAULT_TOP_K = int(API_CFG.top_k)

# City list and INSEE mapping
secteur = API_CFG.secteur_insee
data = list(secteur.keys())

# TODO: Temporary list of factors for demonstration purposes. Replace with real factors based on data analysis.
facteurs_list = ["Combinaison : Courbe serrée + Aquaplaning probable + Nuit sans éclairage",
                 "Multiples collisions par temps de pluie à cette heure.",
                    "Intersection complexe avec visibilité réduite.",
                    "Zone de travaux avec signalisation insuffisante.",
                    "Présence de piétons et de cyclistes à proximité.",
                    "Historique d'accidents similaires à cet endroit.",
                    "Conditions météorologiques défavorables (pluie, brouillard).",
                    "Vitesse élevée enregistrée dans cette zone.",
                    "Proximité d'une école ou d'un lieu de rassemblement.",
                    "Signalisation routière peu claire ou absente.",
                    "Route étroite avec peu d'espace pour manœuvrer.",
                    "Présence de virages dangereux ou de pentes abruptes.",
                    "Luminosité faible + visibilité réduite par brouillard localisé.",
                    "Zone de forte affluence avec des interactions complexes entre véhicules et piétons.",
                    "Historique d'accidents graves à cette intersection, souvent liés à des erreurs de jugement des conducteurs.",
                    "Conditions météorologiques défavorables (pluie, neige)"
                 ]

try:
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading encoder: {e}")


def prepare_data_for_prediction(payload,feature_names,time_series=False):
    logger.info(f"Received prediction request with payload: {payload}")
    # Extract INSEE codes via the 'secteur' mapping dictionary
    unknown_cities = [city for city in payload["cities"] if city not in secteur]
    if unknown_cities:
        raise ValueError(f"Unknown cities: {unknown_cities}")
    insee_codes = [secteur[city] for city in payload["cities"]]

    # 2. Load and filter reference data
    road_secteur_path = str(path_prefix + API_CFG.road_secteur_path)

    df = pd.read_csv(road_secteur_path, sep=";")
    df_secteur = df[df['com'].isin(insee_codes)].copy()

    # # 3. Encode date and time features
    # logger.info("Encoding date and time features")
    # df_secteur = encode_date_time(df_secteur, timestamp=payload["timestamp"])

    # 4. Encode meteorological and environmental features
    logger.info("Encoding meteorological and environmental features")
    df_secteur = encode_meteorological_features(
        df_secteur,
        payload["cities"],
        timestamp=payload["timestamp"],
        time_series=time_series,
        secteur=secteur,
    )

    # 5. Encode categorical variables
    logger.info("Encoding categorical variables")
    df_secteur = encode_categorical_values(df_secteur, encoder, ENCODED_COLS)

    # Security check for API
    still_missing = set(feature_names) - set(df_secteur.columns)
    if still_missing:
        missing_sorted = sorted(list(still_missing))
        logger.error(f"Missing features before prediction: {missing_sorted}")
        raise ValueError(f"Missing features: {missing_sorted}")
    return df_secteur

def model_prediction(payload,model,feature_names):
    """
    Predicts risk levels by municipality and address.
    Returns a dictionary (JSON format) containing municipalities, addresses and predictions.
    example payload: {
        "cities": ["Bassens", "Sainte-Eulalie", "Carbon-Blanc", ...]
        "timestamp": "2026-03-01T12:00:00Z"
    }
    """

    # Prepare data for prediction 
    df_secteur = prepare_data_for_prediction(payload, feature_names, time_series=False)   

    # Make predictions
    predictions = make_predictions(df_secteur, model, feature_names)

    # Extract top predictions
    top_prediction = build_top_predictions(
        df_secteur,
        predictions,
        nb_top=DEFAULT_TOP_K,
        secteur=secteur,
        facteurs_list=facteurs_list,
    )

    # Return prediction results
    logger.info({"status": "success", "data": top_prediction}) # Display top for verification
    return {"status": "success", "data": top_prediction}


def timeline_prediction(payload,model,feature_names):

    # Prepare data for prediction
    df_secteur = prepare_data_for_prediction(payload,feature_names, time_series=True)   

    # Make predictions
    df_secteur['prediction_score'] = make_predictions(df_secteur, model, feature_names)

    timeline_results = []
    for timestamp in df_secteur["prediction_time"].unique():
        df_ts = df_secteur[df_secteur["prediction_time"] == timestamp]
        max_risk = df_ts['prediction_score'].max()
        logger.info(f"Timestamp: {timestamp} - Max Risk Score: {max_risk}")
        risk_level, risk_label = score_to_risk_level(max_risk)
        unix_ts = int(pd.to_datetime(timestamp, unit='s').timestamp())
        timeline_results.append({
            "risk_index": int(max_risk),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "timestamp": unix_ts,
            "temperature_c": df_ts['temperature_c'].mean(), # Average temperature across cities for this timestamp
            "description": df_ts['description'].mode()[0] if not df_ts['description'].mode().empty else "" ,
            "daylight": df_ts['daylight'].mode()[0] if not df_ts['daylight'].mode().empty else 0
        })


    # Return prediction results
    result = {
        "status": "success",
        "data": timeline_results
    }
    logger.info(result) # Display top for verification
    return result

if __name__ == "__main__":

    # Example payload for testing
    example_payload = {
        "cities": ["Bassens", "Sainte-Eulalie", "Carbon-Blanc"],
        "timestamp": "2026-02-10T22:00:00Z"
    }
    print("Running inference engine with example payload:")
    prediction_result = model_prediction(example_payload)
    print("Prediction result:")
    print(prediction_result)    

    print("-----------------------------------")
    # Example payload for testing
    example_payload = {
        "cities": ["Bassens", "Sainte-Eulalie", "Carbon-Blanc"],
        "timestamp": "2027-02-10T22:00:00Z"
    }
    print("Running inference engine with example payload:")
    prediction_result = model_prediction(example_payload)
    print("Prediction result:")
    print(prediction_result)    
