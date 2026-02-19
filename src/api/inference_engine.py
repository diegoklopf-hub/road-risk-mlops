import os
from pathlib import Path
import pandas as pd
import joblib

from src.common_utils import read_yaml
from src.custom_logger import logger
from feature_weather import encode_meteorological_features
from feature_encoder import encode_categorical_values
from prediction import build_shap_factors, build_top_predictions, make_predictions, score_to_risk_level, select_top_predictions


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
TIMELINE_LENGTH = int(API_CFG.timeline_length)

# City list and INSEE mapping
secteur = API_CFG.secteur_insee
data = list(secteur.keys())

def load_encoder_model():
    logger.info("Loading encoder from %s", ENCODER_PATH)
    try:
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        logger.exception("Failed to load encoder from %s", ENCODER_PATH)
        raise RuntimeError(f"Error loading encoder: {e}")
    return encoder


def prepare_data_for_prediction(payload,feature_names,time_series=False):
    logger.info("Received prediction request for %d city(ies)", len(payload.get("cities", [])))
    # Extract INSEE codes via the 'secteur' mapping dictionary
    unknown_cities = [city for city in payload["cities"] if city not in secteur]
    if unknown_cities:
        raise ValueError(f"Unknown cities: {unknown_cities}")
    insee_codes = [secteur[city] for city in payload["cities"]]

    # 2. Load and filter reference data
    road_secteur_path = str(path_prefix + API_CFG.road_secteur_path)

    df = pd.read_csv(road_secteur_path, sep=";")
    df_secteur = df[df['com'].isin(insee_codes)].copy()

    # 3. Encode meteorological and environmental features
    logger.info("Encoding meteorological and environmental features")
    df_secteur = encode_meteorological_features(
        df_secteur,
        payload["cities"],
        timestamp=payload["timestamp"],
        time_series=time_series,
        secteur=secteur,
        n=TIMELINE_LENGTH,
    )

    # 4. Encode categorical variables
    logger.info("Encoding categorical variables")
    encoder = load_encoder_model()
    df_secteur = encode_categorical_values(df_secteur, encoder, ENCODED_COLS)

    # Security check for API
    still_missing = set(feature_names) - set(df_secteur.columns)
    if still_missing:
        missing_sorted = sorted(list(still_missing))
        logger.error(f"Missing features before prediction: {missing_sorted}")
        raise ValueError(f"Missing features: {missing_sorted}")
    return df_secteur


def model_prediction(payload,model,feature_names,shap_explainer):
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

    X_top, y_top = select_top_predictions(df_secteur,predictions, nb_top=DEFAULT_TOP_K)
    factors = build_shap_factors(X_top, shap_explainer, feature_names, num_explanations=4)
    
    # Extract top predictions
    top_prediction = build_top_predictions(
        X_top,
        y_top,
        secteur=secteur,
        factors=factors,
    )

    # Return prediction results
    logger.info("Prediction completed successfully: %d top record(s)", len(top_prediction))
    return {"status": "success", "top_k": DEFAULT_TOP_K, "data": top_prediction}


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
    logger.info("Timeline prediction completed successfully: %d timestamp(s)", len(timeline_results))
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
