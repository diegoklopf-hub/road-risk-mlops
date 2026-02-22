import os
from pathlib import Path
import pandas as pd
import joblib

from src.common_utils import read_yaml
from src.custom_logger import logger
from feature_weather import encode_meteorological_features
from feature_encoder import encode_categorical_values
from prediction import build_shap_factors, build_top_predictions, make_predictions, score_to_risk_level, select_top_predictions

# Prometheus metrics imports
from prometheus_client import Counter, Histogram, Gauge
import time

# ─── Prometheus Metrics ───────────────────────────────────────────────────

# Total number of predictions (labels: status=success|error)
prediction_total = Counter(
    "ml_prediction_total",
    "Nombre total d'appels à model_prediction",
    ["status"]
)

# End-to-end pipeline latency
prediction_latency_seconds = Histogram(
    "ml_prediction_latency_seconds",
    "Latence end-to-end de model_prediction (secondes)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

step_latency_seconds = Histogram(
    "ml_prediction_step_latency_seconds",
    "Latence par étape du pipeline de prédiction",
    ["step"],  # labels: prepare | predict | build_top
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Number of cities in the payload
payload_cities_count = Histogram(
    "ml_prediction_payload_cities_count",
    "Nombre de villes dans chaque payload",
    buckets=[1, 5, 10, 20, 50, 100, 200]
)

# Score of the top prediction
top_prediction_score = Gauge(
    "ml_top_prediction_score",
    "Score de risque de la meilleure prédiction retournée"
)

# Number of top results returned
top_predictions_returned = Histogram(
    "ml_top_predictions_returned_count",
    "Nombre de prédictions dans le top résultat",
    buckets=[1, 5, 10, 20, 50]
)

# Error details
prediction_errors_total = Counter(
    "ml_prediction_errors_total",
    "Erreurs par type dans model_prediction",
    ["error_type"]  # labels: prepare_data | make_predictions | build_top | unknown
)

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
    start_total = time.time()

    try:
        # Prepare data for prediction 
        with step_latency_seconds.labels(step="prepare").time():
            df_secteur = prepare_data_for_prediction(payload, feature_names, time_series=False)   

        # Make predictions
        with step_latency_seconds.labels(step="predict").time():
            predictions = make_predictions(df_secteur, model, feature_names)

        # Extract top predictions
        with step_latency_seconds.labels(step="build_top").time():
            X_top, y_top = select_top_predictions(df_secteur,predictions, nb_top=DEFAULT_TOP_K)
            factors = build_shap_factors(X_top, shap_explainer, feature_names, num_explanations=4)
            top_prediction = build_top_predictions(
                X_top,
                y_top,
                secteur=secteur,
                factors=factors,
            )
        # Metrics on prediction results
        if top_prediction:
            top_predictions_returned.observe(len(top_prediction))

        # Metrics for the top item
        first_score = top_prediction[0].get("prediction")
        if first_score is not None:
            top_prediction_score.set(float(first_score))

        # Return prediction results
        prediction_total.labels(status="success").inc()
        # Track the number of cities in the payload
        payload_cities_count.observe(len(payload.get("cities", [])))
        logger.info("Prediction completed successfully: %d top record(s)", len(top_prediction))
        return {"status": "success", "top_k": DEFAULT_TOP_K, "data": top_prediction}
    except Exception as e:
        # Classify the error based on the likely pipeline stage
        error_type = error_classifier(e)
        prediction_errors_total.labels(error_type=error_type).inc()
        prediction_total.labels(status="error").inc()
        logger.error(f"Prediction failed [{error_type}]: {e}")
        raise

    finally:
        # Always record total latency
        prediction_latency_seconds.observe(time.time() - start_total)

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
    
def error_classifier(exception: Exception) -> str:
    """Classify exceptions for the Prometheus label."""
    name = type(exception).__name__.lower()
    if "prepare" in str(exception).lower() or "keyerror" in name or "valueerror" in name:
        return "prepare_data"
    if "predict" in str(exception).lower():
        return "make_predictions"
    if "top" in str(exception).lower():
        return "build_top"
    return "unknown"

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
