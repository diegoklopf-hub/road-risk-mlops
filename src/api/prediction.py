from pathlib import Path
import pandas as pd
import numpy as np

from src.common_utils import read_yaml
from src.custom_logger import logger
from src.data_processing.schema_manager import SchemaManager
from src.config import SCHEMA_FILE_PATH

CONFIG = read_yaml(Path("src/config.yaml"))
API_CFG = CONFIG.api_inference


def _load_risk_levels_from_config():
    raw_levels = getattr(API_CFG, "risk_levels", None)

    if not raw_levels:
        return [
            {"level": 10, "threshold": 85.0, "label": "CRITIQUE"},
            {"level": 9, "threshold": 80.0, "label": "TRÈS ÉLEVÉ"},
            {"level": 8, "threshold": 77.5, "label": "ÉLEVÉ"},
            {"level": 7, "threshold": 70.0, "label": "SIGNIFICATIF"},
            {"level": 6, "threshold": 55.0, "label": "MODÉRÉ"},
            {"level": 5, "threshold": 40.0, "label": "MOYEN"},
            {"level": 4, "threshold": 30.0, "label": "FAIBLE"},
            {"level": 1, "threshold": None, "label": "NÉGLIGEABLE"},
        ]

    normalized = []
    for item in raw_levels:
        level = int(item.get("level"))
        threshold_value = item.get("threshold")
        threshold = None if threshold_value is None else float(threshold_value)
        label = str(item.get("label"))
        normalized.append({"level": level, "threshold": threshold, "label": label})

    normalized.sort(
        key=lambda x: float("-inf") if x["threshold"] is None else x["threshold"],
        reverse=True,
    )
    return normalized


RISK_LEVELS = _load_risk_levels_from_config()


def score_to_risk_level(score):
    """Convert a raw score into a risk level and label from config thresholds."""
    for risk in RISK_LEVELS:
        threshold = risk["threshold"]
        if threshold is None:
            return risk["level"], risk["label"]
        if score > threshold:
            return risk["level"], risk["label"]

    return 1, "NÉGLIGEABLE"


def make_predictions(df, model, feature_names):
    # Select columns in the exact order expected by the model
    logger.info("Making predictions")
    X = df[feature_names].copy()
    X = X.replace({True: 1, False: 0})  # Convert booleans for the model
    return model.predict(X.replace({True: 1, False: 0}))


def build_top_predictions(df, predictions, secteur, factors):
    # Format JSON response
    # Build the response by associating municipalities, addresses, and results
    insee_to_name = {v: k for k, v in secteur.items()}
    results = []
    for i in range(len(df)):
        # Retrieve the raw INSEE code (ensure it is an int to match the reverse dict)
        try:
            code_insee_raw = int(float(df["com"].iloc[i]))
            nom_commune = insee_to_name.get(code_insee_raw, f"Unknown ({code_insee_raw})")
        except (ValueError, TypeError):
            nom_commune = "Invalid Code"

        risk_level, risk_label = score_to_risk_level(predictions[i])

        results.append({
            "commune": nom_commune,
            "adresse": str(df["adr"].iloc[i]),
            "facteurs": factors[i] if i < len(factors) else "N/A",
            "prediction": round(float(predictions[i]), 4),
            "risk_level": risk_level,
            "risk_label": risk_label
        })

    # Sort by descending prediction
    results_sorted = sorted(results, key=lambda x: x["prediction"], reverse=True)

    # Extract top
    return results_sorted

def select_top_predictions(X, y, nb_top):
    """
    Selects the top `nb_top` rows from X and y with the highest values in y.

    Args:
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target.
        nb_top (int): Number of top rows to select.

    Returns:
        X_top, y_top: Corresponding subsets (same type as X and y).
    """
    # Convert y to a numpy array for processing
    y_values = y.values if isinstance(y, pd.Series) else np.asarray(y)

    # Find the indices of the top `nb_top` values in y
    top_indices = np.argsort(y_values)[-nb_top:][::-1]  # Descending order

    # Select the corresponding rows
    if isinstance(X, pd.DataFrame):
        X_top = X.iloc[top_indices]
        y_top = y.iloc[top_indices] if isinstance(y, pd.Series) else y_values[top_indices]
    else:
        X_top = X[top_indices]
        y_top = y_values[top_indices]

    return X_top, y_top

def build_shap_factors(X, explainer, feature_names, num_explanations=4):
    """
    Builds a list of explanatory factors based on SHAP values with a 5% significance threshold.
    
    Args:
        X (pd.DataFrame or np.ndarray): Features for which to build explanatory factors.
        explainer: Pre-built SHAP explainer instance.
        feature_names (list): List of feature names in the order they appear in X.
        num_explanations (int): Number of top influential features to extract per sample. Default: 8.
    
    Returns:
        list: List of explanatory factors for each row in X.
    """
    try :
        # Ignore main features common for all roads of sector (e.g., location)
        ignored_prefixes = {"dep", "long", "lat", "locp"}
        schema_manager = SchemaManager(read_yaml(SCHEMA_FILE_PATH))
        MIN_PERCENT_THRESHOLD = 0.05
        # Compute SHAP values
        try:
            shap_result = explainer(X[feature_names].copy())
            all_shap_values = getattr(shap_result, "values", shap_result)
            logger.info("SHAP values computed successfully.")
        except Exception as e:
            logger.exception(f"Error computing SHAP values: {e}")
            return []

        factors = []

        # Process each sample
        for sample_idx in range(X.shape[0]):
            sample_factors = []
            # Identify relevant features (excluding ignored prefixes and time-lag features)
            current_sample_shap = all_shap_values[sample_idx]
            
            # Compute total positive contribution for ratio calculation
            positive_contributions = current_sample_shap[current_sample_shap > 0]
            total_positive_influence = np.sum(positive_contributions)
            
            if total_positive_influence == 0:
                # Avoid division by zero; if no positive influence, return no significant factors
                factors.append("Aucun facteur significatif")
                continue

            # Identification and filtering (aggravating + 5% threshold)
            relevant_indices = [
                idx for idx in range(len(feature_names))
                if not any(feature_names[idx].startswith(prefix) for prefix in ignored_prefixes)
                and not feature_names[idx].endswith("-1")
                and current_sample_shap[idx] > 0  # Aggravating factors only
                and (current_sample_shap[idx] / total_positive_influence) >= MIN_PERCENT_THRESHOLD
            ]
            
            if relevant_indices:
                top_indices = sorted(
                    relevant_indices,
                    key=lambda idx: current_sample_shap[idx],
                    reverse=True
                )[:num_explanations]
            else:
                top_indices = []
            
            for idx in top_indices:
                feature = feature_names[idx]
                shap_val = current_sample_shap[idx]
                influence_pct = (shap_val / total_positive_influence) * 100
                
                try:
                    description = schema_manager.get_short_description(feature)
                    # Include influence percentage in final output text
                    sample_factors.append(f"{description} ({influence_pct:.1f}%)")
                    logger.info(f"Sample {sample_idx} - {feature}: {shap_val:.4f} ({influence_pct:.2f}%)")
                except Exception as e:
                    logger.warning(f"Description missing for '{feature}': {e}")
            
            factors.append("   +   ".join(sample_factors) if sample_factors else "Aucun facteur significatif (>5%)")
        
        return factors
    
    except Exception as e:
        logger.exception(f"Error in build_shap_factors: {e}")
        raise e