import random

from src.custom_logger import logger

# Risk level thresholds (10 levels)
RISK_THRESHOLD_10 = 85    # Extrême
RISK_THRESHOLD_9 = 80     # Très Élevé (ancien Very High)
RISK_THRESHOLD_8 = 77.5   # Élevé (ancien High)
RISK_THRESHOLD_7 = 70     # Important
RISK_THRESHOLD_6 = 55     # Modéré
RISK_THRESHOLD_5 = 40     # Moyen
RISK_THRESHOLD_4 = 30     # Très Faible
# Below 30 = negligible


def score_to_risk_level(score):
    """Convert a raw score into a risk level (1-10) and label."""
    if score > RISK_THRESHOLD_10:
        return 10, "CRITIQUE"
    if score > RISK_THRESHOLD_9:
        return 9, "TRÈS ÉLEVÉ"
    if score > RISK_THRESHOLD_8:
        return 8, "ÉLEVÉ"
    if score > RISK_THRESHOLD_7:
        return 7, "SIGNIFICATIF"
    if score > RISK_THRESHOLD_6:
        return 6, "MODÉRÉ"
    if score > RISK_THRESHOLD_5:
        return 5, "MOYEN"
    if score > RISK_THRESHOLD_4:
        return 4, "FAIBLE"
    return 1, "NÉGLIGEABLE"


def make_predictions(df, model, feature_names):
    # Select columns in the exact order expected by the model
    logger.info("Making predictions")
    X = df[feature_names].copy()
    X = X.replace({True: 1, False: 0})  # Convert booleans for the model
    return model.predict(X.replace({True: 1, False: 0}))


def build_top_predictions(df, predictions, nb_top, secteur, facteurs_list):
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
            "facteurs": random.choice(facteurs_list),  # Demo-only factors.
            "prediction": round(float(predictions[i]), 4),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "stabilite": random.choice(["↘️ Decrease", " ↗️ Increase", "➡️ Stable"]),
        })

    # Sort by descending prediction
    results_sorted = sorted(results, key=lambda x: x["prediction"], reverse=True)

    # Extract top
    return results_sorted[:nb_top]
