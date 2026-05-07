import pandas as pd
import joblib
import os

# ---------------------------
# ⚙️ CONFIG SAFE LOAD
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------------------
# 🔮 SINGLE PREDICTION FUNCTION
# ---------------------------
def predict_price(input_dict, feature_columns):
    """
    Predict car price from input dictionary
    and aligned feature columns
    """

    # Fill missing columns with 0
    safe_input = input_dict.copy()

    for col in feature_columns:
        if col not in safe_input:
            safe_input[col] = 0

    # Convert to DataFrame
    df = pd.DataFrame([safe_input])

    # Align columns exactly
    df = df[feature_columns]

    # Predict
    prediction = model.predict(df)[0]

    return float(prediction)