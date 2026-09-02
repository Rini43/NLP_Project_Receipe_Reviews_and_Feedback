from pathlib import Path
import joblib
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = MODEL_DIR / "recipe_sentiment_model.pkl"


# ============================================================
# LOAD TF-IDF + SVM PIPELINE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# LABEL MAPPING
# ============================================================

# Change these labels if your notebook uses a different mapping.
LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_rating(review):
    """
    Predict sentiment for a recipe review.

    Parameters
    ----------
    review : str
        User's recipe review.

    Returns
    -------
    prediction : int
        Predicted class number.

    label : str
        Predicted sentiment label.

    confidence : float
        Prediction confidence as a value between 0 and 1.
    """

    # Validate input
    if review is None:
        raise ValueError("Review cannot be None.")

    review = str(review).strip()

    if not review:
        raise ValueError("Review cannot be empty.")


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = int(model.predict([review])[0])

    # Convert class number to label
    label = LABEL_MAPPING.get(
        prediction,
        str(prediction)
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]

        confidence = float(
            np.max(probabilities)
        )


    return prediction, label, confidence


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_review = (
        "This recipe was delicious and very easy to make!"
    )

    prediction, label, confidence = predict_rating(
        test_review
    )

    print("Review:", test_review)
    print("Prediction:", prediction)
    print("Sentiment:", label)

    if confidence is not None:
        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )