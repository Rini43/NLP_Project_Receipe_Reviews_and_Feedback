from pathlib import Path
import joblib


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
# LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_rating(review):
    """
    Predict sentiment for a recipe review.

    Parameters
    ----------
    review : str
        Recipe review entered by the user.

    Returns
    -------
    prediction : int
        Predicted class.

    label : str
        Predicted sentiment.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if review is None:
        raise ValueError(
            "Review cannot be None."
        )

    review = str(review).strip()

    if not review:
        raise ValueError(
            "Review cannot be empty."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        [review]
    )[0]

    prediction = int(prediction)

    # --------------------------------------------------------
    # Convert prediction to label
    # --------------------------------------------------------

    label = LABEL_MAPPING.get(
        prediction,
        str(prediction)
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return prediction, label


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_review = (
        "This recipe was delicious and very easy to make!"
    )

    prediction, label = predict_rating(
        test_review
    )

    print("=" * 50)
    print("RECIPE SENTIMENT PREDICTION")
    print("=" * 50)

    print(
        "\nReview:",
        test_review
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Sentiment:",
        label
    )