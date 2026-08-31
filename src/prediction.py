from pathlib import Path
import pickle

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = MODEL_DIR / "lstm_model.keras"

TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.pkl"

CONFIG_PATH = MODEL_DIR / "config.pkl"


# ============================================================
# LOAD LSTM MODEL
# ============================================================

model = load_model(MODEL_PATH)


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(TOKENIZER_PATH, "rb") as file:

    tokenizer = pickle.load(file)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(LABEL_MAPPING_PATH, "rb") as file:

    id_to_label = pickle.load(file)


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "rb") as file:

    config = pickle.load(file)


MAX_LENGTH = config["max_length"]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_rating(review):

    # Convert review to string
    review = str(review)

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Get prediction probabilities
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Get predicted class
    prediction = int(
        np.argmax(probabilities)
    )

    # Get confidence
    confidence = float(
        probabilities[prediction]
    )

    return prediction, confidence