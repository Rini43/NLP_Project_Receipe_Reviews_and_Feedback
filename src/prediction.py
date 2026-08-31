import joblib
from src.preprocessing import preprocess_text

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/recipe_rating_model.pkl"
)


def predict_rating(review):

    cleaned_review = preprocess_text(review)

    vector = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max()

    return prediction, confidence
