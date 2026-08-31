import streamlit as st

from src.prediction import predict_rating


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Recipe Review Sentiment Analysis",
    page_icon="🍴",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🍴 Recipe Review Sentiment Analysis")

st.write(
    "Enter a recipe review below and the trained LSTM model "
    "will automatically predict the sentiment."
)


# ============================================================
# USER INPUT
# ============================================================

review = st.text_area(
    "Enter your recipe review:",
    placeholder="Example: This recipe was delicious and very easy to prepare!"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Analyze Review"):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        try:

            prediction, confidence = predict_rating(review)

            # Convert prediction to integer
            prediction = int(prediction)

            # Sentiment mapping
            sentiment_mapping = {
                0: "Negative",
                1: "Neutral",
                2: "Positive"
            }

            sentiment = sentiment_mapping.get(
                prediction,
                str(prediction)
            )

            # ---------------------------------------------
            # DISPLAY RESULT
            # ---------------------------------------------

            st.success("Review analyzed successfully!")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Sentiment",
                    sentiment
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

            st.info(
                f"Predicted class: {prediction}"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )