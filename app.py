# Import required libraries

from xml.parsers.expat import model

from pyexpat import model

import streamlit as st
import joblib

from src.prediction import predict_rating

# PAGE CONFIGURATION

st.set_page_config(
    page_title="RecipeSense AI",
    page_icon="🍴",
    layout="wide"
)

# LOAD TRAINED MODEL AND TF-IDF VECTORIZER

# LOAD THE TF-IDF VECTORIZER TRAINED IN THE NOTEBOOK

tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# LOAD THE TRAINED SVM RECIPE RATING MODEL

svm = joblib.load("models/recipe_rating_model.pkl")

# APPLICATION TITLE

st.title("🍴 RecipeSense AI")

st.subheader(
    "Intelligent Recipe Review Analyzer"
)

st.write(
    "Analyze recipe reviews using Natural Language Processing "
    "and Machine Learning."
)

# REVIEW INPUT

review = st.text_area(
    "Enter your recipe review",
    placeholder="Write your review here...",
    height=180
)

# ANALYZE REVIEW

if st.button("Analyze Review"):

    # Check if the review is empty
    if not review.strip():

        st.warning(
            "Please enter a recipe review first."
        )

    else:

        # Convert the review into TF-IDF features
        review_tfidf = tfidf.transform([review])

        # Predict the recipe rating
        prediction = model.predict(review_tfidf)

        # Get prediction probability if available
        try:
            probabilities = model.predict_proba(review_tfidf)

            confidence = probabilities. max()

        except AttributeError:
            confidence = None 

        # CONVERT RATING INTO SENTIMENT

        if prediction in [0, 1, 2]:
            sentiment = "Negative"

        elif prediction == 3:
            sentiment = "Neutral"

        else:
            sentiment = "Positive"

        # DISPLAY RESULTS

        col1, col2, col3 = st.columns(3)

        # PREDICTED RATING

        with col1:
            st.metric(
                "Predicted Rating",
                f"{prediction} ⭐"
            )

        # SENTIMENT

        with col2:
            st.metric(
                "Sentiment",
                sentiment
            )

        # CONFIDENCE

        with col3:

            if confidence is not None:

                st.metric(
                    "Confidence",
                    f"{confidence:.1%}"
                )

            else:

                st.metric(
                    "Confidence",
                    "N/A"
                )

        # DISPLAY ORIGINAL REVIEW

        st.divider()

        st.subheader(" Your Review")

        st.write(review)

        # RESULT MESSAGE 
        
        if prediction >= 4: 
            st.success( f"The model predicts a {prediction}-star " 
                       "positive recipe review." 
                       ) 
        elif prediction == 3: 
            st.info( "The model predicts a 3-star neutral review." 
                    ) 
        else: 
            st.error( 
                f"The model predicts a {prediction}-star " 
                "negative recipe review." 
                ) 
            
        # SIDEBAR 
        
        with st.sidebar:
             
             st.title("🍴 RecipeSense AI") 

             st.write( "An NLP-based recipe review analysis application." 
                      ) 
             
             st.divider()

             st.subheader("🤖 Model")
             
             st.write("TF-IDF + SVM") 

             st.subheader("📚 NLP") 

             st.write( "Text preprocessing → TF-IDF → " 
                      "Support Vector Machine" 
                      )
             st.divider() 
             st.caption( "Developed as an NLP & Machine Learning project."
                        )

