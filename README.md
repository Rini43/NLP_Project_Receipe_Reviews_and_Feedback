# Recipe Reviews Sentiment Analysis using Natural Language Processing

An end-to-end Natural Language Processing (NLP) project on the Recipe Reviews and User Feedback dataset, covering data understanding, exploratory data analysis (EDA), text preprocessing, sentiment classification, text vectorization, traditional machine learning models, deep learning with BiLSTM, and model evaluation.

### NLP Project

Dataset: Recipe Reviews and User Feedback Dataset

## Project Overview

This project focuses on analyzing user reviews and feedback for recipes using Natural Language Processing techniques.

The main objective is to classify recipe reviews into three sentiment categories — **Negative, Neutral, and Positive** — based on the star ratings associated with each review.

The project follows a complete NLP workflow, beginning with data exploration and cleaning, followed by text preprocessing and sentiment-label creation. Different text representation techniques and machine learning models are then compared to identify an effective approach for sentiment classification.

The project implements both traditional machine learning techniques and a deep learning approach using a Bidirectional Long Short-Term Memory (BiLSTM) model.

---

## Dataset Information

### Dataset Name

**Recipe Reviews and User Feedback Dataset**

The dataset contains user reviews and feedback associated with recipes. Each record contains information about the recipe, reviewer, review text, ratings, and user engagement.

### Dataset Size

* Initial Records: **18,182**
* Initial Features: **15**
* Text Column: `text`
* Rating Column: `stars`
* Number of Recipe Entries: **100**

### Problem Type

**Multiclass Sentiment Classification**

The project converts the original star ratings into three sentiment categories.

### Target Variable

The sentiment labels are created from the `stars` column:

| Star Rating | Sentiment |
| ----------- | --------- |
| 1–2 Stars   | Negative  |
| 3 Stars     | Neutral   |
| 4–5 Stars   | Positive  |

Reviews with invalid/unrated star values are excluded from the final sentiment-classification dataset.

After sentiment-label creation, the dataset contains:

| Sentiment | Number of Reviews |
| --------- | ----------------: |
| Positive  |            15,107 |
| Negative  |               509 |
| Neutral   |               476 |

The dataset is therefore highly imbalanced, with positive reviews representing the majority of the observations.

---

## Technologies Used

* Python
* Google Colab / Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TensorFlow / Keras
* Joblib
* Regular Expressions (`re`)
* HTML text processing

---

## Project Structure

```text
NLP_Project_Receipe_Reviews_and_Feedback/
│
├── data/
│   └── Recipe Reviews and User Feedback Dataset.csv
│
├── notebooks/
│   └── NLP_Project.ipynb
│
├── README.md
│
└── ...
```

---

## Dataset Features

The dataset contains information from several categories.

### Recipe Information

| Feature         | Description                      |
| --------------- | -------------------------------- |
| `recipe_number` | Numeric identifier of the recipe |
| `recipe_code`   | Unique recipe code               |
| `recipe_name`   | Name of the recipe               |

### User Information

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| `user_id`         | Unique identifier of the reviewer |
| `user_name`       | Name of the reviewer              |
| `user_reputation` | Reputation score of the user      |

### Review Information

| Feature       | Description                             |
| ------------- | --------------------------------------- |
| `comment_id`  | Unique identifier of the review/comment |
| `created_at`  | Review creation timestamp               |
| `reply_count` | Number of replies received              |
| `thumbs_up`   | Number of positive votes                |
| `thumbs_down` | Number of negative votes                |
| `stars`       | Star rating assigned to the recipe      |
| `best_score`  | Review engagement/best-score value      |
| `text`        | Written review provided by the user     |

### Target Variable

The `sentiment` column is generated from the star rating.

| Value    | Meaning         |
| -------- | --------------- |
| Negative | 1–2 star review |
| Neutral  | 3 star review   |
| Positive | 4–5 star review |

---

## Exploratory Data Analysis

The project begins with an initial analysis of the dataset to understand its structure and quality.

The following steps were performed:

* Dataset loading
* Inspection of the first records
* Dataset shape analysis
* Data type inspection
* Statistical summary
* Unique-value analysis
* Missing-value analysis
* Duplicate-value analysis
* Review and recipe analysis
* Star-rating distribution
* Sentiment distribution
* Text-related exploration

The original dataset contains **18,182 rows and 15 columns**. Two reviews contain missing values in the text field and are removed during preprocessing.

---

## Data Preprocessing Pipeline

The following preprocessing steps were applied to prepare the review text for NLP modeling.

### Missing Value Handling

Missing review text values were identified and removed from the dataset.

```python
df_review.dropna(subset=["text"], inplace=True)
```

### Duplicate Analysis

Duplicate records were examined to ensure the quality of the dataset before modeling.

### Text Cleaning

The review text and recipe names were cleaned using several techniques:

* Convert text to lowercase
* Remove punctuation
* Detect and remove URLs
* Decode HTML entities
* Detect and process emojis
* Remove unnecessary characters
* Tokenize text
* Remove English stopwords
* Lemmatize words

The notebook specifically checks for URLs in review text and removes them before further processing.

### Text Normalization

Text was normalized into a consistent format before feature extraction and model training.

---

## Sentiment Label Creation

The original `stars` rating is converted into a three-class sentiment variable.

The mapping used is:

```text
1–2 Stars → Negative
3 Stars   → Neutral
4–5 Stars → Positive
```

Reviews without a valid sentiment label are removed.

The resulting sentiment distribution is:

```text
Positive    15107
Negative      509
Neutral       476
```

This significant class imbalance is an important characteristic of the dataset and should be considered when interpreting model performance.

---

## Train-Test Split

After preprocessing and sentiment-label generation, the review text and sentiment labels are divided into training and testing datasets.

The resulting split used in the notebook is:

| Dataset      | Number of Samples |
| ------------ | ----------------: |
| Training Set |            12,873 |
| Testing Set  |             3,219 |

The training data is used to learn vocabulary and model parameters, while the test data is reserved for final evaluation.

---

# Text Vectorization

Text must be converted into numerical representations before it can be used by traditional machine learning algorithms.

Two classical vectorization techniques were implemented:

* Bag of Words
* TF-IDF

---

## Bag of Words

The Bag of Words representation uses `CountVectorizer` with:

* Maximum features: **5,000**
* N-gram range: **1–2**
* Minimum document frequency: **2**
* Maximum document frequency: **0.95**

The resulting matrices are:

```text
Training Shape : (12873, 5000)
Testing Shape  : (3219, 5000)
Vocabulary Size: 5000
```

The Bag of Words representation was evaluated using:

* Logistic Regression
* Support Vector Machine (SVM)

---

## TF-IDF Vectorization

TF-IDF was implemented using `TfidfVectorizer`.

Configuration:

* Maximum features: **10,000**
* N-gram range: **1–2**
* Minimum document frequency: **2**
* Maximum document frequency: **0.95**
* Sublinear TF enabled

The resulting matrices are:

```text
Training Shape : (12873, 10000)
Testing Shape  : (3219, 10000)
Vocabulary Size: 10000
```

TF-IDF representations were evaluated using:

* Logistic Regression
* Support Vector Machine (SVM)

---

# Machine Learning Methodology

The project evaluates different combinations of text representations and classification algorithms.

### Models Implemented

* Logistic Regression
* Support Vector Machine (SVM)
* Bidirectional LSTM

For the traditional machine learning approaches, class weighting was used to help address the imbalance between sentiment categories.

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

## Logistic Regression

Logistic Regression was trained using both:

* Bag of Words
* TF-IDF

The TF-IDF Logistic Regression model achieved an accuracy of approximately **90.90%** and a weighted F1-score of approximately **92.32%**.

---

## Support Vector Machine

A linear SVM classifier was trained using both:

* Bag of Words
* TF-IDF

The TF-IDF SVM model achieved an accuracy of approximately **92.58%** and a weighted F1-score of approximately **93.04%**.

Among the traditional machine learning models, TF-IDF with SVM produced the strongest overall results.

---

# Deep Learning Model

## BiLSTM

A Bidirectional Long Short-Term Memory (BiLSTM) neural network was implemented to capture sequential and contextual information from the review text.

### Tokenization

The Keras tokenizer was configured with:

```text
Maximum Vocabulary Size = 10,000
Maximum Sequence Length  = 100
```

The reviews were converted into numerical sequences and padded to a fixed length of 100 tokens.

Resulting shapes:

```text
X_train_pad: (12873, 100)
X_test_pad : (3219, 100)
```

### BiLSTM Architecture

The model consists of:

```text
Embedding Layer
       ↓
Bidirectional LSTM
       ↓
Dropout
       ↓
Dense Softmax Output
```

The main configuration includes:

* Embedding dimension: **128**
* LSTM units: **64**
* Bidirectional LSTM
* Dropout: **0.5**
* Output classes: **3**
* Activation function: **Softmax**

Early stopping was used during training to reduce overfitting.

The model was trained for up to 10 epochs with:

* Batch size: **32**
* Validation split: **20%**

---

# Results

The performance of all implemented approaches was evaluated using Accuracy, Precision, Recall, and F1-Score.

| Embedding             | Model               |   Accuracy | Precision |     Recall | F1 Score |
| --------------------- | ------------------- | ---------: | --------: | ---------: | -------: |
| Bag of Words          | Logistic Regression |     90.56% |    93.46% |     90.56% |   91.85% |
| Bag of Words          | SVM                 |     92.05% |    93.02% |     92.05% |   92.50% |
| TF-IDF                | Logistic Regression |     90.90% |    94.22% |     90.90% |   92.32% |
| TF-IDF                | SVM                 |     92.58% |    93.57% |     92.58% |   93.04% |
| Tokenizer + Embedding | LSTM                | **94.53%** |    91.87% | **94.53%** |   92.92% |

The results show that the **BiLSTM model achieved the highest accuracy (94.53%)**, while **TF-IDF with SVM achieved the highest F1-score among the traditional machine learning approaches (93.04%)**.

---

## Model Comparison

### Best Traditional Machine Learning Model

**TF-IDF + SVM**

* Accuracy: **92.58%**
* Precision: **93.57%**
* Recall: **92.58%**
* F1-Score: **93.04%**

### Best Overall Accuracy

**BiLSTM**

* Accuracy: **94.53%**
* Precision: **91.87%**
* Recall: **94.53%**
* F1-Score: **92.92%**

The BiLSTM model provides the highest overall accuracy, demonstrating the benefit of learning sequential patterns and contextual relationships in review text.

---

## Evaluation Metrics

The models were evaluated using the following metrics:

### Accuracy

Measures the percentage of correctly classified reviews.

### Precision

Measures how many reviews predicted as a particular sentiment actually belong to that sentiment.

### Recall

Measures how many reviews belonging to a sentiment class were correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Shows the distribution of correct and incorrect predictions across the three sentiment classes.

---

## Project Workflow

```text
Load Dataset
      │
      ▼
Initial Dataset Analysis
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Missing Value Handling
      │
      ▼
Text Cleaning
      │
      ├── Lowercasing
      ├── Punctuation Removal
      ├── URL Removal
      ├── HTML Cleaning
      ├── Emoji Processing
      ├── Tokenization
      ├── Stopword Removal
      └── Lemmatization
      │
      ▼
Create Sentiment Labels
      │
      ▼
Train-Test Split
      │
      ├───────────────────────┐
      ▼                       ▼
Bag of Words              TF-IDF
      │                       │
      ▼                       ▼
Logistic Regression       Logistic Regression
SVM                       SVM
      │                       │
      └───────────┬───────────┘
                  ▼
          Model Evaluation
                  │
                  ▼
          Deep Learning
                  │
                  ▼
        Tokenization + Padding
                  │
                  ▼
          Embedding Layer
                  │
                  ▼
        Bidirectional LSTM
                  │
                  ▼
             Dropout
                  │
                  ▼
          Softmax Output
                  │
                  ▼
          Final Comparison
```

---

## How to Run the Project

### Option 1: Google Colab (Recommended)

The notebook was developed and executed in Google Colab with GPU support.

1. Clone or download the repository.
2. Open `NLP_Project.ipynb` in Google Colab.
3. Make sure the dataset is available in the expected `data` directory or update the dataset path if necessary.
4. Run the notebook from the first cell to the last cell.
5. Enable GPU acceleration when running the BiLSTM section for faster training.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/Rini43/NLP_Project_Receipe_Reviews_and_Feedback.git
cd NLP_Project_Receipe_Reviews_and_Feedback
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow joblib
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/NLP_Project.ipynb
```

Run the notebook sequentially from the beginning.

---

## Key Findings

* The dataset contains **18,182 recipe reviews** across **15 original features**.
* The majority of reviews are positive.
* Sentiment labels were derived from star ratings.
* Text preprocessing significantly prepares the raw review text for NLP modeling.
* Both Bag of Words and TF-IDF were effective text representation techniques.
* SVM performed better than Logistic Regression for the traditional ML approaches.
* TF-IDF + SVM achieved the strongest traditional ML performance.
* The BiLSTM model achieved the highest overall accuracy of **94.53%**.
* The strong class imbalance means accuracy alone should not be used as the only measure of model quality.
* Confusion matrices and per-class metrics are important for understanding performance on the minority sentiment classes.

---

## Expected Outcome

The project demonstrates an end-to-end NLP workflow for classifying recipe reviews according to sentiment.

The final system compares traditional text-classification approaches with a deep learning architecture and demonstrates how different text representations and modeling techniques affect sentiment-classification performance.

The best-performing model in terms of test accuracy was the **BiLSTM model**, while **TF-IDF + SVM** provided the strongest traditional machine learning result.

---

## Future Improvements

Potential improvements to the project include:

* Addressing the strong sentiment-class imbalance using oversampling or other balancing techniques.
* Performing hyperparameter tuning for the traditional ML models.
* Using pretrained word embeddings such as Word2Vec, GloVe, or FastText.
* Experimenting with transformer-based models such as BERT.
* Using stratified cross-validation for more robust model comparison.
* Evaluating macro-averaged metrics alongside weighted metrics.
* Performing more detailed error analysis on Neutral and Negative reviews.
* Exploring explainable NLP techniques to identify words and phrases driving sentiment predictions.
* Deploying the trained sentiment model as an interactive application.

---

## Conclusion

This project demonstrates the complete process of building a multiclass sentiment-analysis system for recipe reviews.

Starting from raw user feedback, the project performs data exploration, cleaning, text preprocessing, sentiment-label generation, feature extraction, traditional machine learning, and deep learning.

The comparison shows that traditional NLP techniques such as TF-IDF combined with SVM can provide strong performance, while the BiLSTM architecture achieves the highest test accuracy by learning sequential information from the review text.

Overall, the project provides a practical demonstration of how NLP and machine learning can be applied to understand user feedback and automatically classify the sentiment expressed in recipe reviews.

---

## License

This project is intended for educational and academic purposes.

