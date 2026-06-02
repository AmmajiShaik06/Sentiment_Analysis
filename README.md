IMDb Movie Review Sentiment Analysis

Project Overview

This project performs Sentiment Analysis on IMDb movie reviews using Natural Language Processing (NLP) and Machine Learning techniques. The model classifies movie reviews as either Positive(1) or Negative(0) based on the review text.

The project demonstrates the complete Machine Learning pipeline including data preprocessing, feature extraction using TF-IDF, model training with Logistic Regression, and performance evaluation using classification metrics and a confusion matrix for sentiment analysis


Objectives

- Analyze movie reviews from the IMDb dataset.
- Perform text preprocessing and cleaning.
- Convert text data into numerical features using TF-IDF.
- Train a Logistic Regression model for sentiment classification.
- Evaluate model performance using Accuracy, Precision, Recall, and F1-Score.
- Visualize results using a Confusion Matrix.
- model saving using Pickle


Dataset

IMDb Movie Reviews Dataset

The dataset contains movie reviews labeled as:

- Positive
- Negative

Dataset Format

Review| Sentiment
This movie was fantastic and inspiring.| Positive
The plot was boring and predictable.| Negative



Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook



Project Workflow

1. Data Collection

Load IMDb movie reviews dataset from a CSV file.

2. Text Preprocessing

- Convert text to lowercase
- Remove punctuation and special characters
- Remove stopwords
- Clean review text

3. Feature Extraction

Apply TF-IDF Vectorization to convert text into numerical features.

4. Model Training

Train a Logistic Regression model using the processed review data.

5. Model Evaluation

Evaluate performance using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

6. Sentiment Prediction

Predict whether a new movie review is Positive or Negative.


Installation

Install required libraries:

pip install -r requirements.txt

Required packages:

pandas
numpy
scikit-learn
nltk
matplotlib
seaborn


Running the Project

Open Jupyter Notebook:

jupyter notebook


Open:

sentiment_analysis.ipynb

Run all notebook cells sequentially.




Results

Evaluation Metrics:

- Accuracy Score
- Precision
- Recall
- F1-Score

Visualization:

- Confusion Matrix Heatmap

Generated Output Files:

sentiment_model.pkl
tfidf_vectorizer.pkl
confusion_matrix.png


Project Structure

IMDb-Sentiment-Analysis/
│
├── sentiment_analysis.ipynb
├── imdb_dataset.csv
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── confusion_matrix.png
├── requirements.txt
└── README.md


Learning Outcomes

- Natural Language Processing Fundamentals
- Text Cleaning and Preprocessing
- Feature Engineering with TF-IDF
- Machine Learning Classification
- Model Evaluation Techniques
- Sentiment Analysis Applications

  
Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can be used to automatically classify movie reviews as positive or negative. The IMDb dataset provides a practical real-world application of sentiment analysis and text classification techniques.
