# Sentiment Analysis on IMDB Movie Reviews

### Overview

This project performs sentiment analysis on IMDB movie reviews using machine learning techniques. The goal is to classify movie reviews as either "positive" or "negative" based on their content. The project includes data preprocessing, feature extraction using TF-IDF, model training with Logistic Regression, and evaluation of the model's performance.

### Dataset

The dataset used is IMDB Dataset.csv, which contains two columns:

review: Text of the movie review.

sentiment: Label indicating the sentiment of the review ("positive" or "negative").

### Dependencies

To run this project, you will need the following Python libraries:

pandas

scikit-learn

nltk (Natural Language Toolkit)

You can install these dependencies using pip:

pip install pandas scikit-learn nltk

### Project Structure

**Data Loading**: The dataset is loaded into a pandas DataFrame for processing.

**Text Preprocessing**:

Removal of special characters and digits.

Conversion to lowercase.

Tokenization and removal of stopwords.

Lemmatization to reduce words to their base forms.

**Feature Engineering**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numerical features.

**Model Training**: A Logistic Regression model is trained on the preprocessed data.

**Model Evaluation**: The model's performance is evaluated using precision, recall, and F1-score metrics.

**Model Saving**: The trained model is saved as fraud_detection_model.pkl for future use.

### Results

The model achieved the following performance metrics:

Precision: 0.88

Recall: 0.90

F1-Score: 0.89

These results indicate that the model performs well in classifying the sentiment of IMDB movie reviews.

### Usage

Training the Model:

Run the Jupyter notebook Text_sentiment_Analysis.ipynb to preprocess the data, train the model, and evaluate its performance.

The trained model will be saved as fraud_detection_model.pkl.

### Future Improvements

Experiment with other machine learning models (e.g., Random Forest, SVM) or deep learning approaches (e.g., LSTM, BERT) for potentially better performance.

Fine-tune hyperparameters of the Logistic Regression model to optimize performance.

Expand the preprocessing steps to handle more complex text features (e.g., n-grams, sentiment lexicons).

### License

This project is open-source and available under the MIT License.
