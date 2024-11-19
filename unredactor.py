import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')



def preprocess_context(context):
    """Remove redacted portions and preprocess context."""
    return re.sub(r'█+', '', context)  # Replace blocks with an empty string


def extract_features(data):
    """
    Extract features for model training.
    :param data: DataFrame with context column.
    :return: Processed feature column.
    """
    data['processed_context'] = data['context'].apply(preprocess_context)
    return data['processed_context']


def load_data(filepath):
    """
    Load and preprocess data from TSV file.
    :param filepath: Path to TSV file.
    :return: Split data into training and validation sets.
    """
    try:
        # Load data with specified column names
        data = pd.read_csv(filepath, sep='\t', header=None, names=['split', 'name', 'context'])
        data.dropna(inplace=True)
        logging.info(f"Loaded data with shape: {data.shape}")

        # Split data into training and validation
        train_data = data[data['split'] == 'training']
        val_data = data[data['split'] == 'validation']
        
        # Extract features and labels
        X_train = extract_features(train_data)
        y_train = train_data['name']
        X_val = extract_features(val_data)
        y_val = val_data['name']

        return X_train, X_val, y_train, y_val
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def build_model():
    """
    Build a text classification model pipeline.
    :return: Scikit-learn Pipeline object.
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipeline


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: None
    """
    report = classification_report(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    logging.info("Model Evaluation Report:")
    print(report)
    logging.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")


def predict_name(model, context):
    """
    Predict the most likely name for a redacted context.
    :param model: Trained model.
    :param context: Context with redacted name.
    :return: Predicted name.
    """
    processed_context = preprocess_context(context)
    return model.predict([processed_context])[0]


if __name__ == "__main__":
    # Filepath to the dataset
    filepath = 'unredactor.tsv'

    # Load data
    X_train, X_val, y_train, y_val = load_data(filepath)

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Evaluate the model
    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)

    # Example prediction
    example_context = """After Zentropa, █████████ moved away from this type of audacious technical experiment."""
    predicted_name = predict_name(model, example_context)
    logging.info(f"Predicted name for context: {predicted_name}")
