import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, r2_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import io
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Process IMDB dataset
def process_imdb(imdb_glob):
    """Process IMDB dataset to extract PERSON entities using multiprocessing."""
    print("Processing IMDB dataset...")
    files = glob.glob(imdb_glob, recursive=True)
    data = []
    print(f"Found {len(files)} files.")
    
    # Parallel processing with ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_single_file, files), total=len(files), desc="Processing Files"):
            data.extend(result)
    
    df = pd.DataFrame(data)
    print(f"Processed {len(df)} entities from IMDB dataset.")
    return df

# Process a single file
def process_single_file(filepath):
    """Extract PERSON entities and features from a single file."""
    with io.open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        names = get_entity(text)
        sentiment_score = sia.polarity_scores(text)['compound']
        return [
            {
                'name': name,
                'context': text,
                'sentiment_score': sentiment_score,
                'name_len': len(name),
                'name_spaces': name.count(' ')
            }
            for name in names
        ]

# Extract PERSON entities
def get_entity(text):
    """Extract PERSON entities from text."""
    entities = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name = ' '.join(c[0] for c in chunk.leaves())
                entities.append(name)
    return entities

# Feature engineering
def create_features(df):
    """Generate additional features."""
    print("Creating additional features...")
    df['name_len'] = df['name'].apply(len)
    df['name_spaces'] = df['name'].apply(lambda x: x.count(' '))
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df['context'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_score'] = df['sentiment_score'] + 1  # Shift to non-negative
    return df

# Prepare feature matrix
def prepare_features(df, vectorizer):
    """Combine text and numerical features into a single feature matrix."""
    print("Preparing feature matrix...")
    text_features = vectorizer.transform(df['context'])
    additional_features = df[['name_len', 'name_spaces', 'sentiment_score']].values
    return np.hstack((text_features.toarray(), additional_features))

# Train on IMDB dataset
def train_with_imdb(imdb_glob, model_path, vectorizer_path):
    """Train a RandomForest model using the IMDB dataset."""
    print("Training with IMDB dataset...")
    imdb_data = process_imdb(imdb_glob)
    imdb_data = create_features(imdb_data)
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=50000)
    vectorizer.fit(imdb_data['context'])
    
    X_train = prepare_features(imdb_data, vectorizer)
    y_train = imdb_data['name']
    
    # Validate feature matrix and labels
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of labels: {len(y_train)}")
    print(f"Unique labels: {len(np.unique(y_train))}")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("IMDB model training completed.")
    return model, vectorizer

# Fine-tune with unredactor.tsv
def fine_tune_with_tsv(tsv_path, model, vectorizer, model_path):
    """Fine-tune the RandomForest model with unredactor.tsv."""
    print("Fine-tuning with unredactor.tsv...")
    df = pd.read_csv(tsv_path, sep='\t', names=['split', 'name', 'context'])
    df = create_features(df)
    
    train_df = df[df['split'] == 'training']
    val_df = df[df['split'] == 'validation']
    
    # Debug dataset
    print(f"Training data size: {train_df.shape}")
    print(f"Validation data size: {val_df.shape}")
    print(f"Unique training labels: {train_df['name'].nunique()}")
    print(f"Unique validation labels: {val_df['name'].nunique()}")
    
    # Prepare training data
    X_train = prepare_features(train_df, vectorizer)
    y_train = train_df['name']
    model.fit(X_train, y_train)
    
    # Prepare validation data
    X_val = prepare_features(val_df, vectorizer)
    y_val = val_df['name']
    y_pred = model.predict(X_val)
    
    # Evaluate
    print("Validation Results:")
    print(classification_report(y_val, y_pred, zero_division=0))
    print(f"Precision: {precision_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")
    try:
        r2 = r2_score(pd.factorize(y_val)[0], pd.factorize(y_pred)[0])
        print(f"R² Score: {r2:.4f}")
    except Exception as e:
        print("R² Score computation failed:", e)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Fine-tuning completed.")

# Predict on test data
def predict_on_test(test_path, model_path, vectorizer_path):
    """Predict names in the test dataset."""
    print("Predicting on test dataset...")
    test_df = pd.read_csv(test_path, sep='\t', names=['split', 'name', 'context'])
    test_df = create_features(test_df)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    X_test = prepare_features(test_df, vectorizer)
    predictions = model.predict(X_test)
    test_df['predicted_name'] = predictions
    print("Predictions:")
    print(test_df[['context', 'predicted_name']].head())
    return test_df

# Predict on test data and generate submission file
submission_path = "submission.tsv"
def match_predicted_name_length(test_df, predictions, training_names):
    """Ensure predicted names match the length of the redacted block."""
    valid_predictions = []
    
    for i, prediction in enumerate(predictions):
        redacted_length = test_df.loc[i, 'num_redacted_chars']
        
        if len(prediction) == redacted_length:
            valid_predictions.append(prediction)
        else:
            # Find a name from training data with the closest matching length
            fallback_name = next((name for name in training_names if len(name) == redacted_length), "Unknown")
            valid_predictions.append(fallback_name)
    
    return valid_predictions
def predict_on_test(test_path, model_path, vectorizer_path, submission_path, training_names):
    """Predict names in the test dataset and generate a submission file."""
    print("Predicting on test dataset...")
    
    # Load test data without column headers
    test_df = pd.read_csv(test_path, sep='\t', header=None, names=['id', 'context'])
    
    # Load the trained model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Generate features for test data
    test_df = create_features(test_df)
    X_test = prepare_features(test_df, vectorizer)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Match predicted names to redacted length
    predictions = match_predicted_name_length(test_df, predictions, training_names)
    
    # Add predictions to the DataFrame
    test_df['predicted_name'] = predictions
    
    # Generate the submission file
    submission_df = test_df[['id', 'predicted_name']]
    submission_df.to_csv(submission_path, sep='\t', index=False, header=False)
    print(f"Submission file saved to {submission_path}")
    return submission_df

def extract_training_names(training_data):
    """Extract unique names from training data for fallback predictions."""
    return training_data['name'].unique().tolist()



predict_on_test(test_path, model_path, vectorizer_path, submission_path, training_names)
# Main function
if __name__ == "__main__":
    #imdb_glob = "/blue/cis6930/share/aclImdb/train/pos/*.txt"
    imdb_glob = "/blue/cis6930/vi.gade/de/imdbdata/train/pos/*.txt"
    tsv_path = "/home/vi.gade/de2/unredactor.tsv"
    test_path = "/home/vi.gade/de2/test.tsv"
    model_path = "/blue/cis6930/vi.gade/rf_unredactor_model1.pkl"
    vectorizer_path = "/blue/cis6930/vi.gade/rf_count_vectorizer1.pkl"

    training_data = pd.read_csv(tsv_path, sep='\t', names=['split', 'name', 'context'])
    training_names = extract_training_names(training_data)

    
    model, vectorizer = train_with_imdb(imdb_glob, model_path, vectorizer_path)
    fine_tune_with_tsv(tsv_path, model, vectorizer, model_path)
    #predict_on_test(test_path, model_path, vectorizer_path)
