import pandas as pd
import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pickle
import glob
import io
import os
from concurrent.futures import ProcessPoolExecutor

# Set environment variables to utilize all CPUs
os.environ["OMP_NUM_THREADS"] = "-1"
os.environ["MKL_NUM_THREADS"] = "-1"

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to extract PERSON entities
def get_entity(text):
    """Extract PERSON entities from text."""
    entities = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name = ' '.join(c[0] for c in chunk.leaves())
                entities.append(name)
    return entities

# Process a single IMDB file
def process_single_file(filepath):
    """Process a single IMDB file to extract PERSON entities and features."""
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

# Process IMDB dataset to extract names and contexts
def process_imdb(imdb_glob):
    """Process IMDB dataset in parallel to extract PERSON entities and contexts."""
    print("Executing: process_imdb")
    files = glob.glob(imdb_glob, recursive=True)
    data = []
    print(f"Found {len(files)} files in IMDB dataset...")
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_single_file, files), desc="Processing IMDB files", total=len(files)):
            data.extend(result)
    df = pd.DataFrame(data)
    print(f"Processed {len(df)} entities from IMDB dataset.")
    return df

# Create additional features
def create_additional_features(df):
    """Create additional features from the dataset."""
    print("Executing: create_additional_features")
    tqdm.pandas(desc="Creating additional features")
    df['name_len'] = df['name'].progress_apply(len)
    df['name_spaces'] = df['name'].progress_apply(lambda x: x.count(' ') if isinstance(x, str) else 0)
    df['sentiment_score'] = df['context'].progress_apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

# Prepare feature matrix
def prepare_feature_matrix(df, countvec):
    """Combine CountVectorizer matrix with additional features."""
    print("Executing: prepare_feature_matrix")
    tqdm.pandas(desc="Vectorizing context data")
    text_features = countvec.transform(df['context'])
    additional_features = df[['name_len', 'name_spaces', 'sentiment_score']].values
    return np.hstack((text_features.toarray(), additional_features))

# Train model using IMDB dataset
def train_with_imdb(imdb_glob, model_path, vectorizer_path):
    """Train a model using the IMDB dataset."""
    print("Executing: train_with_imdb")
    imdb_data = process_imdb(imdb_glob)
    
    if imdb_data.empty:
        print("No entities extracted from IMDB dataset. Exiting...")
        return None, None

    imdb_data = create_additional_features(imdb_data)
    
    # Vectorize the context
    countvec = CountVectorizer(ngram_range=(1, 2), max_features=10000, n_jobs=-1)
    countvec.fit(imdb_data['context'])
    
    # Prepare feature matrix
    X_train_imdb = prepare_feature_matrix(imdb_data, countvec)
    y_train_imdb = imdb_data['name']
    
    # Train the model
    model = ComplementNB()
    print("Training on IMDB dataset...")
    model.fit(X_train_imdb, y_train_imdb)
    
    # Save intermediate model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(countvec, f)

    print("Training with IMDB dataset completed.")
    return model, countvec

# Fine-tune the model with unredactor.tsv
def fine_tune_with_tsv(tsv_path, model, countvec, model_path):
    """Fine-tune the model using the unredactor.tsv file."""
    print("Executing: fine_tune_with_tsv")
    unredactor_data = pd.read_csv(tsv_path, sep='\t', names=['split', 'name', 'context'])
    unredactor_data = create_additional_features(unredactor_data)
    
    # Split data
    train_data = unredactor_data[unredactor_data['split'] == 'training']
    val_data = unredactor_data[unredactor_data['split'] == 'validation']
    
    # Prepare feature matrices
    X_train_unredactor = prepare_feature_matrix(train_data, countvec)
    y_train_unredactor = train_data['name']
    X_val_unredactor = prepare_feature_matrix(val_data, countvec)
    y_val_unredactor = val_data['name']
    
    # Fine-tune the model
    model.partial_fit(X_train_unredactor, y_train_unredactor, classes=np.unique(y_train_unredactor))
    
    # Evaluate on validation set
    y_pred = model.predict(X_val_unredactor)
    print("Validation Results (Fine-tuned Model):")
    print(classification_report(y_val_unredactor, y_pred))
    
    # Save final model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("Fine-tuning completed.")

# Predict names using test.tsv
def predict_on_test(test_path, model_path, vectorizer_path):
    """Predict names on test.tsv file."""
    print("Executing: predict_on_test")
    test_data = pd.read_csv(test_path, sep='\t', names=['split', 'name', 'context'])
    test_data = create_additional_features(test_data)
    
    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        countvec = pickle.load(f)
    
    # Prepare test feature matrix
    X_test = prepare_feature_matrix(test_data, countvec)
    
    # Predict
    predictions = model.predict(X_test)
    test_data['predicted_name'] = predictions
    print("Predictions on test data:")
    print(test_data[['context', 'predicted_name']].head())

# Main Execution
if __name__ == "__main__":
    # Paths
    imdb_glob = "/Users/vijaykumarreddygade/Desktop/de3/aclImdb/train/pos/*.txt"
    tsv_path = "unredactor.tsv"
    model_path = "unredactor_model.pkl"
    vectorizer_path = "count_vectorizer.pkl"
    test_path = "/home/vi.gade/de2/test.tsv"
    
    # Train with IMDB
    model, countvec = train_with_imdb(imdb_glob, model_path, vectorizer_path)
    
    if model and countvec:
        fine_tune_with_tsv(tsv_path, model, countvec, model_path)
        predict_on_test(test_path, model_path, vectorizer_path)
