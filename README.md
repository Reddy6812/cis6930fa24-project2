---

# Project 2: The Unredactor

## Overview
The Unredactor is an NLP tool designed to predict and replace redacted names in text. Using a machine learning model trained on contextual data, it identifies the most likely names to fill redacted sections while ensuring the predicted names match the redacted block's length.

---

## Features
- Extracts person names and context from IMDB review files.
- Fine-tunes a `RandomForestClassifier` model using additional redacted training data/we get similar Results with ComplementNB
- Predicts names for redacted blocks in test files.
- Ensures predicted name length matches the redacted block's length.

---

## Installation

### Prerequisites
- Python 3.8+
- `pipenv` for dependency management.

### Install Dependencies
1. Clone the repository:
   ```bash
   https://github.com/Reddy6812/cis6930fa24-project2.git
   cd cis6930fa24-project2
   ```
2. Install dependencies:
   ```bash
   pipenv install
   ```

---

## Usage

### Training and Prediction
1. Train the model and predict names:
   ```bash
   pipenv run python unredactor.py
   ```

2. Run tests:
   ```bash
   pipenv run python -m pytest
   ```

3. Update paths in `unredactor.py` to match your environment:
   - **Training data**: IMDB and redacted data paths.
   - **Test data**: Path to `test.tsv`.
   - **Output**: Path for `submission.tsv`.

---

## Functions Used

1. **`process_imdb(imdb_glob)`**:
   Processes IMDB review files to extract person names and context.

2. **`process_single_file(filepath)`**:
   Processes a single IMDB review file to extract person entities and features.

3. **`get_entity(text)`**:
   Extracts PERSON entities from a given text using NLTK's `ne_chunk`.

4. **`create_features(df)`**:
   Generates additional features for training, including name length, spaces, and sentiment score.

5. **`prepare_features(df, vectorizer)`**:
   Combines text and numerical features into a single feature matrix.

6. **`train_with_imdb(imdb_glob, model_path, vectorizer_path)`**:
   Trains a model using the IMDB dataset and saves the trained model and vectorizer.

7. **`fine_tune_with_tsv(tsv_path, model, vectorizer, model_path)`**:
   Fine-tunes the model using redacted training data and evaluates its performance.

8. **`predict_on_test(test_path, model_path, vectorizer_path, submission_path)`**:
   Predicts names for redacted blocks in the test dataset and generates a submission file.

9. **`match_predicted_name_length(test_df, predictions, training_names)`**:
   Ensures predicted names match the redacted block's length.

---

## Results

### Metrics
- Trained with **1000 IMDB files**:
  - **F1 Score**: 0.292
  - **Precision**: 0.297

### Challenges
1. **Computational Resources**:
   - Requires significant runtime memory and disk space.
2. **Data Limitations**:
   - Full dataset training was not possible due to resource constraints.
3. **Context Understanding**:
   - Accurate predictions require extensive data to classify context effectively.
4. **Prediction Length**:
   - Ensuring predicted name lengths match redacted block lengths.

---

## Known Bugs/Issues
- Occasionally struggles with gender-neutral labels.
- Predictions may not always align semantically with the context.

---

## Example Output

### Input (`test.tsv`):
```
1   His wife, scheming ██████████████, longs for the finer things in life.
2   This movie starred ██████████████ in a great role.
```

### Output (`submission.tsv`):
```
1   Catherine Zeta
2   William Holden
```

---

## Future Improvements
1. Optimize memory and disk usage during training.
2. Train on the full IMDB dataset for better generalization.
3. Introduce gender-specific and gender-neutral classifications.
4. Incorporate advanced context-classification techniques for better predictions.

---

## Author
**Vijay Kumar Reddy Gade**  
Email: [vi.gade@ufl.edu](mailto:vi.gade@ufl.edu)

---
