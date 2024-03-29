import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
import pandas as pd

# Paths to preprocessed data
train_path = Path(__file__).parent.parent / "data" / "preprocessed" / "train_dataset.csv"
val_path = Path(__file__).parent.parent / "data" / "preprocessed" / "val_dataset.csv"
output_path = Path(__file__).parent.parent / "results" / "model_nb.pkl"


def prepare_data(train_path, val_path):
    # Load datasets
    train_dataset = pd.read_csv(train_path)
    val_dataset = pd.read_csv(val_path)

    # Vectorize text
    print("Vectorizing the text...")
    vectorizer = CountVectorizer(min_df=10)
    X_train = vectorizer.fit_transform(train_dataset['essay'])
    X_val = vectorizer.transform(val_dataset['essay'])

    # Compute TF-IDF matrices
    print("Computing the TF-IDF matrices...")
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_val_tfidf = tfidf_transformer.transform(X_val)

    return X_train_tfidf, train_dataset['emotion'], X_val_tfidf, val_dataset['emotion'], vectorizer, tfidf_transformer


def train_and_evaluate(X_train, y_train, X_val, y_val):
    print("Training the model...")
    model = ComplementNB(alpha=1, force_alpha=True, fit_prior=True, class_prior=None, norm=False)
    model.fit(X_train, y_train, sample_weight=1)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    cm = confusion_matrix(y_val, y_pred)
    labels = sorted(y_val.unique())

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.show()

    print(f"Validation Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return model


if __name__ == "__main__":
    # Train and evaluate
    X_train, y_train, X_val, y_val, vectorizer, tfidf_transformer = prepare_data(train_path, val_path)
    model = train_and_evaluate(X_train, y_train, X_val, y_val)

    # Save model + vectorizer
    with open(output_path, 'wb') as f:
        pkl.dump((model, vectorizer, tfidf_transformer), f)
