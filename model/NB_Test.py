import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle as pkl
import pandas as pd

# Paths to training data and model
model_path = Path(__file__).parent.parent / "results" / "model_nb.pkl"
data_path = Path(__file__).parent.parent / "data" / "preprocessed" / "test_dataset.csv"


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model, vectorizer, tfidf_transformer = pkl.load(f)
    return model, vectorizer, tfidf_transformer


def evaluate_model(model, vectorizer, tfidf_transformer, dataset):
    X = vectorizer.transform(dataset['essay'])
    X_tfidf = tfidf_transformer.transform(X)
    y_true = dataset['emotion']

    # Predict using loaded model
    y_pred = model.predict(X_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(y_true.unique())

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.show()

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    # Load model + vectorizer
    model, vectorizer, tfidf_transformer = load_model(model_path)
    # Load dataset
    dataset = pd.read_csv(data_path)
    # Evaluate model
    evaluate_model(model, vectorizer, tfidf_transformer, dataset)
