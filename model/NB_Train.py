import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle as pkl

# Paths to preprocessed data
train_path = Path(__file__).parent.parent / "data" / "preprocessed" / "train_dataset.csv"
val_path = Path(__file__).parent.parent / "data" / "preprocessed" / "val_dataset.csv"
output_path = Path(__file__).parent.parent / "model" / "model_nb.pkl"

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(train_path, val_path):
    # Load datasets
    train_dataset = load_data(train_path)
    val_dataset = load_data(val_path)
    
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
    model = ComplementNB()
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    cm = confusion_matrix(y_val, y_pred)

    print(f"Validation Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    return model

# Train and evaluate
X_train, y_train, X_val, y_val, vectorizer, tfidf_transformer = prepare_data(train_path, val_path)
model = train_and_evaluate(X_train, y_train, X_val, y_val)

# Save the model + vectorizer
with open(output_path, 'wb') as f:
    pkl.dump((model, vectorizer, tfidf_transformer), f)
