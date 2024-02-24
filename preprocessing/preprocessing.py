import string
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import nltk
nltk.download('stopwords')

def clean_text(doc):
    doc = doc.lower()
    for char in string.punctuation:
        doc = doc.replace(char, ' ')

    # Split the text into tokens (words) using white space as a delimiter
    tokens = doc.split()
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]

    # Filter out short tokens (length <= 1)
    tokens = [word for word in tokens if len(word) > 1]

    # Join the tokens back into a single string with spaces in between
    doc = " ".join(tokens)
    return doc

# Load train data
file_name_train = Path(__file__).parent.parent / "data" / "train_data.csv"
df_train = pd.read_csv(file_name_train, sep=';')
df_train = df_train.drop(columns=["article_id"])
df_train['essay'] = df_train['essay'].apply(lambda x: clean_text(x))

# Splitting train data into train and val sets
train_size = int(0.7 * len(df_train))
train_dataset = df_train[:train_size]
val_dataset = df_train[train_size:]
train_dataset = train_dataset.reset_index(drop=True)
val_dataset = val_dataset.reset_index(drop=True)

# Load test data
file_name_test = Path(__file__).parent.parent / "data" / "test_data.csv"
df_test = pd.read_csv(file_name_test, sep=';')
df_test = df_test.drop(columns=["article_id"])
df_test['essay'] = df_test['essay'].apply(lambda x: clean_text(x))
df_test = df_test.reset_index(drop=True)

# Display the datasets (optional)
print("Train Dataset:")
print(train_dataset.head())
print("Length of Train Dataset:", len(train_dataset))
print("\nValidation Dataset:")
print(val_dataset.head())
print("Length of Validation Dataset:", len(val_dataset))
print("\nTest Dataset:")
print(df_test.head())
print("Length of Test Dataset:", len(df_test))
