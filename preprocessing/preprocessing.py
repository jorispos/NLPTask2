import string
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path


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


file_name = Path(__file__).parent.parent / "data" / "messages.csv"
df = pd.read_csv(file_name)
df = df.drop(columns = ["message_id","response_id","article_id"])
cleaned_df = df
cleaned_df['essay'] = df['essay'].apply(lambda x: clean_text(x))
print(cleaned_df)
