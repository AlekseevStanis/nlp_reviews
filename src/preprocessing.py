# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10000
MAX_LEN = 100

def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "comment_txt": "text",
        "sentiment_txt": "tone",
        "parent_class_txt": "level_1",
        "child_class_txt": "level_2"
    })
    df["tone"] = df["tone"].str.replace("SENTIMENT_", "", regex=False).str.capitalize()
    df = df[df["tone"] != "UNSPECIFIED"]

    df_grouped = df.groupby("text").agg({
        "tone": "first",
        "level_1": lambda x: list(set(x)),
        "level_2": lambda x: list(set(x))
    }).reset_index()

    return df_grouped

def tokenize_texts(texts):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return tokenizer, padded

def encode_targets(df_grouped):
    # Tone
    lbl_tone = LabelEncoder()
    y_tone = lbl_tone.fit_transform(df_grouped["tone"])

    # Level 1
    mlb_lvl1 = MultiLabelBinarizer()
    y_lvl1 = mlb_lvl1.fit_transform(df_grouped["level_1"])

    # Level 2
    mlb_lvl2 = MultiLabelBinarizer()
    y_lvl2 = mlb_lvl2.fit_transform(df_grouped["level_2"])

    return lbl_tone, y_tone, mlb_lvl1, y_lvl1, mlb_lvl2, y_lvl2
