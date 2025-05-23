import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

MAX_LEN = 100

# Загрузка
tokenizer = joblib.load("models/tokenizer.joblib")
model_tone = load_model("models/model_tone.h5")
model_lvl1 = load_model("models/model_lvl1.h5")
model_lvl2 = load_model("models/model_lvl2.h5")
lbl_tone = joblib.load("models/lbl_tone.joblib")
mlb_lvl1 = joblib.load("models/mlb_lvl1.joblib")
mlb_lvl2 = joblib.load("models/mlb_lvl2.joblib")

def full_predict(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    tone_pred = model_tone.predict(padded).argmax(axis=1)
    tone_labels = lbl_tone.inverse_transform(tone_pred)

    lvl1_pred = (model_lvl1.predict(padded) > 0.5).astype(int)
    lvl1_labels = mlb_lvl1.inverse_transform(lvl1_pred)

    lvl2_pred = (model_lvl2.predict([padded, lvl1_pred]) > 0.5).astype(int)
    lvl2_labels = mlb_lvl2.inverse_transform(lvl2_pred)

    return [
        {"tone": tone, "level_1": l1, "level_2": l2}
        for tone, l1, l2 in zip(tone_labels, lvl1_labels, lvl2_labels)
    ]
