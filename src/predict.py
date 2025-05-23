import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

# Загрузка моделей и энкодеров
model_tone = load_model("models/tone_lstm_model.h5")
model_lvl1 = load_model("models/level1_lstm_model.h5")
model_lvl2 = load_model("models/level2_lstm_model.h5")

tokenizer = joblib.load("models/tokenizer_tone.pkl")  # общий
lbl_tone = joblib.load("models/tone_label_encoder.pkl")
mlb_lvl1 = joblib.load("models/level1_mlb.pkl")
mlb_lvl2 = joblib.load("models/level2_mlb.pkl")

def predict_pipeline(texts: list[str]) -> pd.DataFrame:
    # Препроцессинг
    sequences = tokenizer.texts_to_sequences(texts)
    X_seq = pad_sequences(sequences, maxlen=MAX_LEN)

    # Предикт тональности
    tone_preds = model_tone.predict(X_seq)
    tone_labels = lbl_tone.inverse_transform(np.argmax(tone_preds, axis=1))

    # Предикт уровня 1
    lvl1_preds = model_lvl1.predict(X_seq)
    lvl1_binary = (lvl1_preds > 0.5).astype(int)
    lvl1_labels = mlb_lvl1.inverse_transform(lvl1_binary)

    # Предикт уровня 2 (используем lvl1 как вход)
    lvl2_preds = model_lvl2.predict([X_seq, lvl1_preds])
    lvl2_binary = (lvl2_preds > 0.5).astype(int)
    lvl2_labels = mlb_lvl2.inverse_transform(lvl2_binary)

    # Сбор в DataFrame
    df_out = pd.DataFrame({
        "text": texts,
        "tone": tone_labels,
        "level_1": lvl1_labels,
        "level_2": lvl2_labels
    })

    return df_out
