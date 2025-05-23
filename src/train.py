

import joblib
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_prepare, tokenize_texts, encode_targets, MAX_WORDS, MAX_LEN

# === Load and prepare ===
df = load_and_prepare("data/raw/dataset.csv")
tokenizer, X = tokenize_texts(df["text"])
lbl_tone, y_tone, mlb_lvl1, y_lvl1, mlb_lvl2, y_lvl2 = encode_targets(df)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_tone, stratify=y_tone, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# === Tone LSTM ===
tone_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
tone_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tone_model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=32)

# === Level 1 ===
X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(X, y_lvl1, test_size=0.2, random_state=42)
model_lvl1 = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_lvl1.shape[1], activation='sigmoid')
])
model_lvl1.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['recall'])
model_lvl1.fit(X_train_l1, y_train_l1, validation_data=(X_test_l1, y_test_l1), epochs=10, batch_size=32)

# === Level 2 ===
X_tr2, X_te2, y_tr1, y_te1, y_tr2, y_te2 = train_test_split(X, y_lvl1, y_lvl2, test_size=0.2, random_state=42)

input_text = Input(shape=(MAX_LEN,))
x = Embedding(MAX_WORDS, 64)(input_text)
x = LSTM(64)(x)

input_lvl1 = Input(shape=(y_lvl1.shape[1],))
combined = Concatenate()([x, input_lvl1])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.3)(combined)
output = Dense(y_lvl2.shape[1], activation='sigmoid')(combined)

model_lvl2 = Model(inputs=[input_text, input_lvl1], outputs=output)
model_lvl2.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['recall'])
model_lvl2.fit([X_tr2, y_tr1], y_tr2, validation_data=([X_te2, y_te1], y_te2), epochs=10, batch_size=32)

# === Save ===
tone_model.save("models/tone_lstm_model.h5")
model_lvl1.save("models/level1_lstm_model.h5")
model_lvl2.save("models/level2_lstm_model.h5")

joblib.dump(tokenizer, "models/tokenizer.pkl")
joblib.dump(lbl_tone, "models/tone_label_encoder.pkl")
joblib.dump(mlb_lvl1, "models/level1_mlb.pkl")
joblib.dump(mlb_lvl2, "models/level2_mlb.pkl")
