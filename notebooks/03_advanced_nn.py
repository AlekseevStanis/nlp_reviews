# %% [markdown]
# # 🧠 Multitask Classification: Tone, Level 1, Level 2

# %% 📦 Imports
import pandas as pd
import numpy as np
import re
import nltk
import pymorphy2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.optimizers import Adam
import joblib

# Download Russian stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("russian"))
lemmatizer = pymorphy2.MorphAnalyzer()

# %% [markdown]
# ## 📥 Load and Preprocess Dataset

# %%
df = pd.read_csv(r"C:\Users\StasAndLiza\port\nlp\data\raw\dataset.csv")
df = df.rename(columns={
    "comment_txt": "text",
    "sentiment_txt": "tone",
    "parent_class_txt": "level_1",
    "child_class_txt": "level_2"
})
df["tone"] = df["tone"].str.replace("SENTIMENT_", "", regex=False).str.capitalize()
df = df[df['tone'] != 'UNSPECIFIED']

# Group multilabel structure
df_grouped = df.groupby("text").agg({
    "tone": "first",
    "level_1": lambda x: list(set(x)),
    "level_2": lambda x: list(set(x))
}).reset_index()

# %% [markdown]
# ## 🔹 Tone Classification with Simple LSTM

# Tokenization
MAX_WORDS = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df_grouped["text"])
sequences = tokenizer.texts_to_sequences(df_grouped["text"])
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)

# Encode tone
lbl_tone = LabelEncoder()
y_tone = lbl_tone.fit_transform(df_grouped["tone"])
y_tone_cat = to_categorical(y_tone)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_tone, test_size=0.2, stratify=y_tone, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# LSTM model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=32)

# Plot training history
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Tone Classification Accuracy (LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
y_pred = model.predict(X_test).argmax(axis=1)
print("Tone Classification Report:")
print(classification_report(y_test, y_pred, target_names=lbl_tone.classes_))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=lbl_tone.classes_, yticklabels=lbl_tone.classes_)
plt.title("Confusion Matrix (Tone)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %% [markdown]
# ## 🔹 Level 1 Classification (Multilabel, LSTM)

# Tokenize text
X_seq = pad_sequences(tokenizer.texts_to_sequences(df_grouped["text"]), maxlen=MAX_LEN)
mlb_lvl1 = MultiLabelBinarizer()
y_lvl1 = mlb_lvl1.fit_transform(df_grouped["level_1"])

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_lvl1, test_size=0.2, random_state=42)

# LSTM model for Level 1
model_lvl1 = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')
])

model_lvl1.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['recall'])
history = model_lvl1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Training curve
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title("Training Recall (Level 1)")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
y_pred = (model_lvl1.predict(X_test) > 0.5).astype(int)
print("Classification Report (Level 1):")
print(classification_report(y_test, y_pred, target_names=mlb_lvl1.classes_))

# Class-wise metrics
recalls = recall_score(y_test, y_pred, average=None)
precisions = precision_score(y_test, y_pred, average=None)
f1s = f1_score(y_test, y_pred, average=None)

class_report_df = pd.DataFrame({
    "class": mlb_lvl1.classes_,
    "precision": precisions,
    "recall": recalls,
    "f1": f1s
}).sort_values(by="recall")

# Plot class-wise recall
plt.figure(figsize=(10, 6))
sns.barplot(data=class_report_df, x="recall", y="class", palette="coolwarm")
plt.title("Recall per Class (Level 1)")
plt.xlabel("Recall")
plt.ylabel("Class")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🔹 Level 2 Classification (Using Level 1 as Input)

# Tokenization
X_text = pad_sequences(tokenizer.texts_to_sequences(df_grouped["text"]), maxlen=MAX_LEN)
mlb_lvl2 = MultiLabelBinarizer()
y_lvl2 = mlb_lvl2.fit_transform(df_grouped["level_2"])

# Combined split
X_tr_text, X_te_text, y_tr_lvl1, y_te_lvl1, y_tr_lvl2, y_te_lvl2 = train_test_split(
    X_text, y_lvl1, y_lvl2, test_size=0.2, random_state=42
)

# Dual-input model
input_text = Input(shape=(MAX_LEN,))
x_text = Embedding(MAX_WORDS, 64)(input_text)
x_text = LSTM(64)(x_text)

input_lvl1 = Input(shape=(y_tr_lvl1.shape[1],))
x = Concatenate()([x_text, input_lvl1])
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(y_tr_lvl2.shape[1], activation='sigmoid')(x)

model_lvl2 = Model(inputs=[input_text, input_lvl1], outputs=output)
model_lvl2.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['recall'])

# Training
history_lvl2 = model_lvl2.fit(
    [X_tr_text, y_tr_lvl1], y_tr_lvl2,
    validation_data=([X_te_text, y_te_lvl1], y_te_lvl2),
    epochs=10,
    batch_size=32
)

# Evaluation
y_pred_lvl2 = (model_lvl2.predict([X_te_text, y_te_lvl1]) > 0.5).astype(int)
print("Classification Report (Level 2):")
print(classification_report(y_te_lvl2, y_pred_lvl2, target_names=mlb_lvl2.classes_))

# %% [markdown]
# ## 💾 Save All Models and Preprocessors

# Save tone model
model.save(r"C:\Users\StasAndLiza\port\nlp\models\tone_lstm_model.h5")
joblib.dump(lbl_tone, r"C:\Users\StasAndLiza\port\nlp\models\tone_label_encoder.pkl")
joblib.dump(tokenizer, r"C:\Users\StasAndLiza\port\nlp\models\tokenizer_tone.pkl")

# Save Level 1 model
model_lvl1.save(r"C:\Users\StasAndLiza\port\nlp\models\level1_lstm_model.h5")
joblib.dump(mlb_lvl1, r"C:\Users\StasAndLiza\port\nlp\models\level1_mlb.pkl")
joblib.dump(tokenizer, r"C:\Users\StasAndLiza\port\nlp\models\tokenizer_level1.pkl")

# Save Level 2 model
model_lvl2.save(r"C:\Users\StasAndLiza\port\nlp\models\level2_lstm_model.h5")
joblib.dump(mlb_lvl2, r"C:\Users\StasAndLiza\port\nlp\models\level2_mlb.pkl")
