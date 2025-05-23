# %% [markdown]
# # 🤖 Multitask Feedback Modeling
# This notebook covers classical and neural network models for:
# - Sentiment classification (tone)
# - Level 1 multilabel classification
# - Level 2 multilabel classification

# %%
# 📦 Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from catboost import CatBoostClassifier
import optuna

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical

# %% [markdown]
# ## 📥 Load & Preprocess Data

# %%
df = pd.read_csv(r"C:\Users\StasAndLiza\port\nlp\data\raw\dataset.csv")

# Rename columns for clarity
df = df.rename(columns={
    "comment_txt": "text",
    "sentiment_txt": "tone",
    "parent_class_txt": "level_1",
    "child_class_txt": "level_2"
})

# Clean tone
df["tone"] = df["tone"].str.replace("SENTIMENT_", "", regex=False).str.capitalize()
df = df[df['tone'] != 'UNSPECIFIED']

# Group for multilabel
df_grouped = df.groupby("text").agg({
    "tone": "first",
    "level_1": lambda x: list(set(x)),
    "level_2": lambda x: list(set(x))
}).reset_index()

# %% [markdown]
# ## 🧠 Model 1 — Tone Classification (TF-IDF + ML + NN)

# %%
# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_grouped["text"])
lbl = LabelEncoder()
y_tone = lbl.fit_transform(df_grouped["tone"])

# %% [markdown]
# ### 🔍 Optuna Tuning: Logistic Regression

# %%
def tone_objective(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    clf = LogisticRegression(C=C, max_iter=1000)
    return cross_val_score(clf, X, y_tone, scoring="accuracy", cv=5).mean()

study_tone = optuna.create_study(direction="maximize")
study_tone.optimize(tone_objective, n_trials=30)

# Fit best model
X_train, X_test, y_train, y_test = train_test_split(X, y_tone, test_size=0.2, stratify=y_tone, random_state=42)
clf_tone = LogisticRegression(**study_tone.best_params, max_iter=1000)
clf_tone.fit(X_train, y_train)
y_pred_lr = clf_tone.predict(X_test)

print("📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr, target_names=lbl.classes_))

# %% [markdown]
# ### 🧠 Simple Neural Network (TF-IDF)

# %%
# One-hot
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

def create_model(trial):
    units = trial.suggest_int("units", 64, 256)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = Sequential()
    model.add(Dense(units, activation=activation, input_dim=X_train.shape[1]))
    model.add(Dropout(dropout))
    model.add(Dense(units // 2, activation=activation))
    model.add(Dense(y_train_cat.shape[1], activation="softmax"))

    optimizer = Adam(lr) if optimizer_name == "adam" else RMSprop(lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def objective(trial):
    model = create_model(trial)
    model.fit(X_train, y_train_cat, validation_split=0.2, epochs=10,
              batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]), verbose=0)
    y_pred = model.predict(X_test).argmax(axis=1)
    return recall_score(y_test, y_pred, average="macro")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# %%
from optuna.trial import FixedTrial
model = create_model(FixedTrial(study.best_params))
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_test, y_test_cat),
                    epochs=15, batch_size=study.best_params["batch_size"], verbose=1)

# %%
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("TF-IDF NN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

y_pred_nn = model.predict(X_test).argmax(axis=1)
print("📊 NN Report:")
print(classification_report(y_test, y_pred_nn, target_names=lbl.classes_))

sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt="d", cmap="Blues",
            xticklabels=lbl.classes_, yticklabels=lbl.classes_)
plt.title("Confusion Matrix (NN)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %% [markdown]
# ## 🧠 Model 2 — Multilabel Level 1 & 2 Classification (LSTM)

# %%
# Tokenize
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df_grouped["text"])
X_seq = pad_sequences(tokenizer.texts_to_sequences(df_grouped["text"]), maxlen=MAX_LEN)

# Multilabel targets
mlb_lvl1 = MultiLabelBinarizer()
y_lvl1 = mlb_lvl1.fit_transform(df_grouped["level_1"])

mlb_lvl2 = MultiLabelBinarizer()
y_lvl2 = mlb_lvl2.fit_transform(df_grouped["level_2"])

# Split
X_train, X_test, y_train_lvl1, y_test_lvl1 = train_test_split(X_seq, y_lvl1, test_size=0.2, random_state=42)
_, _, y_train_lvl2, y_test_lvl2 = train_test_split(X_seq, y_lvl2, test_size=0.2, random_state=42)

# %% [markdown]
# ### 🧠 LSTM — Level 1

# %%
model_lvl1 = Sequential()
model_lvl1.add(Embedding(MAX_WORDS, 64, input_length=MAX_LEN))
model_lvl1.add(LSTM(64))
model_lvl1.add(Dropout(0.3))
model_lvl1.add(Dense(32, activation="relu"))
model_lvl1.add(Dense(y_train_lvl1.shape[1], activation="sigmoid"))

model_lvl1.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["recall"])
model_lvl1.fit(X_train, y_train_lvl1, validation_data=(X_test, y_test_lvl1), epochs=10, batch_size=32)

y_pred_lvl1 = model_lvl1.predict(X_test) > 0.5
print("📊 Level 1 Report:")
print(classification_report(y_test_lvl1, y_pred_lvl1, target_names=mlb_lvl1.classes_))

# %% [markdown]
# ### 🧠 LSTM — Level 2

# %%
model_lvl2 = Sequential()
model_lvl2.add(Embedding(MAX_WORDS, 64, input_length=MAX_LEN))
model_lvl2.add(LSTM(64))
model_lvl2.add(Dropout(0.3))
model_lvl2.add(Dense(32, activation="relu"))
model_lvl2.add(Dense(y_train_lvl2.shape[1], activation="sigmoid"))

model_lvl2.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["recall"])
model_lvl2.fit(X_train, y_train_lvl2, validation_data=(X_test, y_test_lvl2), epochs=10, batch_size=32)

y_pred_lvl2 = model_lvl2.predict(X_test) > 0.5
print("📊 Level 2 Report:")
print(classification_report(y_test_lvl2, y_pred_lvl2, target_names=mlb_lvl2.classes_))
