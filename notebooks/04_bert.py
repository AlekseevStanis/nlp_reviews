# %% [markdown]
# # 🎯 BERT Tone Classification (RuBERT)
# Многоклассовая классификация пользовательских комментариев: позитив / негатив / нейтральный.
# Мы используем предобученную модель `DeepPavlov/rubert-base-cased` и дообучаем её на задаче классификации тональности.

# %% 📦 Импорты и системная настройка
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel

# 🔇 Отключаем лишние логи и включаем GPU (если есть)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU доступен.")
    except RuntimeError as e:
        print("⚠️ Не удалось настроить GPU:", e)
else:
    print("❌ GPU не найден, будет использоваться CPU.")
#%%
import tensorflow as tf
print("✅ TF version:", tf.__version__)
print("📦 GPU devices:", tf.config.list_physical_devices("GPU"))
#%%

df = pd.read_csv(r"C:\Users\StasAndLiza\port\nlp\data\raw\dataset.csv")

# Переименование колонок для удобства
df = df.rename(columns={
    "comment_txt": "text",
    "sentiment_txt": "tone",
    "parent_class_txt": "level_1",
    "child_class_txt": "level_2"
})

# Приведение тональности к чистому виду
df["tone"] = df["tone"].str.replace("SENTIMENT_", "", regex=False).str.capitalize()
df = df[df["tone"] != "UNSPECIFIED"]

# Кодирование целевой переменной
texts = df["text"].tolist()
lbl = LabelEncoder()
y = lbl.fit_transform(df["tone"])
y_cat = to_categorical(y)

# %% 🔠 Токенизация с использованием RuBERT
MODEL_NAME = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np"
)

X_ids = np.array(encodings["input_ids"])
X_mask = np.array(encodings["attention_mask"])

# %% 🧪 Разделение на обучающую и тестовую выборки
X_train_ids, X_test_ids, y_train, y_test = train_test_split(
    X_ids, y_cat, test_size=0.2, stratify=y_cat.argmax(axis=1), random_state=42
)

X_train_mask, X_test_mask = train_test_split(
    X_mask, test_size=0.2, stratify=y_cat.argmax(axis=1), random_state=42
)

# %% 🧠 Архитектура модели на базе BERT
bert_model = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)

input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

# Извлекаем [CLS] токен для классификации
bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] токен

x = Dropout(0.3)(cls_output)
x = Dense(64, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# %% 🚀 Обучение модели
history = model.fit(
    [X_train_ids, X_train_mask],
    y_train,
    validation_data=([X_test_ids, X_test_mask], y_test),
    epochs=3,
    batch_size=16
)

# %% 📈 График обучения
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("BERT Tone Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% 📊 Оценка качества модели
y_pred = model.predict([X_test_ids, X_test_mask]).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred, target_names=lbl.classes_))

conf = confusion_matrix(y_true, y_pred)
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
            xticklabels=lbl.classes_, yticklabels=lbl.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# %%
