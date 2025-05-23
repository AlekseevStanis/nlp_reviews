# 🧠 NLP Text Classifier API (Multitask + Multilabel)

## 🔍 Project Overview

This project implements a production-ready pipeline for classifying user comments by:
- **Sentiment**: Positive / Negative
- **Level 1 category**: High-level complaint types (e.g. Prices, Cleanliness, Loyalty Program)
- **Level 2 category**: Fine-grained subcategories (up to 70+ types)

The system is designed to be lightweight and interpretable, serving as an alternative to heavy LLM-based models.

---

## 🏗️ Project Architecture

```
nlp_project/
│
├── data/                   # Raw data (not included in Git)
│
├── models/                 # Trained model files (.h5, .joblib)
│   ├── tone_model.h5
│   ├── level1_model.h5
│   └── level2_model.h5
│
├── src/
│   ├── preprocessing.py    # Text cleaning, tokenization, lemmatization
│   ├── train.py            # Training pipeline for tone, level 1, level 2
│   └── predict.py          # Full pipeline for text classification
│
├── api/
│   └── main.py             # FastAPI server for predictions
│
├── notebooks/              # EDA, modeling, and experiments
│   ├── 01_eda.ipynb
│   ├── 02_modeling_lstm.ipynb
│   └── 03_multitask_modeling.ipynb
│
├── tests/                  # Unit tests
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Features

- ✅ **Text Preprocessing**: Stopwords removal, lemmatization, tokenization
- ✅ **Multilabel Output**: Classifies multiple categories per comment
- ✅ **Multitask Pipeline**: Level 2 depends on predicted Level 1
- ✅ **LSTM Neural Networks**: Efficient model training on modest hardware
- ✅ **FastAPI Serving**: Simple and scalable prediction interface
- ✅ **BERT (optional)**: Ready-to-extend with transformer models

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Unix
pip install -r requirements.txt
```

### 2. Train All Models

```bash
python src/train.py
```

This will generate 3 model files in the `/models` directory.

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔮 Inference Example

You can send a comment to the API and receive a full structured response:

```json
{
  "text": "Цены выросли, кассир хамит и скидки не работают",
  "tone": "Negative",
  "level_1": ["Уровень цен", "Вежливость и отзывчивость персонала", "Скидки и акции"],
  "level_2": ["Некорректные цены", "Грубость персонала", "Отказ в скидке"]
}
```

---

## 📊 Model Evaluation

| Task       | Accuracy / Recall | Model Type     |
|------------|-------------------|----------------|
| Tone       | 93.2%             | LSTM           |
| Level 1    | 89% micro recall  | LSTM multilabel|
| Level 2    | ~76% recall       | LSTM + Level 1 |
| BERT       | *Not deployed*    | Optional       |

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`.

Optional GPU acceleration possible with TensorFlow (>=2.11.0).

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🔐 Notes

- Dataset is not included due to privacy/compliance. Add your labeled `.csv` to `/data/raw`.
- `.gitignore` ensures models, logs, and virtual env are not pushed.

---

## 📍 Future Plans

- [ ] Add Telegram/Slack alerting for failed predictions
- [ ] Explore Quantization & Distillation of large models
- [ ] Add real-time logging with Grafana/Prometheus
- [ ] Switch to `LLama` or `ruGPT-3` for heavy tasks

---

## 👨‍💻 Author

Built by [Your Name] as a lean alternative to LLMs for Russian-language classification in production environments.
