# FastAPI app for prediction
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Union
import joblib
import numpy as np
import tensorflow as tf

from src.pipeline import full_predict  # основной предикт-пайплайн

app = FastAPI(title="Multilabel Comment Classifier")

class TextRequest(BaseModel):
    texts: Union[str, List[str]]

@app.post("/predict")
def predict(req: TextRequest):
    texts = req.texts if isinstance(req.texts, list) else [req.texts]
    predictions = full_predict(texts)
    return {"predictions": predictions}
