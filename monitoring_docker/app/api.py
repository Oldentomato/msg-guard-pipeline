from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from .schemas import Wine, Rating, feature_names
from .monitoring import instrumentator
from data_preprocessing import MsgPredict
from predict import predict

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()


instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.get("/")
def root():
    return "Online"
    


@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine):
    features = MsgPredict(sample.dict())

    prediction = predict(features)
    response.headers["X-model-Msg"] = str(prediction)
    return prediction


@app.post("/urlpredict")
def url_predict():
    return 0


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}