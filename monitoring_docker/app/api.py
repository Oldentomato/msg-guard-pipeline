from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from .schemas import Wine, Rating, feature_names
from .monitoring import instrumentator
from data_preprocessing import MsgPredict

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
model = load(ROOT_DIR / "artifacts/model.joblib")


instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.get("/")
def root():
    return "Online Check"
    


@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine):
    features = MsgPredict(sample.dict())


    # sample_dict = sample.dict()
    # features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    # prediction = model.predict(features)[0]
    prediction = model.predict(features)
    response.headers["X-model-Msg"] = str(prediction)
    return prediction


@app.post("/urlpredict")
def url_predict():
    return 0


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}