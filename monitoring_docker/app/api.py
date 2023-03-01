from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from .monitoring import instrumentator
from predict import prediction
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()

class MSG(BaseModel):
    id: int
    msg_body: str

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.get("/")
def root():
    return "Online"
    


@app.post("/predict")
def predict(msg: MSG):
    result = prediction(msg)
    return {"spam": result}


@app.post("/urlpredict")
def url_predict():
    return 0


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}