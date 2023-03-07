from pathlib import Path
from fastapi import FastAPI
# from app.monitoring import instrumentator
from app.predict import prediction
from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methoed=["*"],
#     allow_headers=["*"]
# )


class MSG(BaseModel):
    id: int
    msg_body: str

# instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.get("/")
def root():
    return {"Online":"OK"}
    


@app.post("/predict")
def predict_msg(msg: MSG):
    result = prediction(msg)
    return {"spam": result}


@app.post("/urlpredict")
def url_predict():
    return 0


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}