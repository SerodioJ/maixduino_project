from fastapi import FastAPI

from routers import experiment, result, predict
from services import ImageClassification, ImageDetection
from settings import settings

app = FastAPI()

app.state.experiment = None
app.state.sample = 0
app.state.model = {
    "classification": ImageClassification(settings.class_model),
    "detection": ImageDetection(settings.detect_model),
}


app.include_router(experiment.router, prefix="/experiment")
app.include_router(result.router, prefix="/result")
app.include_router(predict.router, prefix="/predict")
