import io
import json
import base64
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from PIL import Image
from time import perf_counter

from services.save import json_file, image_file
from schemas import PredictBody, Operations
from .utils import update_sample
from settings import settings

router = APIRouter()


@router.post("/{op}")
async def predict(request: Request, body: PredictBody, op: Operations):
    start = perf_counter()
    image_bytes = io.BytesIO()
    base64.decode(io.BytesIO(body.image), image_bytes)
    image = Image.open(image_bytes)
    exp_id = request.app.state.experiment
    if exp_id == None:
        return JSONResponse(
                status_code=404,
                content={
                    "message": "no experiment found, it needs to be initialized"
                },
            )
    sample = update_sample(request)
    image_file(exp_id, sample, image)
    results = request.app.state.model[op].predict(image)
    if op == Operations.detect:
        response = {}
        for elem in results:
            response[elem["name"]] = response.get(elem["name"], 0) + 1
    response = response if response else results
    save_result = {
        "net": settings.model_dict[op],
        "net_result": results,
        "time_device": body.time,
        "time_server": perf_counter() - start,
        "response": response,
        "req_len": int(request.headers["content-length"]),
        "res_len": len(json.dumps(response)),
        "tag": body.tag
    }
    json_file(exp_id, sample, json.dumps(save_result))
    return response
