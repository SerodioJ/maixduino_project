import json

from fastapi import APIRouter, Request

from services.save import json_file
from .utils import update_sample
from schemas import ResultBody

router = APIRouter()

OK_MSG = {"message": "OK"}


@router.post("/save")
async def save_result(request: Request, body: ResultBody):
    exp_id = request.app.state.experiment
    sample = update_sample(request)
    save_result = {
        "net": body.net,
        "time_device": body.time,
        "time_server": 0,
        "result": body.result,
        "req_len": int(request.headers["content-length"]),
        "res_len": len(json.dumps(OK_MSG)),
    }
    json_file(exp_id, sample, save_result)
    return OK_MSG
