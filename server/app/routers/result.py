import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.save import json_file
from services.export import export_experiment, get_exp_ids
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
        "tag": body.tag,
    }
    json_file(exp_id, sample, json.dumps(save_result))
    return OK_MSG


@router.get("/export/{id}")
async def export(request: Request, id: str):
    exp_content = export_experiment(id)
    if exp_content is None:
        return JSONResponse(
            status_code=404,
            content={"message": "experiment not found"},
        )
    return exp_content


@router.get("/ids")
async def get_ids(request: Request):
    return get_exp_ids()
