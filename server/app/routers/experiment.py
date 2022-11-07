from uuid import uuid4
import os

from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/new")
async def experiment(request: Request):
    new_id = uuid4()
    request.app.state.experiment = new_id
    request.app.state.sample = 0
    os.mkdir(f"json/{new_id}")
    os.mkdir(f"images/{new_id}")
    return {"id": new_id}


@router.get("/current")
async def experiment(request: Request):
    exp_id = request.app.state.experiment
    sample = request.app.state.sample
    return {"id": str(exp_id), "sample": sample}
