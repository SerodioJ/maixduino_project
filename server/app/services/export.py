import os
import json
import glob


def get_json_content(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content


def export_experiment(exp_id):
    if not (os.path.exists(f"json/{exp_id}/")):
        return None
    files = glob.glob(f"json/{exp_id}/*.json")
    export = []
    for file in files:
        export.append(get_json_content(file))
    return export


def get_exp_ids():
    exps = glob.glob(f"json/*")
    exps = [exp.split("/")[-1] for exp in exps]
    return exps
