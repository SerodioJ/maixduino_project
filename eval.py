import os
import glob
import argparse
import json
import yaml
import io
import base64
from time import perf_counter, sleep
import requests
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def extract_args(args):
    is_edge = args.placement == "edge"
    if args.use_all:
        config = (
            ["yolov5s", "yolov5n", "yolov5m", "yolov5l"]
            if is_edge
            else ["yolov5s", "yolov5n", "yolov5m", "yolov5l", "yolov5x"]
        )
    else:
        config = args.nn_config
        if is_edge:
            if "yolov5x" in config:
                config.remove("yolov5x")
    ip_port = f"{args.service_ip}:{8000 if is_edge else 80}"
    images = glob.glob(os.path.join(args.input, f"*.{args.extension}"))
    with open(f"server/kubernetes/{'edge' if is_edge else 'cloud'}.yaml", "r") as f:
        deployment = yaml.safe_load(f)
    return is_edge, config, ip_port, deployment, images, args.output


def payload(img_path):
    img = Image.open(img_path)
    img_bytes = io.BytesIO()
    img.save(img_bytes, img.format)
    img_str = base64.encodebytes(img_bytes.getvalue()).decode("ascii")

    tag = img_path.split("/")[-1]
    return {"image": img_str, "tag": tag}


def update_deployment(deployment, conf):
    config.load_kube_config()
    v1 = client.AppsV1Api()
    try:
        v1.delete_namespaced_deployment(
            name=deployment["metadata"]["name"], namespace="default"
        )
    except ApiException as e:
        pass
    deployment["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"] = conf
    v1.create_namespaced_deployment(namespace="default", body=deployment)
    return


def wait_service(ip_port):
    sleep(10)
    status_code = None
    c = 0
    while True:
        try:
            req = requests.get(f"http://{ip_port}/experiment/current")
            if req.status_code == 200:
                return True
        except:
            pass
        sleep(5)
        c += 1
        print("waiting deployment...")
        if c == 5:
            return False


def evaluate(args):
    is_edge, nn_config, ip_port, deployment, images, output = extract_args(args)
    exec_metrics = {}
    for config in nn_config:
        exec_metrics[config] = {}
        print(f"Executing Experiment for {config}")
        update_deployment(deployment, config)
        if not wait_service(ip_port):
            print(f"Service deployment for {config} failed! Skipping...")
            continue
        start = requests.post(f"http://{ip_port}/experiment/new")
        if start.status_code != 200:
            print("Failed to start experiment")
            return
        exp_id = start.json()["id"]
        error = 0
        exec_metrics[config]["requests"] = {}
        for image in tqdm(images[:100]):
            body = payload(image)
            pred = perf_counter()
            req = requests.post(f"http://{ip_port}/predict/detection", json=body)
            pred = perf_counter() - pred
            if req.status_code != 200:
                error += 1
                print(f"Error on image {image}")
            exec_metrics[config]["requests"][body["tag"]] = {
                "request_time": pred,
                "status_code": req.status_code,
            }
        exec_metrics[config]["id"] = exp_id
        exec_metrics[config]["total"] = len(images)
        exec_metrics[config]["errors"] = error

        print(f"[{config}] Sucessful Requests: {len(images)-error}/{len(images)}")
        results_bundle = requests.get(f"http://{ip_port}/result/export/{exp_id}")
        
        with open(os.path.join(output, f"{exp_id}.json"), "w", encoding="utf-8") as f:
            json.dump(results_bundle.json(), f, indent=4)

        with open(
            os.path.join(output, f"{'edge' if is_edge else 'cloud'}_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(exec_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input", help="path to images directory", type=Path, required=True
    )
    parser.add_argument(
        "-o", "--output", help="path to JSON outputs", type=Path, default=""
    )

    parser.add_argument(
        "-e", "--extension", help="image file extension", type=str, default="jpg"
    )

    parser.add_argument(
        "-s", "--service-ip", help="Service IP address", type=str, required=True
    )

    parser.add_argument(
        "-n",
        "--nn-config",
        help="neural network configs to use",
        nargs="+",
        type=str,
        choices=["yolov5s", "yolov5n", "yolov5m", "yolov5l", "yolov5x"],
    )

    parser.add_argument(
        "-a",
        "--all",
        help="use all NN configurations",
        dest="use_all",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--placement",
        help="where to place service",
        choices=["edge", "cloud"],
        default="edge",
    )

    args = parser.parse_args()

    print(args)
    evaluate(args)
