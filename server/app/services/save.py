def json_file(exp_id, sample, json_content):
    with open(f"json/{exp_id}/{sample}.json", "w") as f:
        f.write(json_content)


def image_file(exp_id, sample, image):
    image.save(f"images/{exp_id}/{sample}.jpg")
