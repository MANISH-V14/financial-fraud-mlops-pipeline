import os
import re

def get_next_version(model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1

    files = os.listdir(model_dir)
    versions = []

    for f in files:
        match = re.search(r"model_v(\d+)\.pt", f)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return 1

    return max(versions) + 1