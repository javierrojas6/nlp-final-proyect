import os
import numpy as np
import json

def load_dataset(path):
    files = np.array(os.listdir(path))

    payload = []
    for file in files:
        tmp_obj = json.load(open(path + file))
        payload += tmp_obj

    return np.array(payload)