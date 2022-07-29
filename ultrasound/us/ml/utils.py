import os
from glob import glob
from pathlib import Path
import numpy as np


def find_last_checkpoint(logs_folder: Path = Path("us", "ml", "logs", "lightning_logs")):
    last_folder = max(logs_folder.iterdir(), key=lambda f: f.stat().st_mtime)
    return sorted(Path(last_folder, "checkpoints").glob("*.ckpt"))[0]
    #list_of_files = glob(Path(logs_folder, "lightning_logs")) # * means all if need specific format then *.csv
    #latest_file = max(list_of_files, key=os.path.getctime)
    #print(latest_file)

def moving_average(x, w):
    x = np.array(x)
    return np.convolve(x, np.ones(w), 'same') / w
