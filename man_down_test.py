import os, sys
from pathlib import Path
from man_down import Algorithm

# Load Data:
FILE = Path(__file__).resolve()  # file path
ROOT = FILE.parents[0]  # file root path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
dir_name = 'test'

classes_dict_path = ROOT / 'classes.json'
model_weights = 'yolov5s'
source_path = ROOT / 'data/images/img4.jpg'
# source_path = ROOT / 'data/videos/vid.mp4'
output_path = ROOT / 'saved'

# Run Algorithm
algorithm = Algorithm(model_weights, classes_dict_path, source_path, output_path, dir_name)
algorithm.run()