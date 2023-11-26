import pathlib

base_path = pathlib.Path(".").absolute().parent
debug = True

# parameters to control the behavior of recommendation system
topk = 125
time_delta = 86400