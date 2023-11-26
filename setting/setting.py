import pathlib

# base_path is only used for local debugging purposes
base_path = pathlib.Path().absolute().parent
debug = False

# parameters to control the behavior of recommendation system
topk = 125
time_delta = 86400
