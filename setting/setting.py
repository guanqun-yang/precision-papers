import pathlib

# base_path is only used for local debugging purposes
base_path = pathlib.Path().absolute().parent
debug = False

# parameters to control the behavior of the recommendation system
topk = 250
time_delta = 86400
