import pathlib

# base_path is only used for local debugging purposes
base_path = pathlib.Path().absolute().parent
debug = True

# parameters to control the behavior of the recommendation system
topk = 125
time_delta = 86400
