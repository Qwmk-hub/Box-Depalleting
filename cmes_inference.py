from utils import load_json
from utils.log import generate_logger

opts_server = load_json("configs/opts_server.json")
ipc_type = opts_server["mode"].lower()
logger = generate_logger(opts_server["log_path"], level=opts_server["log_level"])

from cmes_ipc.ipc_grpc import start_grpc_server

start_server = start_grpc_server

start_server(filepath_opts=opts_server["path_opts_inferencer"], logger=logger)
