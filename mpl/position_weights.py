import os

import dotenv
import numpy as np
import torch

dotenv.load_dotenv()
data_dir = os.environ["DATA_ROOT"]


def load_position_weights_torch(
    path, K=None, dtype=torch.float32, device=torch.device("cpu")
):
    if path.endswith(".pt"):
        ret = torch.load(path)
    elif path.endswith(".npy") or path.endswith(".npz"):
        ret = torch.tensor(np.load(path))
    else:
        raise ValueError("Path must end with .pt, .npy or .npz")
    if K is not None:
        ret = ret[:K]
    if isinstance(dtype, str):
        dtype = eval(dtype)
    return ret.to(dtype=dtype, device=device)
