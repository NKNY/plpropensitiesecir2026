import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)


def get_pos_bias(pos_bias, K, device, dtype):
    return {
        "dcg": lambda K: torch.log2(
            torch.arange(1, K + 1, device=device, dtype=dtype) + 1
        )
        ** -1,
        "invrank": lambda K: torch.arange(1, K + 1, device=device, dtype=dtype) ** -1,
    }[pos_bias](K)


def params_to_path(params):
    return ";".join(
        [
            f"{k}={v}"
            for k, v in {k2: params[k2] for k2 in sorted(params)}.items()
            if not str(k).endswith("path")
        ]
    )


def maybe_float(s: str) -> bool | str:
    try:
        return float(s)
    except ValueError:
        return s


def str_to_d(x, sep=";"):
    return dict([[maybe_float(z) for z in y.split("=")] for y in x.split(sep)])


def path_to_params(path, root, mappings, sep_outer=";", sep_inner="="):
    p = path.split(root + "/")[1]
    s = [x for x in p.split("/")]
    mapping = mappings[len(s)]
    s = [str_to_d(x, sep_outer) if sep_inner in x else x for x in s]
    return dict(zip(mapping, s))


def flatten_dict(d, parent_key="", sep=";"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
