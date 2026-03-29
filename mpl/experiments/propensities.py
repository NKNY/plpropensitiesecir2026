import argparse
import json
import os
import sys
import time

import dotenv
import numpy as np
import torch
import tqdm

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import mpl.mpl.mpl as mpl
import mpl.utils.sampling as vlr_sampling
import mpl.utils.argparsing as argparsing
import mpl.experiments.utils as OPE_utils


def policy_to_propensities(
    model,
    dataloader,
    P_fn,
    device,
    seed=0,
    dtype=torch.float32,
    warmup_batches=2,
    **params,
):
    torch.manual_seed(seed)
    P_hat = []
    t = time.time()
    for i, batch in tqdm.tqdm(enumerate(iter(dataloader))):
        if i == warmup_batches:
            print(f"Recording time from epoch {i}")
            t = time.time()
        X, mask, y = [x.to(device=device) for x in batch]
        bs, n_docs, *_ = y.shape
        pred = model(X, mask)["logits"].squeeze(-1).to(dtype=dtype)
        sampled_P = P_fn(pred, padding_mask=mask, **params).detach()

        P_hat.append(sampled_P)
    P_hat = (
        torch.concat(P_hat, 0).view(-1, 1, n_docs, sampled_P.shape[-1]).cpu().numpy()
    )
    duration = time.time() - t
    return P_hat, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--fold", type=int, help="Fold number")
    parser.add_argument("--subset", type=str, help="validation or test")
    parser.add_argument("--K", type=int, help="ranking size")
    parser.add_argument("--method", type=str, help="MPL or MC")
    parser.add_argument("--l", type=float, help="Integration limit parameter")
    parser.add_argument(
        "--N",
        type=int,
        help="Number of intergration points for MPL or samples (M) for MC",
    )
    parser.add_argument("--bs", type=int, help="batch size")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    with open(args.params_path) as f:
        params = json.load(f)
    if args.l is not None:
        params["l"] = args.l
    if args.fold is not None:
        params["FOLD"] = fold = args.fold
    else:
        params["FOLD"] = fold = 1
    if args.subset is not None:
        params["subset"] = args.subset
    if args.N is not None:
        params["N"] = args.N
    if args.K is not None:
        params["K"] = args.K
    if args.bs is not None:
        params["batch_size"] = args.bs
    if args.method is not None:
        params["method"] = args.method
    if args.seed is not None:
        params["seed"] = args.seed
    params = argparsing.parse_nested_dict(params)

    if "seed" in params:
        torch.manual_seed(params["seed"])

    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float64": torch.float64}[params["dtype"]]

    # Initialise all light objects
    print(f"Torch random seed: {torch.torch.random.initial_seed()}")

    model, data = params["model"], params["data"]
    model.load_state_dict(
        torch.load(params["saved_model_path"], map_location=torch.device(device))[
            "model_state_dict"
        ]
    )
    model = model.to(device=device)
    dataloader = data()[params["subset"]]
    policy_name = os.path.split(params["saved_model_path"])[-1]

    output_params = {
        "N": params["N"],
        "K": params["K"],
        "seed": params["seed"],
        "subset": params["subset"],
    }

    if params["method"] == "MPL":
        output_params = {**output_params, "l": params["l"]}

    output_dir = f"{params['output_root']}/{policy_name}/{params['method']}/{OPE_utils.params_to_path(output_params)}"
    print(f"Output directory: {output_dir}")
    output_params = {k: v for k, v in output_params.items() if k not in ["subset"]}

    # MPL
    if params["method"] == "MPL":
        sampling_fn = lambda *a, **kwargs: mpl.MPL(*a, **kwargs, K=params["K"])
        if params["compile"]:
            sampling_fn = torch.compile(sampling_fn, fullgraph=True)
        l = torch.tensor(params["l"], dtype=dtype, device=device)
        xi, wi = [
            torch.tensor(x, dtype=dtype, device=device)
            for x in np.polynomial.legendre.leggauss(params["N"])
        ]
        output_params = {**output_params, "l": l, "xi": xi, "wi": wi}
    # MC
    else:
        sampling_fn = vlr_sampling.sample_count_P
        output_params = {
            **output_params,
            "compile": params["compile"],
            "chunksize": params["chunksize"],
        }

    propensities, duration = policy_to_propensities(
        model,
        dataloader,
        P_fn=sampling_fn,
        device=device,
        dtype=dtype,
        warmup_batches=params["warmup_batches"],
        **output_params,
    )
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/propensities.npy", propensities)
    np.save(f"{output_dir}/time.npy", duration)
