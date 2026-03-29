import argparse
import json
import os
import sys

import dotenv
import numpy as np
import torch
import tqdm

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import mpl.utils.sampling as vlr_sampling
import mpl.utils.argparsing as argparsing
import mpl.experiments.utils as OPE_utils


def generate_clicks(
    model,
    dataloader,
    N,
    K,
    pos_bias,
    device,
    dtype,
    seed=0,
    chunksize=None,
    *args,
    **kwargs,
):
    torch.manual_seed(seed)
    if isinstance(pos_bias, str):
        pos_bias = OPE_utils.get_pos_bias(pos_bias, K, device, dtype)
    new_relevances = []
    chunksize = min(N if chunksize is None else chunksize, N)
    for batch in tqdm.tqdm(iter(dataloader)):
        X, mask, y = [x.to(device=device) for x in batch]
        bs, n_docs, *_ = y.shape

        # Get logits
        pred = model(X, mask)["logits"].squeeze(-1)  # (bs, n_docs)

        # Expand to represent multiple experiment repeats
        pred = pred.view(bs, 1, n_docs).expand(bs, 1, n_docs).reshape(1 * bs, n_docs)

        # Combine back
        R_empirical_num = torch.zeros(bs * 1, n_docs, K, device=device, dtype=dtype)

        iterator = range(0, N, chunksize)
        for start in iterator:
            chunksize = min(chunksize, N - start)
            # Sample rankings from model
            rankings = vlr_sampling.sample_P(pred, K, chunksize)  # (bs * 1, N, k)

            # # Sample clicks  (assume C = Observed * Thought was Relevant)
            # # Assume relevance in 0-1
            PR = (
                y.squeeze(-1)
                .gather(-1, rankings.view(bs, -1))
                .view(bs * 1, chunksize, K)
            )
            C = (
                (
                    torch.rand(bs * 1, chunksize, K, device=device, dtype=dtype)
                    <= pos_bias * PR
                )
                .view(bs * 1, chunksize * K)
                .to(dtype=dtype)
            )

            R_empirical_num.scatter_add_(1, rankings, C.view(bs * 1, -1, K))

        R_empirical = (R_empirical_num / N).view(bs, 1, n_docs, K)  # R_empirical_denom
        padding_mask = mask.view(bs, 1, n_docs, 1)
        R_empirical = torch.where(padding_mask, R_empirical, -1)

        new_relevances.append(R_empirical.detach())

    new_relevances = torch.concat(new_relevances).cpu().numpy()
    return new_relevances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--fold", type=int, help="Fold number")
    parser.add_argument("--subset", type=str, help="validation or test")
    parser.add_argument("--K", type=int, help="ranking size")
    parser.add_argument(
        "--N",
        type=int,
        help="Number of intergration points for MPL or samples (M) for MC",
    )

    args = parser.parse_args()
    with open(args.params_path) as f:
        params = json.load(f)
    if args.fold is not None:
        params["FOLD"] = fold = args.fold
    if args.subset is not None:
        params["subset"] = args.subset
    if args.N is not None:
        params["N"] = args.N
    if args.K is not None:
        params["K"] = args.K
    if "seed" in params:
        torch.manual_seed(params["seed"])
    params = argparsing.parse_nested_dict(params)

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
        "pos_bias": params["pos_bias"],
        "seed": params["seed"],
        "subset": params["subset"],
    }
    output_dir = f"{params['output_root']}/{policy_name}/{OPE_utils.params_to_path(output_params)}"
    output_params["pos_bias"] = OPE_utils.get_pos_bias(
        params["pos_bias"], params["K"], device, dtype
    )
    output_params["chunksize"] = params["chunksize"]
    output_params = {k: v for k, v in output_params.items() if k not in ["subset"]}
    clicks = generate_clicks(
        model, dataloader, device=device, dtype=dtype, **output_params
    )

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/clicks.npy", clicks)
