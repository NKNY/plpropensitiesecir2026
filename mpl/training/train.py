import argparse
import json
import os
import sys
import time

import dotenv
import torch
import tqdm.auto as tqdm

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import mpl.utils.vlpl as vlpl_utils
import mpl.utils.argparsing as argparsing


def train(
    train_dataloader,
    model: torch.nn.Module,
    loss_fn,
    optimizer,
    slot_weights,
    epochs,
    early_stopping=None,
    checkpointing=None,
    device="cpu",
    metrics={},
    validation_dataloader=None,
    verbose=True,
    validation_frequency_epochs=1,
):
    # Restart model (and optimizer) from existing checkpoint
    if checkpointing and checkpointing.restore_from_path:
        checkpointing.load(checkpointing.restore_from_path, device=device)

    iterator_outer = (
        tqdm.tqdm(range(epochs), position=tqdm.tqdm._get_free_pos())
        if verbose
        else range(epochs)
    )

    e = 0
    stop_training = False
    for e in iterator_outer:
        # Save metrics
        if e:
            for subset in metrics.values():
                for metric in subset.values():
                    metric.reset()

        iterator_inner = (
            tqdm.tqdm(enumerate(train_dataloader), position=tqdm.tqdm._get_free_pos())
            if verbose
            else enumerate(train_dataloader)
        )
        model.train()

        for batch, (X, mask, y) in iterator_inner:
            X, mask, y = (
                X.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )

            # assume model outputs either {"logits": tensor} or {"alphas": tensor}
            pred = model(X, padding_mask=mask, position_weights=slot_weights)

            loss = loss_fn(pred, y, mask=mask, pO_slot=slot_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                iterator_inner.set_postfix(loss=loss.item())

            if "train" in metrics:
                for train_metric in metrics["train"].values():
                    if (1 + batch) % train_metric.log_interval == 0:
                        # Position_weights passed via MetricAccumulator.__init__ and can differ from ones used in loss.
                        train_metric.update(pred, y_true=y, mask=mask)

        # Can run validation every n epochs
        run_validation = (e + 1) % validation_frequency_epochs == 0
        if validation_dataloader is not None and run_validation:
            model.eval()
            iterator_inner_validation = (
                tqdm.tqdm(enumerate(validation_dataloader))
                if verbose
                else enumerate(validation_dataloader)
            )
            for batch, (X, mask, y) in iterator_inner_validation:
                X, mask, y = (
                    X.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    y.to(device, non_blocking=True),
                )

                pred = model(X, padding_mask=mask, slot_weights=slot_weights, y_true=y)

                if "validation" in metrics:
                    for validation_metric in metrics["validation"].values():
                        if (1 + batch) % validation_metric.log_interval == 0:
                            validation_metric.update(pred, y_true=y, mask=mask)

        elif not run_validation and validation_dataloader is not None:
            print(f"Skipping validation for epoch {e}.")

        # Early stopping
        if early_stopping:
            metric = metrics
            # Skip early stopping when tgt metric is a validation one, yet we didn't run validation this epoch
            skip_early_stopping_check = (
                early_stopping.metric_key.startswith("validation.")
                and not run_validation
            )
            if not skip_early_stopping_check:
                for key in early_stopping.metric_key.split("."):
                    metric = metric[key]
                if early_stopping(metric.value()):
                    break
        if checkpointing:
            if (
                early_stopping
                and not skip_early_stopping_check
                and early_stopping.counter == 0
            ) or not early_stopping:
                checkpointing.save(e)
        if stop_training:
            break

    if (
        checkpointing
        and checkpointing.restore_best
        and os.path.isfile(checkpointing.path)
    ):
        checkpointing.load(device=device)

    metrics["global_step"] = e + 1

    return metrics


def evaluate(
    dataloader,
    model,
    position_weights,
    metrics,
    device,
    verbose=True,
    eval_key="test",
):
    model.eval()

    for metric in metrics.get(eval_key, {}).values():
        metric.reset()

    t = time.time()
    iterator_inner = (
        tqdm.tqdm(enumerate(dataloader)) if verbose else enumerate(dataloader)
    )
    for batch, (X, mask, y) in iterator_inner:
        X, mask, y = (
            X.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )

        pred = model(X, padding_mask=mask, position_weights=position_weights, y=y)

        if eval_key in metrics:
            for evaluation_metric in metrics[eval_key].values():
                if (1 + batch) % evaluation_metric.log_interval_eval == 0:
                    evaluation_metric.update(pred, y_true=y, mask=mask)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    params_group = parser.add_mutually_exclusive_group(required=True)
    params_group.add_argument(
        "--params", type=str, help="Deeply nested dictionary as a JSON string"
    )
    params_group.add_argument(
        "--params_path", type=str, help="Path to a JSON file containing parameters"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (optional)."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Overwrite FOLD variable in config (optional).",
    )
    args = parser.parse_args()

    params = (
        json.loads(args.params) if args.params else json.load(open(args.params_path))
    )
    if args.fold is not None:
        params["FOLD"] = args.fold

    # Initialise all light objects
    params = argparsing.parse_nested_dict(params)
    if args.seed is not None:
        vlpl_utils.set_random_seed(args.seed)
    elif params["seed"] is not None:
        vlpl_utils.set_random_seed(params["seed"])
    # print(f"Torch random seed: {torch.torch.random.initial_seed()}")

    # Initialise heavier objects or those that require access to other initialised instances.
    params["data"] = params["data"]()
    params["optimizer"] = params["optimizer"](params["model"].parameters())
    params["checkpointing"] = (
        params["checkpointing"](model=params["model"], optimizer=params["optimizer"])
        if "checkpointing" in params
        else None
    )

    # Add GPU support
    params["model"] = params["model"].to(device=params["device"])
    params["position_weights"] = params["position_weights"].to(device=params["device"])
    print(f"Device: {params['device']}")

    # Optional: compile for quicker execution (requires static dimensions).
    compile_list = params.get("compile", {})
    for k, v in compile_list.items():
        v = v or {"fullgraph": False}
        params[k] = torch.compile(params[k], **v)

    t = time.time()
    print("Starting training")
    metrics = train(
        params["data"]["train"],
        params["model"],
        params["loss"],
        params["optimizer"],
        params["position_weights"],
        metrics=params["metrics"],
        validation_dataloader=params["data"]["validation"],
        checkpointing=params["checkpointing"],
        **params["train"],
    )

    if params.get("rerun_validation", False):
        print(f"Re-running validation at {time.time() - t} seconds.")
        metrics = evaluate(
            params["data"]["validation"],
            params["model"],
            params["position_weights"],
            metrics=params["metrics"],
            eval_key="validation",
            **params["eval"],
        )
    else:
        print(f"Skipping re-running evaluation at {time.time() - t} seconds.")

    if params.get("run_test", True):
        print(f"Running evaluation on test set {time.time() - t} seconds.")
        metrics = evaluate(
            params["data"]["test"],
            params["model"],
            params["position_weights"],
            metrics=params["metrics"],
            eval_key="test",
            **params["eval"],
        )
    else:
        print(f"Skipping evaluation on test set. {time.time() - t} seconds.")

    print(
        {
            subset_name: {
                name: metric.value() for name, metric in subset_metrics.items()
            }
            for subset_name, subset_metrics in metrics.items()
            if isinstance(subset_metrics, dict)
        }
    )

    if params["checkpointing"] and params["checkpointing"].remove_after_training:
        params["checkpointing"].remove()

    print(f"Finished training at {time.time() - t} seconds.")
