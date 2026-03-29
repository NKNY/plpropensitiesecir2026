import os

import dotenv
import torch

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]


def get_path_from_project_root(*args, **kwargs):
    return os.path.join(
        PROJECT_ROOT, *[str(x) for x in args], *[str(x) for x in list(kwargs.values())]
    )


def get_path_from_data_root(*args, **kwargs):
    return os.path.join(
        DATA_ROOT, *[str(x) for x in args], *[str(x) for x in list(kwargs.values())]
    )


class EarlyStopping:
    def __init__(
        self,
        patience=1,
        min_delta=0,
        metric_key="loss",
        mode="max",
        verbose=True,
        baseline=None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.epoch = 0
        self.best_epoch = 0
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.metric_key = metric_key
        self.verbose = verbose
        self.baseline = baseline

    def __call__(self, score):
        required_score = (
            self.best_score - self.min_delta
            if self.mode == "min"
            else self.best_score + self.min_delta
        )
        if self.baseline is not None:
            required_score = (
                min(required_score, self.baseline)
                if self.mode == "min"
                else max(required_score, self.baseline)
            )
        if self.verbose:
            print(
                f"Epoch {self.epoch} {self.metric_key}: {score}. Best was epoch {self.best_epoch}: {self.best_score}. "
                f"Needed: {required_score} by epoch {self.best_epoch + self.patience}."
            )
        if ((self.mode == "min") & (score <= required_score)) or (
            (self.mode == "max") & (score >= required_score)
        ):
            self.best_epoch = self.epoch
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                print(
                    f"Epoch: {self.epoch}: insufficient improvement in {self.metric_key} for {self.patience} epochs."
                )
                return True
            if (
                (self.baseline is not None)
                and (self.patience == self.epoch)
                and (
                    ((self.mode == "min") and (self.best_score > self.baseline))
                    or ((self.mode == "max") and (self.best_score < self.baseline))
                )
            ):
                print(
                    f"Epoch: {self.epoch}: insufficient improvement over baseline {self.baseline} in {self.metric_key} in {self.patience} epochs."
                )
                return True
        self.epoch += 1
        return False

    def reset(self):
        self.epoch = 0
        self.counter = 0
        self.best_epoch = 0
        self.best_score = float("inf") if self.mode == "min" else -float("inf")


class CheckpointSaver:
    def __init__(
        self,
        model,
        path,
        optimizer=None,
        restore_best=False,
        restore_from_path=None,
        remove_after_training=False,
    ):
        self.model = model
        self.path = path
        self.optimizer = optimizer
        self.restore_best = restore_best
        self.restore_from_path = restore_from_path
        self.remove_after_training = remove_after_training

    def save(self, epoch):
        output = {"epoch": epoch, "model_state_dict": self.model.state_dict()}
        if self.optimizer is not None:
            output["optimizer_state_dict"] = self.optimizer.state_dict()
        os.makedirs(os.path.split(self.path)[0], exist_ok=True)
        torch.save(output, self.path)
        print(f"Saved checkpoint for epoch {epoch} to {self.path}")

    def load(self, path=None, device="cpu"):
        path = self.path if path is None else path
        checkpoint = torch.load(path, device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint for epoch {checkpoint['epoch']} from {path}).")

    def remove(self, path=None):
        path = self.path if path is None else path
        os.remove(path)
        print(f"Removed checkpoint from {path}.")
