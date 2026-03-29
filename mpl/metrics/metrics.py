import torch


class MetricAccumulator(object):
    def __init__(
        self,
        metric_fn,
        y_pred_key="pred",
        y_true_key="y",
        log_interval=1.0,
        log_interval_eval=1.0,
        print_on_update=False,
        name=None,
        compile=False,
        *metric_fn_args,
        **metric_fn_kwargs,
    ):
        self.name = name
        self.num = 0.0
        self.denom = 0.0
        self.metric_fn_args = metric_fn_args
        self.metric_fn_kwargs = metric_fn_kwargs
        self.metric_fn = lambda *args, **kwargs: (
            metric_fn(*args, *metric_fn_args, **kwargs, **metric_fn_kwargs)
        )
        self.compile = compile
        if self.compile:
            self.metric_fn = torch.compile(self.metric_fn, fullgraph=True)
            print(f"{self.name} will be compiled")
        self.values = []
        self.y_pred_key = y_pred_key
        self.log_interval = log_interval
        self.log_interval_eval = log_interval_eval
        self.print_on_update = print_on_update
        self.y_true_key = y_true_key

    def update(self, y_pred, y_true, *args, **kwargs):
        # Supports either single value predictions or extracting the relevant prediction from a dict of preds.
        if self.y_pred_key is not None and isinstance(y_pred, dict):
            y_pred = y_pred[self.y_pred_key]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach()
        if self.y_true_key is not None and isinstance(y_true, dict):
            y_true = y_true[self.y_true_key]
        *_, num_docs_max, k = y_true.shape
        y_pred, y_true = (
            y_pred.reshape(-1, num_docs_max, k),
            y_true.view(-1, num_docs_max, k),
        )
        batch_metrics = self.metric_fn(y_pred, y_true, *args, **kwargs)
        # Supports self.metric_fn either returning a loss tensor (of any shape) or a dict with num & denom.
        if (
            isinstance(batch_metrics, dict)
            and "num" in batch_metrics
            and "denom" in batch_metrics
        ):
            d_num, d_denom = batch_metrics["num"], batch_metrics["denom"]
        else:
            d_num, d_denom = (
                batch_metrics.sum() / batch_metrics.shape[-1],
                len(batch_metrics),
            )
        self.num += d_num
        self.denom += d_denom
        if self.print_on_update:
            print(f"Batch {self.name}: {d_num / d_denom}")
        return d_num / d_denom

    def save(self, return_update=False):
        self.values.append(self.value())
        return self.value()

    def reset(self):
        self.num = 0.0
        self.denom = 0.0

    def save_and_reset(self):
        self.values.append(self.value())
        self.num = 0.0
        self.denom = 0.0

    def value(self):
        return self.num / self.denom if self.denom else 0.0
