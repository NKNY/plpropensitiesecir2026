import os
import sys

import torch

# Set project root
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)

import mpl.gradient.VLPL_grad as vlpl_grad
import mpl.utils.vlpl as vlr_utils


class VLPLLoss(torch.nn.Module):
    def __init__(
        self,
        N,
        K,
        N_loss_estimate=0,
        loss_fn_name="VLRCTRLossFunctionVLR1",
        use_baseline=False,
        y_pred_key=None,
        y_true_key=None,
        reduction: str = "mean",
        dtype=None,
        compile_backward=False,
        compile_backward_params={},
        *args,
        **kwargs,
    ):
        super().__init__()
        self.reduction = reduction
        self.N = N
        self.K = K
        self.loss_fn = {
            "VLPL1LossFunction": {
                (
                    ("fullgraph", True),
                    ("mode", "reduce-overhead"),
                ): VLPL1LossFunction_compiled_fullgraph_reduce_overhead,
                (
                    ("fullgraph", 1),
                    ("mode", "reduce-overhead"),
                ): VLPL1LossFunction_compiled_fullgraph_reduce_overhead,
                (("fullgraph", True),): VLPL1LossFunction_compiled_fullgraph,
                (("fullgraph", 0),): VLPL1LossFunction_compiled_fullgraph,
            }.get(tuple(sorted(compile_backward_params.items())), VLPL1LossFunction)
        }[loss_fn_name].apply
        print(
            f"Chose {loss_fn_name} as loss fn with compile: {compile_backward}, {compile_backward_params}"
        )

        self.N_loss_estimate = N_loss_estimate
        self.use_baseline = use_baseline
        self.y_pred_key = y_pred_key
        self.y_true_key = y_true_key
        self.args = args
        self.kwargs = kwargs
        self.loss_dtype = dtype if dtype is not None else torch.float32
        self.compile_backward = compile_backward
        self.compile_backward_params = compile_backward_params

    def forward(
        self,
        y_pred: torch.Tensor | dict,
        y_true: torch.Tensor,
        pO_slot: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        y_pred = (
            y_pred
            if (self.y_pred_key is None or not isinstance(y_pred, dict))
            else y_pred[self.y_pred_key]
        )
        y_true = (
            y_true
            if (self.y_true_key is None or not isinstance(y_true, dict))
            else y_true[self.y_true_key]
        )
        *_, num_docs_max, k = y_true.shape
        y_pred, y_true = (
            y_pred.reshape(-1, num_docs_max, k).to(
                dtype=self.loss_dtype, non_blocking=True
            ),
            y_true.view(-1, num_docs_max, k).to(
                dtype=self.loss_dtype, non_blocking=True
            ),
        )
        return self.loss_fn(
            y_pred,
            y_true,
            pO_slot,
            self.N,
            self.K,
            None,
            self.N_loss_estimate,
            self.reduction,
            mask,
            self.use_baseline,
        )


class VLRCTRLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y_pred,
        y_true,
        pO_slot,
        N,
        K,
        Pij=None,
        N_loss_estimate=0,
        reduction="mean",
        mask=None,
        use_baseline=False,
    ):
        ctx.N = N
        ctx.K = K
        ctx.reduction = reduction
        ctx.N_loss_estimate = N_loss_estimate
        ctx.save_for_backward(y_pred, y_true, pO_slot, Pij)
        ctx.use_baseline = use_baseline
        if N_loss_estimate > 0:
            ret = vlr_utils.sample_expected_reward_torch(
                y_pred, y_true, pO_slot, N_loss_estimate, K
            )
            ret.requires_grad_()
            return ret.mean()
        else:
            return torch.tensor(
                1.0, requires_grad=True, dtype=y_pred.dtype
            )

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true, pO_slot, Pij = ctx.saved_tensors
        batch_size, n_docs_max, k = y_pred.shape
        grad = vlpl_grad.VLPL1_grad(
            y_pred,
            y_true,
            pO_slot,
            batch_size,
            ctx.K,
            k,
            ctx.N,
            Pij,
            reduction=ctx.reduction,
            use_baseline=ctx.use_baseline,
        )
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class VLPL1LossFunction(VLRCTRLossFunction):
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true, pO_slot, Pij = ctx.saved_tensors
        batch_size, n_docs_max, k = y_pred.shape
        f = vlpl_grad.VLPL1LossFunction
        grad = f(
            y_pred,
            y_true,
            pO_slot,
            batch_size,
            ctx.K,
            k,
            ctx.N,
            Pij,
            reduction=ctx.reduction,
            use_baseline=ctx.use_baseline,
        )
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class VLPL1LossFunction_compiled_fullgraph_reduce_overhead(VLRCTRLossFunction):
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true, pO_slot, Pij = ctx.saved_tensors
        batch_size, n_docs_max, k = y_pred.shape
        f = vlpl_grad.VLPL1LossFunction_compiled_fullgraph_reduce_overhead
        grad = f(
            y_pred,
            y_true,
            pO_slot,
            batch_size,
            ctx.K,
            k,
            ctx.N,
            Pij,
            reduction=ctx.reduction,
            use_baseline=ctx.use_baseline,
        )
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class VLPL1LossFunction_compiled_fullgraph(VLRCTRLossFunction):
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true, pO_slot, Pij = ctx.saved_tensors
        batch_size, n_docs_max, k = y_pred.shape
        f = vlpl_grad.VLPL1LossFunction_compiled_fullgraph
        grad = f(
            y_pred,
            y_true,
            pO_slot,
            batch_size,
            ctx.K,
            k,
            ctx.N,
            Pij,
            reduction=ctx.reduction,
            use_baseline=ctx.use_baseline,
        )
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
