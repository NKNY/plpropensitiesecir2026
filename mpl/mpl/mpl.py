"""Marginalized Plackett Luce core functions"""

import numpy as np
import torch


def log1mexp(x, mask_for_gradient=True):
    """https://github.com/pytorch/pytorch/issues/39242
    :param mask_for_gradient: bool, if True replace masked values with inf. Does not change the computation but prevents
    NaNs appearing in the gradient.
    """
    log2 = 0.6931471805599453  # log(2)
    mx = -x
    mask = x <= log2
    emx = torch.exp(mx)
    ret = torch.where(
        mask,
        torch.log(-torch.expm1(mx)),
        torch.log1p(
            -torch.where(mask, torch.inf, emx) if mask_for_gradient else -emx
        ),  # Detach the branch we'll never actually visit but that can still make grads NaN.
    )
    return ret


def precompute_logsumexp_diagonal_idx_mask(x_shape, max_offset, device):
    """Calculate the positions and binary masks for efficient antidiagonal logsumexp calculation."""
    *D, A, B = x_shape
    max_offset = min(max_offset, A + B - 1)
    max_diag_len = A

    i_idx = torch.arange(max_diag_len, device=device)
    j_idx = (
        torch.arange(1, max_offset, device=device)
        .view(max_offset - 1, 1)
        .expand((max_offset - 1, max_diag_len))
        - i_idx
    )
    mask = ((0 <= i_idx) & (i_idx < max_diag_len)) & (
        (0 <= j_idx) & (j_idx < max_diag_len)
    )
    j_idx = torch.clamp(j_idx, 0, max_diag_len - 1)
    return j_idx, mask


def logsumexp_diagonal(x, max_offset, idx=None, mask=None):
    """Compute the logsumexp across antidiagonals of x.

    :param x: torch.Tensor of shape (..., K, K)
    :param max_offset: The number of antidiagonals for which to compute logsumexp.
    :param idx: optional tensor with indices that belong to each antidiagonal (useful for debugging).
    :param mask: optional boolean tensor with positions that should be masked for each antidiagonal (useful for debugging).
    :return: torch.Tensor of shape (..., max_offset) with antidiagonal logsumexps.
    """
    if idx is None or mask is None:
        idx, mask = precompute_logsumexp_diagonal_idx_mask(
            x.shape, max_offset, x.device
        )
    *D, A, B = x.shape
    values = x.gather(
        -1, idx.T.view(*[1] * len(D), *idx.shape[::-1]).expand(*D, *idx.shape[::-1])
    ).transpose(-1, -2)
    masked_values = torch.where(mask, values, -torch.inf)
    ret = masked_values.logsumexp(-1)
    return ret


def MPL(
    logits,
    K,
    padding_mask=None,
    N=100,
    l=1e-6,
    xi=None,
    wi=None,
    mask_for_gradient=False,
):
    """Main function. Compute Plackett-Luce placement probabilities for K positions.

    :param logits: torch tensor of shape (batch_size, n_docs) containing unnormalized document scores
    :param K: int, the size of the ranking
    :param padding_mask: torch tensor of shape (batch_size, n_docs) with False denoting masked entries (padding)
    :param N: int, number of integration points
    :param l: float, the standard Gumbel quantile to control integration limit
    :param xi, wi: torch.tensors of shape (N, ), integration points and weights from Gauss-Legendre quadrature
    :param mask_for_gradient: bool, setting to True allows to compute gradient for each position at the cost of a slower
    forward pass.
    :return: torch tensor of shape (batch_size, n_docs, K) containing the Plackett-Luce placement probabilities
    """
    bs, n_docs = logits.shape
    device = logits.device
    dtype = logits.dtype
    l = torch.tensor(l, dtype=dtype, device=device)

    if padding_mask is None:
        padding_mask = torch.ones_like(logits, dtype=bool)

    # Integration points and weights, pass them directly to the function to avoid computing every batch
    # (and allow compilation)
    if xi is None or wi is None:
        xi, wi = [
            torch.tensor(x, dtype=logits.dtype, device=device)
            for x in np.polynomial.legendre.leggauss(N)
        ]

    # Integration limits
    b = (logits.amax(dim=-1, keepdim=True) - (-(1 - l).log()).log()).detach()
    a = (
        torch.where(padding_mask, logits, torch.inf).amin(dim=-1, keepdim=True)
        - (-l.log()).log()
    )

    # Inputs to pdf/cdf
    x = (b - a) / 2 * xi + (b + a) / 2  # (bs, n)

    lmx = logits.unsqueeze(-2) - x.unsqueeze(-1)
    melmx = -torch.exp(lmx)
    lp = melmx
    log_p, M = lp.view(bs * N, n_docs), K - 1

    # Calculate the poisson binomial convolution for every x (eq 8)
    lpoisson_binomial = conv_stable(
        log_p, M, padding_mask=padding_mask, mask_for_gradient=mask_for_gradient
    )
    # Integrand
    lf_d = (
        wi.log().view(1, N, 1, 1)
        + lmx.view(bs, N, n_docs, 1)
        + melmx.view(bs, N, n_docs, 1)
        + lpoisson_binomial
    )
    # Integrate and correct for change of variables
    lse = lf_d.logsumexp(1) + ((b - a) / 2).log().view(bs, 1, 1)
    # Rank 1 probabilities are just softmax
    P = torch.concat([logits.softmax(-1).unsqueeze(-1), lse.exp()], dim=-1)
    P_mask = (
        torch.nn.functional.pad(padding_mask, (0, K - n_docs), value=False)
        if K > n_docs
        else padding_mask
    )
    # Mask all documents and positions where we're dealing with padding documents
    P = torch.where(P_mask[:, None, :K] & padding_mask.unsqueeze(-1), P, 0)

    return P


def conv_stable(log_p, M, padding_mask=None, mask_for_gradient=True):
    """
    Computes the Poisson-binomial probability of the number of successes Pr(S=k|d excluded) based on individual success
    probabilities, simultaneously for all documents and positions via Dynamic Programming.
    Based on https://www.sciencedirect.com/science/article/pii/S0167947318300082

    :param log_p: torch.Tensor of shape (batch_size * num_integration_points, n_docs). Each row contains
    success log_probability for all the documents of a (query, integration_point) pair.
    :param M: int, maximum number of successes.
    :param padding_mask: torch.Tensor of shape (batch_size, n_docs) with False denoting masked entries (padding).
    :param mask_for_gradient: bool, setting to True allows to compute gradient for each position at the cost of a slower
    forward pass.
    :return: torch.Tensor of shape (batch_size, n_docs,K) containing the log-probability of number of successes

    Note: keeps the computation in log-space for numerical stability.
    """
    # Successes and failures flipped (p = 1-F_j(x))
    if padding_mask is None:
        padding_mask = torch.ones_like(log_p, dtype=bool)

    dtype, device = log_p.dtype, log_p.device
    bs_n, _ = log_p.shape
    bs, n_docs = padding_mask.shape
    n = bs_n // bs

    # Simultaneously compute c^{1:|D|} and c^{|D|:1} along the last dimension
    lSk = torch.full(
        (bs_n, n_docs, M + 1, 2), fill_value=-torch.inf, device=device, dtype=dtype
    )
    log_p = torch.stack([log_p, log_p.flip(-1)], -1)
    lS0 = torch.cumsum(log_p, -2)
    log_1_minus_p = log1mexp(-log_p, mask_for_gradient=mask_for_gradient)
    lp, lnot_p = log_1_minus_p, log_p
    z = torch.zeros((1,), device=device, dtype=torch.int64)
    minf = torch.full((1,), fill_value=-torch.inf, device=device, dtype=dtype)

    d0 = torch.concatenate(
        [lnot_p[:, :1], lp[:, :1], minf.view(1, 1, 1).expand(bs_n, M - 1, 2)], dim=-2
    )
    lSk.scatter_(
        -2,
        z.view(1, 1, 1, 1).expand(bs_n, n_docs, 1, 2),
        lnot_p.view(bs_n, n_docs, 1, 2),
    )
    lSk.scatter_(
        -3, z.view(1, 1, 1, 1).expand(bs_n, 1, M + 1, 2), d0.view(bs_n, 1, M + 1, 2)
    )

    arange_ndocs = torch.arange(n_docs, device=device, dtype=torch.int64)

    # Apply the DC algorithm
    for i in range(1, n_docs):
        upd1 = lSk[:, i - 1, :-1] + lp[:, i : i + 1]
        upd2 = lSk[:, i - 1, 1:] + lnot_p[:, i : i + 1]
        if mask_for_gradient:
            upd1, upd2 = (
                torch.where(upd1.isinf(), -torch.inf, upd1),
                torch.where(upd2.isinf(), -torch.inf, upd2),
            )  # Needed for grad
        upd3 = torch.logaddexp(upd1, upd2)
        upd = torch.concat([lS0[:, i : i + 1], upd3], dim=-2)
        lSk.scatter_(
            -3,
            arange_ndocs[i].view(1, 1, 1, 1).expand(bs_n, 1, M + 1, 2),
            upd.view(bs_n, 1, M + 1, 2),
        )
    padding = (
        torch.concat([z, minf.view(1).expand(M)], dim=-1)
        .view(1, 1, M + 1, 1)
        .expand(bs_n, 1, M + 1, 2)
    )
    lSk_padded = torch.concat([padding, lSk[:, :-1]], dim=-3)
    abcd, edcb = lSk_padded[..., 0], lSk_padded[..., 1].flip(-2)
    pairwise = abcd.unsqueeze(-1) + edcb.unsqueeze(-2)
    if mask_for_gradient:
        lse_inf_only = (abcd.unsqueeze(-1).isinf()) | (edcb.unsqueeze(-2).isinf())
        pairwise = torch.where(lse_inf_only, -torch.inf, pairwise)
    conv = logsumexp_diagonal(pairwise, M + 1)
    lpoisson_binomial = conv.view(bs, n, n_docs, M)
    if mask_for_gradient:
        lpoisson_binomial = torch.where(
            padding_mask.view(bs, 1, n_docs, 1)
            & padding_mask[:, 1 : M + 1].view(bs, 1, 1, M),
            lpoisson_binomial,
            0,
        )
    return lpoisson_binomial
