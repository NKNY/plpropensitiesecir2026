import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import mpl.utils.vlpl as vlpl_utils


def VLPL1LossFunction(*args, **kwargs):
    return VLPL1_grad(*args, **kwargs)


# @torch.compile(mode="reduce-overhead", fullgraph=True)
def VLPL1LossFunction_compiled_fullgraph_reduce_overhead(*args, **kwargs):
    return VLPL1_grad(*args, **kwargs)


# @torch.compile(fullgraph=True)
def VLPL1LossFunction_compiled_fullgraph(*args, **kwargs):
    return VLPL1_grad(*args, **kwargs)


def VLPL1_grad(
    m,
    y,
    pO_slot,
    batch_size,
    K,
    k,
    N,
    Pij=None,
    sisl=None,
    use_baseline=False,
    reduction="mean",
):
    device, dtype = m.device, m.dtype
    n_docs_max = y.shape[1]
    extra_padding_size = max(K - n_docs_max, 0)
    z, o, false = (
        torch.zeros(1, device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=bool),
    )

    _n_docs_max = n_docs_max + extra_padding_size
    arange_K = torch.arange(K, device=device, dtype=int)
    arange_k = torch.arange(k, device=device, dtype=int)
    arange_bs = torch.arange(batch_size, device=device, dtype=int)
    arange_N = torch.arange(N, device=device, dtype=int)
    arange_ndocs = torch.arange(_n_docs_max, device=device, dtype=int)

    Pij = Pij if Pij is not None else vlpl_utils.get_start_end_position_weights(pO_slot)

    si, sl = vlpl_utils.VLPL2_sample(m, K, N) if sisl is None else sisl

    _sl = sl[:, :, k - 1, :K]
    _si = si[:, :, k - 1, :K]

    # Placement rewards
    padding_mask = sl != 0
    _m = torch.concat(
        [
            m,
            -torch.inf
            * torch.ones(
                (batch_size, extra_padding_size + 1, k), device=device, dtype=int
            ),
        ],
        dim=1,
    )
    _y = torch.concat(
        [
            y,
            -torch.inf
            * torch.ones(
                (batch_size, extra_padding_size + 1, k), device=device, dtype=int
            ),
        ],
        dim=1,
    )
    mask = _y[:, :_n_docs_max] >= 0

    R = y[
        arange_bs[:, None],
        si.reshape(batch_size, (2 * k - 1) * N * (K + k - 1)),
        sl.reshape(batch_size, (2 * k - 1) * N * (K + k - 1)) - 1,
    ].view(batch_size, N, 2 * k - 1, K + k - 1)
    R = torch.where(padding_mask, R, 0)

    s = torch.concat(
        [
            torch.ones((batch_size, N, 2 * k - 1, 1), dtype=int, device=device),
            sl[:, :, :, :-1],
        ],
        dim=-1,
    ).cumsum(dim=-1)
    s = torch.where(padding_mask, s, 0)
    _s = s[:, :, k - 1, :K]
    theta_a = (s - torch.arange(-k + 1, k, device=device, dtype=int)[:, None]) - 1
    theta_b = sl - 1
    Theta = torch.where(theta_a >= 0, Pij[theta_a, theta_a + theta_b], z)

    RTheta = R * Theta
    PR = RTheta.flip(-1).cumsum(dim=-1).flip(-1)
    _PR = PR[:, :, k - 1, :K]
    _Theta = Theta[:, :, k - 1, :K]

    _si = si[:, :, k - 1, :K]
    _padding_mask = padding_mask[:, :, k - 1, :K]

    cant_use_len_mask = (K - torch.where(_padding_mask, _s, K + 1) + 1).unsqueeze(
        -1
    ) < arange_k + 1
    sorted_padded_ranking = torch.full(
        (batch_size, N, K + _n_docs_max), _n_docs_max, dtype=int, device=device
    )
    sorted_padded_ranking[:, :, :K] = _si
    num_docs_placed = (_padding_mask).sum(-1)
    binary_mask = torch.ones(
        (batch_size, N, _n_docs_max + 1), dtype=bool, device=device
    )
    binary_mask[
        arange_bs[:, None, None],
        arange_N[:, None],
        torch.where(_sl > 0, _si, _n_docs_max),
    ] = false
    update_idx = num_docs_placed.unsqueeze(-1) + arange_ndocs
    sorted_padded_ranking = sorted_padded_ranking.scatter(
        -1,
        update_idx,
        torch.where(
            binary_mask,
            torch.arange(_n_docs_max + 1, device=device, dtype=int),
            _n_docs_max,
        ),
    )

    all_scores = _m[arange_bs[:, None, None], sorted_padded_ranking]
    logdenom_per_rank_per_len = torch.logcumsumexp(all_scores.flip(-2), dim=-2).flip(
        -2
    )[:, :, :K]
    logdenom_per_rank_per_len = torch.where(
        cant_use_len_mask, -torch.inf, logdenom_per_rank_per_len
    )
    logdenom_per_rank = torch.logsumexp(logdenom_per_rank_per_len, dim=-1)

    _Pij = torch.empty(K, k, device=device, dtype=dtype)
    for i in range(k):
        _Pij[: K - i, i] = torch.diagonal(Pij, offset=i)

    RI = torch.where(
        _padding_mask, (_PR / torch.exp(logdenom_per_rank)).cumsum(dim=-1), 0
    )
    DR = torch.where(
        _padding_mask.unsqueeze(-1),
        (
            _Pij[_s.flatten() - 1].view(batch_size, N, K, k)
            / torch.exp(logdenom_per_rank).unsqueeze(-1)
        ).cumsum(dim=-2),
        0,
    )

    len_is_limited = cant_use_len_mask.any(dim=-2)
    last_valid_per_len = (
        torch.where(
            len_is_limited, torch.argmax(cant_use_len_mask.to(dtype=int), dim=-2), K
        )
        - 1
    )
    last_valid_per_doc = torch.where(_padding_mask, arange_K, K - 1)
    last_valid_per_doc_len = torch.where(
        _padding_mask.unsqueeze(-1),
        torch.min(last_valid_per_doc.unsqueeze(-1), last_valid_per_len.unsqueeze(-2)),
        0,
    )

    DR_sampled = DR.gather(-2, last_valid_per_doc_len)
    RI_sampled = RI.gather(-1, last_valid_per_doc_len.view(batch_size, N, K * k)).view(
        batch_size, N, K, k
    )
    DR_update = torch.tile(
        DR.gather(-2, last_valid_per_len.unsqueeze(-2)), (1, 1, _n_docs_max + 1, 1)
    )
    DR_update[
        arange_bs[:, None, None],
        arange_N[:, None],
        torch.where(_padding_mask, _si, _n_docs_max),
    ] = DR_sampled
    DR_update *= _y.unsqueeze(1)

    RI_update = torch.tile(
        RI.gather(-1, last_valid_per_len.view(batch_size, N, k)).view(
            batch_size, N, 1, k
        ),
        (1, 1, _n_docs_max + 1, 1),
    )
    RI_update[
        arange_bs[:, None, None],
        arange_N[:, None],
        torch.where(_padding_mask, _si, _n_docs_max),
    ] = RI_sampled
    update = DR_update - RI_update
    sign = torch.sign(update)
    _update = -torch.log(torch.abs(update))
    ret = torch.where(update == z, z, torch.exp(_m.unsqueeze(1) - _update) * sign)

    PR_sampled = PR[:, :, k - 1, 1:K]
    ret[
        arange_bs[:, None, None],
        arange_N[:, None],
        torch.where(_padding_mask, _si, _n_docs_max)[:, :, : K - 1],
        (_sl - 1)[:, :, : K - 1],
    ] += PR_sampled

    ret = torch.where(mask[:, None, :n_docs_max], ret[:, :, :n_docs_max], 0)
    ret = -ret.mean(1)

    return ret
