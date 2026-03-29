import random
from typing import Literal

import numpy as np
import torch


def get_start_end_position_weights(
    *args, version: Literal["torch", "numpy"] = "torch", **kwargs
):
    if version == "torch":
        return get_start_end_position_weights_torch(*args, **kwargs)
    elif version == "numpy":
        return get_start_end_position_weights_np(*args, **kwargs)
    else:
        raise ValueError(
            f"Please specify version as either torch or numpy (default torch)."
        )


def get_start_end_position_weights_np(pO_slot):
    # Calculate position_weights for each (start, end) point (not start, length). Zero-indexed.
    p = np.cumprod((1 - pO_slot)[::-1])[::-1]
    Pij = np.triu(
        1 - p[:, None] / np.concatenate([np.where(p[1:] != 0, p[1:], 1), [1.0]])[None,]
    )
    return Pij


def get_start_end_position_weights_torch(pO_slot):
    ones = torch.ones(1, device=pO_slot.device)
    # Calculate position_weight for each (start, end) point (not start, length). Zero-indexed.
    p = torch.cumprod((ones - pO_slot).flipud(), dim=-1).flipud()
    Pij = torch.triu(
        ones
        - p[:, None]
        / torch.concatenate([torch.where(p[1:] != 0, p[1:], ones), ones])[None,]
    )
    return Pij


def set_random_seed(x=0):
    # Python built-in RNG
    random.seed(x)

    # NumPy RNG
    np.random.seed(x)

    # PyTorch RNGs
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    print(f"Set random seed to {x}")


def VLPL2_sample(m, K, N, padding_value=-torch.inf):
    batch_size, max_n_docs, k = m.shape
    device, dtype = m.device, m.dtype
    # Bound uniform to avoid sampling 0, which leads to -inf gumbel noise. Values taken from torch.distributions.Gumbel.
    uniform_low, uniform_high = torch.finfo(m.dtype).tiny, 1 - torch.finfo(m.dtype).eps
    uniform_range = uniform_high - uniform_low

    # Number of shifted rankings
    num_subrankings = 2 * k - 1

    # Handle max_n_docs < K
    if max_n_docs < K:
        m = torch.concatenate(
            [
                m,
                padding_value
                * torch.ones(
                    (batch_size, K - max_n_docs, k), device=device, dtype=dtype
                ),
            ],
            dim=1,
        )
        max_n_docs = K

    m = torch.concatenate(
        [m, padding_value * torch.ones((batch_size, 1, k), device=device, dtype=dtype)],
        dim=1,
    )  # Add a fake item
    arange_K = torch.arange(K, device=device, dtype=int)
    batch_size, max_n_docs, k = m.shape
    o = torch.ones(1, device=device, dtype=int)
    z = torch.zeros(1, device=device, dtype=int)
    false = torch.zeros(1, device=device, dtype=bool)

    m_padded = torch.repeat_interleave(m, N, dim=0)
    remainder = torch.ones(batch_size * N, dtype=int, device=device) * K
    mask = m_padded != padding_value

    # Additional condition to handle K < k
    mm = torch.arange(k, device=device)[None, :] < remainder[:, None]
    mask &= mm.unsqueeze(1)

    # Initialise outputs
    sampled_items = torch.zeros((batch_size * N, K * 2), dtype=int, device=device)
    sampled_lens = torch.zeros((batch_size * N, K * 2), dtype=int, device=device)
    num_items_placed = torch.zeros((batch_size * N), dtype=int, device=device)

    # Check which samples we've not yet fully completed
    # The second mask allows us to handle cases where n_docs < K. But might mask bugs.
    remainder_mask = (remainder > z) & mask.any(dim=1).any(dim=1)

    # Choose items
    docs_scores = torch.logsumexp(torch.where(mask, m_padded, -torch.inf), dim=-1)
    rand_doc_noise = (
        torch.rand((batch_size * N, max_n_docs), device=device, dtype=dtype)
        * uniform_range
        + uniform_low
    )
    docs_noise = -torch.log(-torch.log(rand_doc_noise))
    docs = torch.topk(docs_scores + docs_noise, K, sorted=True)

    # Choose lens
    rand_len_noise = (
        torch.rand((batch_size * N, K, k), device=device, dtype=dtype) * uniform_range
        + uniform_low
    )
    noise_lens = -torch.log(-torch.log(rand_len_noise))
    docslensscores = m_padded[
        torch.arange(batch_size * N, device=device)[:, None], docs.indices
    ]
    noisedlens = docslensscores + torch.where(
        mask[torch.arange(batch_size * N, device=device)[:, None], docs.indices],
        noise_lens,
        -torch.inf,
    )
    lens = torch.argsort(noisedlens, dim=-1)[:, :, -1] + o

    # Find where sampled items would exceed K OR had to sample padding items bc of small n_docs of the ranking.
    cumlens = torch.cumsum(lens, dim=-1)
    starts = torch.concat(
        [torch.ones(batch_size * N, 1, dtype=int, device=device), lens[:, :-1]], dim=-1
    ).cumsum(dim=-1)
    include = (starts <= (remainder[:, None] - 2 * k + 2)) & (~torch.isinf(docs.values))
    remainder -= torch.where(
        remainder_mask,
        cumlens[
            torch.arange(N * batch_size, device=device),
            (torch.argmax(~include * 1, dim=-1) - 1),
        ],
        z,
    )
    update_idx = arange_K + num_items_placed.view(-1, 1)
    sampled_items.scatter_(1, update_idx, torch.where(include, docs.indices, z))
    sampled_lens.scatter_(1, update_idx, torch.where(include, lens, z))
    num_items_placed += include.sum(dim=-1)

    mask[
        torch.arange(N * batch_size, device=device, dtype=int).view(-1, 1),
        torch.where(include, docs.indices, -o),
    ] = false
    remainder = torch.tile(
        remainder.view(batch_size * N, 1), (1, num_subrankings)
    ) + torch.arange(-k + 1, k, device=device, dtype=int)

    # Expand previous tensors as now we'll be extending 2k-1 sequences separately
    num_items_placed = torch.tile(
        num_items_placed.view(batch_size * N, 1), (1, num_subrankings)
    )
    sampled_items = torch.tile(
        sampled_items.view(batch_size, N, 1, K * 2), (1, 1, num_subrankings, 1)
    )
    sampled_lens = torch.tile(
        sampled_lens.view(batch_size, N, 1, K * 2), (1, 1, num_subrankings, 1)
    )
    num_items_placed = num_items_placed.view(batch_size, N, num_subrankings)
    remainder = remainder.view(batch_size, N, num_subrankings)

    # Sample 2k document from each of the k lengths
    dl_noise_uniform = (
        torch.rand((batch_size * N, max_n_docs, k), device=device, dtype=dtype)
        * uniform_range
        + uniform_low
    )
    dl_noise = -torch.log(-torch.log(dl_noise_uniform))
    dl = torch.topk(
        torch.where(mask, m_padded + dl_noise, -torch.inf), 2 * k, dim=-2, sorted=False
    )

    # Order the 2*k*k sampled documents (across all lengths combined)
    sorted_flattened_values, sorted_flattened_idx = torch.sort(
        dl.values.view(batch_size, N, 2 * k * k), dim=-1, descending=True
    )  # (bs, N, K*k)
    lengths = sorted_flattened_idx % k + o
    iids = (
        dl.indices.view(batch_size * N, 2 * k * k)
        .gather(-1, sorted_flattened_idx.view(batch_size * N, 2 * k * k))
        .view(batch_size, N, 2 * k * k)
    )

    # Make sure that padding is not included in the final rankings even if has to be sampled
    # No need to remove options that are too long at this stage as we're sampling same amount from every length.
    sampled_is_padding = torch.isinf(sorted_flattened_values)
    iids, lengths = (
        torch.where(sampled_is_padding, -1, iids),
        torch.where(sampled_is_padding, -1, lengths),
    )
    last_pos = torch.zeros((batch_size, N, num_subrankings), device=device, dtype=int)
    doc_mask = torch.ones((batch_size, N, 1, 2 * k * k), device=device, dtype=bool)

    for i in range(2 * k):
        # Take next action, which is:
        # doc not too long
        pos_not_too_long = lengths.unsqueeze(-2) <= remainder.view(
            batch_size, N, num_subrankings, 1
        )

        # doc after last added doc
        last_pos_mask = last_pos.unsqueeze(-1) <= torch.arange(
            2 * k**2, dtype=int, device=device
        )

        # True = not padding
        padding_mask = ~sampled_is_padding.view(batch_size, N, 1, 2 * k * k)

        mask = pos_not_too_long & last_pos_mask & doc_mask & padding_mask

        any_sampled = mask.any(dim=-1)
        first_valid_dl_idx = torch.argmax(
            mask.to(dtype=int), dim=-1
        )  # (bs, N, num_subrankings)

        sampled_d = torch.where(any_sampled, iids.gather(-1, first_valid_dl_idx), z)
        sampled_l = torch.where(any_sampled, lengths.gather(-1, first_valid_dl_idx), z)

        # Update masks for next step
        doc_mask = torch.where(
            sampled_d.view(*sampled_d.shape, 1)
            == iids.view(batch_size, N, 1, 2 * k * k),
            false,
            doc_mask,
        )
        remainder -= sampled_l
        last_pos = torch.where(any_sampled, first_valid_dl_idx, last_pos)
        update_position = torch.where(any_sampled, num_items_placed, 2 * K - 1).view(
            *num_items_placed.shape, 1
        )
        num_items_placed = torch.where(
            any_sampled, num_items_placed + 1, num_items_placed
        )
        sampled_items.scatter_(-1, update_position, sampled_d.view(*sampled_d.shape, 1))
        sampled_lens.scatter_(-1, update_position, sampled_l.view(*sampled_l.shape, 1))

    ret = sampled_items[:, :, :, : K + k - 1], sampled_lens[:, :, :, : K + k - 1]
    return ret


def sample_expected_reward_torch(pred, y, position_weights, N, K, *args, **kwargs):
    # The difference from get_policy_val_batch is that the latter gets an exhaustive list of all possible rankings
    # and calculates the probability/reward of each one. This can be expensive when n_docs is high.
    # This method instead samples N times from the distribution of logscores pred and gets the reward of the sample.
    start_end_position_weights = get_start_end_position_weights(position_weights)
    si, sl = VLPL2_sample(pred, K, N)
    si, sl = (
        si[:, :, 0, :K].view((-1, N, K)),
        sl[:, :, 0, :K].view((-1, N, K)),
    )
    si, sl = si.reshape((-1, N, K)), sl.reshape((-1, N, K))
    dl_sampled = torch.concatenate([si[:, :, :, None], sl[:, :, :, None]], dim=-1)
    options_sampled = dl_array_to_dls_torch_vectorised(dl_sampled)
    expected_rewards_sampled = get_rankings_expected_rewards_torch_vectorised(
        options_sampled, y, start_end_position_weights
    )
    return expected_rewards_sampled


def sample_expected_reward_torch_dataset_normalized(
    pred, y, position_weights, N, K, mask, *args, **kwargs
):
    # Only implemented for k = 1
    y = torch.where(mask.unsqueeze(-1), y, 0)
    sampled_metric = sample_expected_reward_torch(
        pred, y, position_weights, N, K, *args, **kwargs
    )  # (bs, N)
    skyline_metric = get_best_ranking_score(y, mask, K, position_weights)  # (bs, )
    num = sampled_metric.mean(-1).sum()
    denom = skyline_metric.sum()
    return {"num": num, "denom": denom}


def get_best_ranking_score(y, mask, K, position_weights):
    # Only implemented for k = 1
    bs, n_docs, _ = y.shape
    y = torch.where(mask, y.squeeze(-1), 0.0)  # (bs, n_docs)
    if K > n_docs:
        y = torch.nn.functional.pad(y, (0, K - n_docs), value=0.0)
    topk_values, topk_idx = torch.topk(y, K, dim=-1, sorted=True)  # (bs, K)
    ret = (topk_values * position_weights).sum(dim=-1)  # (bs, )
    return ret


def dl_array_to_dls_torch_vectorised(dl):
    s = torch.cumsum(
        torch.concatenate(
            [
                torch.ones((*dl.shape[:2], 1), dtype=int, device=dl.device),
                dl[:, :, :, 1],
            ],
            dim=-1,
        )[:, :, :-1],
        dim=-1,
    ).unsqueeze(-1) * (dl[:, :, :, 1:2] > 0)
    dls = torch.concatenate([dl, s], dim=-1)
    return dls


def get_rankings_expected_rewards_torch_vectorised(
    options_array, rel, start_end_position_weights
):
    # Takes in dense matrix of rankings (batch_size, num_rankings, K, 3)
    # containing (doc, len, start_idx from 1) (zero-padded)
    # together with rel (num_docs, k) and start_end_position_weights (K, K)
    # And returns the expected value of each of the rankings
    device = options_array.device
    batch_size, N, K, _ = options_array.shape
    lens_0idx = (
        options_array[:, :, :, 1] - 1
    )  # >-1 with this equivalent to len > 0 aka not masked position.
    starts_0idx_flattened = options_array[:, :, :, 2].flatten() - 1
    lens_0idx_flattened = lens_0idx.flatten()
    pos_rel = rel[
        torch.arange(len(options_array), device=device)[:, None, None],
        options_array[:, :, :, 0],
        (lens_0idx > -1) * lens_0idx,
    ]
    pos_w = (
        (lens_0idx_flattened > -1)
        * start_end_position_weights[
            starts_0idx_flattened, starts_0idx_flattened + lens_0idx_flattened
        ]
    ).reshape((batch_size, N, K))
    pos_expected_reward = pos_rel * pos_w
    ranking_expected_reward = torch.sum(pos_expected_reward, dim=-1)
    return ranking_expected_reward
