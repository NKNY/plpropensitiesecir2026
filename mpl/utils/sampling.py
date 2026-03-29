import os
import sys

import torch
import tqdm

# Set project root
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)


def sample_P(m, K, N, noise_tensor_placeholder=None):
    # Use noise_tensor_placeholder to pre-allocate memory if all samples do not fit into memory and thus using batching.
    batch_size, max_n_docs = m.shape
    if max_n_docs < K:
        K = max_n_docs
    device, dtype = m.device, m.dtype
    # Bound uniform to avoid sampling 0, which leads to -inf gumbel noise. Values taken from torch.distributions.Gumbel.
    uniform_low, uniform_high = torch.finfo(m.dtype).tiny, 1 - torch.finfo(m.dtype).eps
    uniform_range = uniform_high - uniform_low
    if noise_tensor_placeholder is None:
        _noise = (
            torch.rand((batch_size, N, max_n_docs), device=device, dtype=dtype)
            * uniform_range
            + uniform_low
        )
    else:
        noise_tensor_placeholder.uniform_()
        _noise = noise_tensor_placeholder * uniform_range + uniform_low
    noise = -torch.log(-torch.log(_noise))
    scores = m.unsqueeze(1) + noise
    order = torch.topk(scores, K, -1).indices
    return order


def sample_count_P(
    m,
    K,
    N,
    padding_mask=None,
    chunksize=None,
    int_dtype=torch.int32,
    verbose=False,
    compile=True,
):
    batch_size, max_n_docs = m.shape
    device = m.device
    padding_mask = (
        padding_mask
        if padding_mask is not None
        else torch.ones((batch_size, max_n_docs), dtype=torch.bool, device=device)
    )
    chunksize = min(N if chunksize is None else chunksize, N)
    counts = 0
    # Pre-allocate memory for the sampling noise
    noise_tensor = torch.empty(
        (batch_size, chunksize, max_n_docs), device=device, dtype=m.dtype
    )
    counts_tensor = torch.zeros(
        (batch_size, max_n_docs, K), dtype=int_dtype, device=m.device
    )
    iterator = tqdm.trange(0, N, chunksize) if verbose else range(0, N, chunksize)
    sample_fn, count_fn = (
        (
            torch.compile(sample_P, fullgraph=True),
            torch.compile(get_counts, fullgraph=True),
        )
        if compile
        else (sample_P, get_counts)
    )
    N_considered = 0
    for start in iterator:
        if chunksize > N - start:
            break
        samples = sample_fn(m, K, chunksize, noise_tensor_placeholder=noise_tensor)
        new_counts = count_fn(samples, max_n_docs, counts_tensor, int_dtype=int_dtype)
        counts = counts + new_counts
        N_considered = N_considered + chunksize
    counts = torch.where(padding_mask.unsqueeze(-1), counts, 0)
    return counts / N_considered


def get_counts(x, n_docs_max, counts_tensor=None, int_dtype=torch.int32):
    # x.shape (..., N, k), e.g. (bs, N, k)
    # ret.shape (..., n_docs_max, k), i.e. we sum counts over N
    *other_shape, N, k = x.shape
    if counts_tensor is None:
        ret = torch.zeros(
            (*other_shape, n_docs_max + 1, k), dtype=int_dtype, device=x.device
        )
    else:
        ret = counts_tensor.zero_()
    o = torch.ones([1] * x.ndim, dtype=int_dtype, device=x.device).expand(*x.shape)
    ret.scatter_add_(1, x, o)
    return ret


is_iterable = lambda x: isinstance(x, (list, tuple))
