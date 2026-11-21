"""Longest Exact Causal Prefix (LECP) CoW KV Cache for zero-copy multi-branch CFG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class CoWBranchSpec:
    """Descriptor for one CFG branch's view into shared prefix + private suffix."""

    branch_idx: int
    total_len: int
    shared_prefix_len: int
    suffix_len: int


def _is_same_storage_prefix(a: Any, b: Any, length: int) -> bool:
    """O(1) check whether tensors `a[:length]` and `b[:length]` alias the exact same GPU memory."""
    if length <= 0:
        return True
    if a is None or b is None:
        return False
    if a.shape[0] < length or b.shape[0] < length:
        return False
    return (
        a.data_ptr() == b.data_ptr()
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def detect_lecp_prefix_len(
    cache_a: Any,
    cache_b: Any,
    layer_check_indices: Sequence[int] | None = None,
) -> int:
    """Detect Longest Exact Causal Prefix length `L` where `cache_a[:L] == cache_b[:L]` bitwise.

    Fast path checks O(1) CoW storage pointer identity across all layers before touching GPU memory.
    Fallback verifies exact tensor equality via `torch.equal`.
    """
    ka0 = cache_a.key_cache[0]
    kb0 = cache_b.key_cache[0]
    if ka0 is None or kb0 is None:
        return 0
    len_a = int(ka0.shape[0])
    len_b = int(kb0.shape[0])
    l_min = min(len_a, len_b)
    if l_min == 0:
        return 0

    num_layers = len(cache_a.key_cache)
    # Fast O(1) CoW storage-identity path (works without importing torch or GPU sync!)
    if _is_same_storage_prefix(ka0, kb0, l_min) and _is_same_storage_prefix(
        cache_a.value_cache[0], cache_b.value_cache[0], l_min
    ):
        if all(
            _is_same_storage_prefix(cache_a.key_cache[l], cache_b.key_cache[l], l_min)
            and _is_same_storage_prefix(cache_a.value_cache[l], cache_b.value_cache[l], l_min)
            for l in range(num_layers)
        ):
            return l_min

    cow_hint = max(
        int(getattr(cache_a, "_cow_shared_prefix_len", 0) or 0),
        int(getattr(cache_b, "_cow_shared_prefix_len", 0) or 0),
    )
    if 0 < cow_hint <= l_min and _is_same_storage_prefix(ka0, kb0, cow_hint):
        return cow_hint

    import torch

    check_layers = list(range(num_layers)) if layer_check_indices is None else list(layer_check_indices)

    def _prefix_equal(length: int) -> bool:
        for l in check_layers:
            if not torch.equal(cache_a.key_cache[l][:length], cache_b.key_cache[l][:length]):
                return False
            if not torch.equal(cache_a.value_cache[l][:length], cache_b.value_cache[l][:length]):
                return False
        return True

    if _prefix_equal(l_min):
        return l_min

    diff_k = (ka0[:l_min] != kb0[:l_min]).reshape(l_min, -1).any(dim=1)
    diff_v = (cache_a.value_cache[0][:l_min] != cache_b.value_cache[0][:l_min]).reshape(l_min, -1).any(dim=1)
    nz = torch.nonzero(diff_k | diff_v, as_tuple=False)
    cand = int(nz[0].item()) if nz.numel() > 0 else l_min
    if cand == 0:
        return 0
    if _prefix_equal(cand):
        return cand
    lo, hi = 0, cand
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _prefix_equal(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def cow_fork_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Zero-copy Copy-on-Write fork of a BAGEL context dict (replaces `deepcopy(gen_context)`)."""
    orig_cache = ctx["past_key_values"]
    cache_cls = type(orig_cache)
    num_layers = orig_cache.num_layers
    forked_cache = cache_cls(num_layers)
    for l in range(num_layers):
        forked_cache.key_cache[l] = orig_cache.key_cache[l]
        forked_cache.value_cache[l] = orig_cache.value_cache[l]
    forked_cache.key_values_lens = list(orig_cache.key_values_lens) if orig_cache.key_values_lens is not None else None
    seq_len = int(orig_cache.key_cache[0].shape[0]) if orig_cache.key_cache[0] is not None else 0
    forked_cache._cow_shared_prefix_len = seq_len
    forked_cache._cow_origin_id = id(orig_cache)
    return {
        "kv_lens": list(ctx["kv_lens"]),
        "ropes": list(ctx["ropes"]),
        "past_key_values": forked_cache,
    }


def rebind_cow_prefix(parent_cache: Any, child_cache: Any, prefix_len: int | None = None) -> int:
    """Rebind `child_cache` tensors as zero-copy views into `parent_cache[:L]`, freeing duplicated HBM!

    When `parent_cache` appends new tokens via `torch.cat([old_k, new_k])`, `parent_cache` gets a new
    contiguous buffer containing `old_k` as its exact prefix `[:L]`. Rebinding `child_cache` (e.g., `cfg_text`
    whose length equals `L`) to `parent_cache[:L]` immediately drops the redundant `old_k` allocation and
    restores O(1) storage-pointer identity for `SharedPrefixCache`.
    """
    k_p0 = parent_cache.key_cache[0]
    k_c0 = child_cache.key_cache[0]
    if k_p0 is None or k_c0 is None:
        return 0
    len_p = int(k_p0.shape[0])
    len_c = int(k_c0.shape[0])
    target_len = len_c if prefix_len is None else int(prefix_len)
    if target_len <= 0 or target_len != len_c or target_len > len_p:
        return 0
    for l in range(parent_cache.num_layers):
        child_cache.key_cache[l] = parent_cache.key_cache[l][:target_len]
        child_cache.value_cache[l] = parent_cache.value_cache[l][:target_len]
    child_cache._cow_shared_prefix_len = target_len
    return target_len


class SharedPrefixCache:
    """Zero-copy multi-branch CFG KV container replacing `NaiveCache.merge(cfg_branch_caches)`."""

    _lecp_memo: dict[tuple[Any, ...], tuple[int, tuple[int, ...]]] = {}

    def __init__(
        self,
        num_layers: int,
        num_branches: int,
        shared_prefix_len: int,
        shared_branch_ids: list[int],
        unshared_branch_ids: list[int],
        shared_prefix_k: dict[int, Any],
        shared_prefix_v: dict[int, Any],
        branch_suffix_k: list[dict[int, Any]],
        branch_suffix_v: list[dict[int, Any]],
        branch_specs: list[CoWBranchSpec],
    ) -> None:
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.shared_prefix_len = shared_prefix_len
        self.shared_branch_ids = shared_branch_ids
        self.unshared_branch_ids = unshared_branch_ids
        self.shared_prefix_k = shared_prefix_k
        self.shared_prefix_v = shared_prefix_v
        self.branch_suffix_k = branch_suffix_k
        self.branch_suffix_v = branch_suffix_v
        self.branch_specs = branch_specs

        self.key_values_lens = [s.total_len for s in branch_specs]
        has_any = any(s.total_len > 0 for s in branch_specs)
        self.key_cache = {
            l: (shared_prefix_k[l] if shared_prefix_len > 0 else (branch_suffix_k[0][l] if has_any else None))
            for l in range(num_layers)
        }
        self.value_cache = {
            l: (shared_prefix_v[l] if shared_prefix_len > 0 else (branch_suffix_v[0][l] if has_any else None))
            for l in range(num_layers)
        }

    @property
    def seq_lens(self) -> int:
        return sum(self.key_values_lens)

    @property
    def unique_kv_tokens(self) -> int:
        return self.shared_prefix_len + sum(s.suffix_len for s in self.branch_specs)

    @property
    def naive_kv_tokens(self) -> int:
        return sum(s.total_len for s in self.branch_specs)

    @property
    def padded_naive_kv_tokens(self) -> int:
        max_len = max((s.total_len for s in self.branch_specs), default=0)
        return max_len * self.num_branches

    @property
    def kv_reduction_ratio(self) -> float:
        uniq = max(1, self.unique_kv_tokens)
        return self.naive_kv_tokens / uniq

    @classmethod
    def clear_memo(cls) -> None:
        cls._lecp_memo.clear()

    @classmethod
    def from_branch_caches(
        cls,
        caches: Sequence[Any],
        min_shared_tokens: int = 32,
        verify_all_layers: bool = True,
    ) -> SharedPrefixCache:
        """Build a zero-copy `SharedPrefixCache` from BAGEL's `cfg_branch_caches`."""
        num_branches = len(caches)
        num_layers = caches[0].num_layers

        sig_parts = []
        lens = []
        for c in caches:
            k0 = c.key_cache[0]
            ln = int(k0.shape[0]) if k0 is not None else 0
            ptr = int(k0.data_ptr()) if k0 is not None else 0
            lens.append(ln)
            sig_parts.extend((id(c), ln, ptr))
        memo_key = tuple(sig_parts)

        cached = cls._lecp_memo.get(memo_key)
        if cached is not None:
            l_shared, shared_tuple = cached
            shared_branch_ids = list(shared_tuple)
        else:
            layer_check = None if verify_all_layers else [0, num_layers // 2, num_layers - 1]
            pair_lens: dict[int, int] = {}
            for b in range(1, num_branches):
                if min(lens[0], lens[b]) >= min_shared_tokens:
                    pl = detect_lecp_prefix_len(caches[0], caches[b], layer_check_indices=layer_check)
                    if pl >= min_shared_tokens:
                        pair_lens[b] = pl

            if not pair_lens:
                l_shared = 0
                shared_branch_ids = []
            else:
                best_savings = 0
                best_len = 0
                best_group: list[int] = []
                for cand_l in sorted(set(pair_lens.values()), reverse=True):
                    grp = [0] + [b for b, pl in pair_lens.items() if pl >= cand_l]
                    savings = (len(grp) - 1) * cand_l
                    if savings > best_savings:
                        best_savings = savings
                        best_len = cand_l
                        best_group = grp
                l_shared = best_len
                shared_branch_ids = best_group

            if len(cls._lecp_memo) > 256:
                cls._lecp_memo.clear()
            cls._lecp_memo[memo_key] = (l_shared, tuple(shared_branch_ids))

        shared_set = set(shared_branch_ids)
        unshared_branch_ids = [b for b in range(num_branches) if b not in shared_set]

        shared_prefix_k: dict[int, Any] = {}
        shared_prefix_v: dict[int, Any] = {}
        if l_shared > 0:
            for l in range(num_layers):
                shared_prefix_k[l] = caches[0].key_cache[l][:l_shared]
                shared_prefix_v[l] = caches[0].value_cache[l][:l_shared]
        else:
            for l in range(num_layers):
                shared_prefix_k[l] = None
                shared_prefix_v[l] = None

        branch_suffix_k: list[dict[int, Any]] = [{l: None for l in range(num_layers)} for _ in range(num_branches)]
        branch_suffix_v: list[dict[int, Any]] = [{l: None for l in range(num_layers)} for _ in range(num_branches)]
        branch_specs: list[CoWBranchSpec] = []

        for b in range(num_branches):
            ln = lens[b]
            if b in shared_set:
                suf_len = ln - l_shared
                if suf_len > 0:
                    for l in range(num_layers):
                        branch_suffix_k[b][l] = caches[b].key_cache[l][l_shared:ln]
                        branch_suffix_v[b][l] = caches[b].value_cache[l][l_shared:ln]
                branch_specs.append(
                    CoWBranchSpec(branch_idx=b, total_len=ln, shared_prefix_len=l_shared, suffix_len=suf_len)
                )
            else:
                if ln > 0:
                    for l in range(num_layers):
                        branch_suffix_k[b][l] = caches[b].key_cache[l]
                        branch_suffix_v[b][l] = caches[b].value_cache[l]
                branch_specs.append(
                    CoWBranchSpec(branch_idx=b, total_len=ln, shared_prefix_len=0, suffix_len=ln)
                )

        return cls(
            num_layers=num_layers,
            num_branches=num_branches,
            shared_prefix_len=l_shared,
            shared_branch_ids=shared_branch_ids,
            unshared_branch_ids=unshared_branch_ids,
            shared_prefix_k=shared_prefix_k,
            shared_prefix_v=shared_prefix_v,
            branch_suffix_k=branch_suffix_k,
            branch_suffix_v=branch_suffix_v,
            branch_specs=branch_specs,
        )
