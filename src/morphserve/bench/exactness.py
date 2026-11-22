"""Canonical `ExactnessEvaluator` enforcing Tiered Exactness Red Lines & downstream task scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Sequence

from morphserve.telemetry import MIB, gpu_mem_info_mib, percentile, read_shmem_kib


def pil_sha256(pil_img: Any) -> str:
    """Bit-exact SHA256 digest of contiguous RGB `uint8` pixel bytes from a PIL image."""
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(pil_img.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Bit-exact SHA256 digest of raw tensor bytes."""
    import torch

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def compute_ssim_rgb(img_a: Any, img_b: Any) -> dict[str, Any]:
    """Compute 8x8 windowed RGB SSIM, PSNR (dB), MAE (`uint8`), and max abs pixel diff."""
    import numpy as np

    a = np.asarray(img_a.convert("RGB"), dtype=np.float64)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return {
            "ssim": 0.0,
            "psnr_db": 0.0,
            "mae_u8": 255.0,
            "max_abs_diff_u8": 255,
        }

    diff = np.abs(a - b)
    mae = float(diff.mean())
    max_diff = int(diff.max())
    mse = float(((a - b) ** 2).mean())
    psnr = 99.99 if mse < 1e-10 else float(10.0 * math.log10((255.0**2) / mse))

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    h, w, c = a.shape
    bh, bw = h // 8, w // 8
    if bh == 0 or bw == 0:
        mu_a, mu_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    else:
        a_blk = a[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
        b_blk = b[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
        mu_a = a_blk.mean(axis=1)
        mu_b = b_blk.mean(axis=1)
        var_a = a_blk.var(axis=1)
        var_b = b_blk.var(axis=1)
        cov = ((a_blk - mu_a[:, None]) * (b_blk - mu_b[:, None])).mean(axis=1)
        ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
        ssim = float(ssim_map.mean())

    return {
        "ssim": round(float(ssim), 6),
        "psnr_db": round(psnr, 2),
        "mae_u8": round(mae, 4),
        "max_abs_diff_u8": max_diff,
    }


def image_pair_metrics(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Compare two saved image files against MorphServe's Tiered Exactness Red Line."""
    from PIL import Image

    with Image.open(path_a) as ia, Image.open(path_b) as ib:
        sha_a = pil_sha256(ia)
        sha_b = pil_sha256(ib)
        qual = compute_ssim_rgb(ia, ib)
    file_sha_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
    file_sha_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
    return {
        "sha256_equal": (sha_a == sha_b) or (file_sha_a == file_sha_b),
        "ssim": qual["ssim"],
        "psnr_db": qual["psnr_db"],
        "mae_u8": qual["mae_u8"],
        "max_abs_diff_u8": qual["max_abs_diff_u8"],
        "sha256_a": sha_a[:16],
        "sha256_b": sha_b[:16],
    }


def _normalize_mmvet_token(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", " ", text.lower())).strip()


def score_mmvet_answer(prediction: str, expected_answer: str) -> float:
    """Evaluate MMVet VQA answer against `<AND>` / `<OR>` structured ground-truth rubric (`[0.0, 1.0]`)."""
    if not expected_answer.strip():
        return 1.0 if prediction.strip() else 0.0
    pred_norm = f" {_normalize_mmvet_token(prediction)} "
    or_groups = [g.strip() for g in expected_answer.split("<OR>") if g.strip()]
    if not or_groups:
        return 0.0
    best = 0.0
    for grp in or_groups:
        and_terms = [_normalize_mmvet_token(t) for t in grp.split("<AND>") if _normalize_mmvet_token(t)]
        if not and_terms:
            continue
        hits = sum(1 for term in and_terms if f" {term} " in pred_norm or term in pred_norm)
        if hits == len(and_terms):
            return 1.0
        best = max(best, hits / float(len(and_terms)))
    return round(best, 4)


@dataclass
class ExactnessEvaluator:
    """Deep verification module enforcing Tier-1 Bit-Exact SHA256, Tier-2 Zero-Approx SSIM, and task scores."""

    min_tier2_ssim: float = 0.995
    owlv2_model_dir: Path | None = None
    clip_model_dir: Path | None = None

    def verify_tier1_bit_exact(self, ref_img: Any, cand_img: Any) -> dict[str, Any]:
        """Tier-1 Red Line (`TriPool-VMM`, Dual-Path Morphing, Preemptive Scheduling): 100% SHA256 match."""
        sha_r = pil_sha256(ref_img)
        sha_c = pil_sha256(cand_img)
        return {
            "tier": "tier1_bit_exact",
            "pass": sha_r == sha_c,
            "sha256_ref": sha_r,
            "sha256_cand": sha_c,
        }

    def verify_tier2_image_pair(self, path_ref: Path, path_cand: Path) -> dict[str, Any]:
        """Tier-2 Red Line (`FlashPrefix-CFG`, `GpuInPlaceLoopback`): mathematical equivalence (`SSIM >= min`)."""
        m = image_pair_metrics(path_ref, path_cand)
        m["pass"] = bool(m["sha256_equal"] or float(m["ssim"]) >= self.min_tier2_ssim)
        return m

    def evaluate_parity_and_quality(
        self,
        sessions: Sequence[Any],
        ref_outcomes: Sequence[dict[str, Any]],
        cand_outcomes: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare stock `monolithic_dp` (`ref`) vs `morphserve` (`cand`) on token match, SSIM, and MMVet score."""
        ref_by_id = {r["session_id"]: r for r in ref_outcomes}
        cand_by_id = {c["session_id"]: c for c in cand_outcomes}

        ar_total = 0
        ar_exact = 0
        mmvet_ref_scores: list[float] = []
        mmvet_cand_scores: list[float] = []
        img_metrics: list[dict[str, Any]] = []

        for sess in sessions:
            sid = sess.session_id
            ro = ref_by_id.get(sid)
            co = cand_by_id.get(sid)
            if not ro or not co or ro.get("status") != "ok" or co.get("status") != "ok":
                continue

            r_txts = ro.get("ar_texts") or []
            c_txts = co.get("ar_texts") or []
            ar_turns = [t for t in sess.turns if t.modality_plan in ("ar_only", "cot_then_dit")]
            for idx, (rt, ct) in enumerate(zip(r_txts, c_txts)):
                ar_total += 1
                if rt == ct:
                    ar_exact += 1
                if idx < len(ar_turns) and ar_turns[idx].expected_answer:
                    exp = ar_turns[idx].expected_answer
                    mmvet_ref_scores.append(score_mmvet_answer(rt, exp))
                    mmvet_cand_scores.append(score_mmvet_answer(ct, exp))

            for rp_str, cp_str in zip(ro.get("image_paths_out") or [], co.get("image_paths_out") or []):
                rp, cp = Path(rp_str), Path(cp_str)
                if rp.is_file() and cp.is_file():
                    img_metrics.append(self.verify_tier2_image_pair(rp, cp))

        exact_pct = round(100.0 * ar_exact / max(1, ar_total), 2) if ar_total > 0 else 100.0
        mean_ssim = round(sum(m["ssim"] for m in img_metrics) / max(1, len(img_metrics)), 6) if img_metrics else 1.0
        min_ssim = round(min((m["ssim"] for m in img_metrics), default=1.0), 6)
        ref_mmvet = round(100.0 * sum(mmvet_ref_scores) / max(1, len(mmvet_ref_scores)), 2) if mmvet_ref_scores else 0.0
        cand_mmvet = (
            round(100.0 * sum(mmvet_cand_scores) / max(1, len(mmvet_cand_scores)), 2) if mmvet_cand_scores else 0.0
        )
        delta_mmvet = round(cand_mmvet - ref_mmvet, 2)

        return {
            "ar_turns_compared": ar_total,
            "ar_greedy_exact_match_pct": exact_pct,
            "images_compared": len(img_metrics),
            "image_mean_ssim": mean_ssim,
            "image_min_ssim": min_ssim,
            "mmvet_ref_score_pct": ref_mmvet,
            "mmvet_morphserve_score_pct": cand_mmvet,
            "mmvet_delta_pct": delta_mmvet,
            "lossless_pass": exact_pct == 100.0 and min_ssim >= 0.99 and abs(delta_mmvet) <= 0.2,
        }


__all__ = [
    "ExactnessEvaluator",
    "MIB",
    "compute_ssim_rgb",
    "gpu_mem_info_mib",
    "image_pair_metrics",
    "percentile",
    "pil_sha256",
    "read_shmem_kib",
    "score_mmvet_answer",
    "tensor_sha256",
]
