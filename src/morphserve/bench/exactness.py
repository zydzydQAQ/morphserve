"""Canonical `ExactnessEvaluator` enforcing Tiered Exactness Red Lines & downstream task scoring (`OWLv2`, `CLIP`, `MMVet`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

from morphserve.engine.workload import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.telemetry import MIB, gpu_mem_info_mib, percentile, read_shmem_kib

DEFAULT_OWLV2_REPO = "google/owlv2-base-patch16-ensemble"
DEFAULT_CLIP_REPO = "openai/clip-vit-large-patch14"


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
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    cleaned = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", cleaned)
    cleaned = re.sub(r"[^\w\s.]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


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
        hits = sum(1 for term in and_terms if f" {term} " in pred_norm)
        if hits == len(and_terms):
            return 1.0
        best = max(best, hits / float(len(and_terms)))
    return round(best, 4)


def _has_evaluator_weights(cand: Path) -> bool:
    if not (cand.is_dir() and (cand / "config.json").is_file()):
        return False
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((cand / fn).is_file() for fn in weight_files)


def resolve_evaluator_model_dir(
    explicit_dir: Path | None,
    env_key: str,
    repo_id: str,
    *,
    allow_hf_download: bool = False,
    staging_root: Path | None = None,
) -> Path | None:
    """Resolve an evaluator model directory offline-first, with optional container `/tmp` staging."""
    basename = repo_id.split("/")[-1]
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir))

    local_tmp = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    if staging_root is not None:
        eff_staging = Path(staging_root)
        candidates.append(eff_staging / basename)
    else:
        env_specific = os.environ.get(env_key, "").strip()
        if env_specific:
            candidates.append(Path(env_specific))
        env_root = os.environ.get("MORPHSERVE_EVAL_MODELS_DIR", "").strip()
        if env_root:
            candidates.append(Path(env_root) / basename)
        eff_staging = Path(local_tmp) / "eval_models" if local_tmp else Path("/tmp/morphserve_eval_models")
        candidates.extend(
            (
                Path("/shared/data/MORPHSERVE/eval_models") / basename,
                Path("/shared/data/MORPHSERVE/models") / basename,
                eff_staging / basename,
            )
        )
    for cand in candidates:
        if _has_evaluator_weights(cand):
            return cand

    if not allow_hf_download:
        return None

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        target = eff_staging / basename
        target.mkdir(parents=True, exist_ok=True)
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            endpoint=endpoint,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        if _has_evaluator_weights(target):
            return target
    except Exception:
        return None
    return None


def _box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-9 else 0.0


def _nms_boxes(dets: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    ordered = sorted(dets, key=lambda d: float(d["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in ordered:
        if all(_box_iou_xyxy(cand["box"], prev["box"]) < iou_threshold for prev in kept):
            kept.append(cand)
    return kept


_CANONICAL_RGB_PROTOTYPES: dict[str, tuple[float, float, float]] = {
    "red": (210.0, 45.0, 45.0),
    "orange": (230.0, 130.0, 35.0),
    "yellow": (230.0, 215.0, 50.0),
    "green": (50.0, 175.0, 65.0),
    "blue": (45.0, 95.0, 215.0),
    "purple": (145.0, 60.0, 195.0),
    "pink": (230.0, 135.0, 180.0),
    "brown": (135.0, 85.0, 45.0),
    "black": (30.0, 30.0, 30.0),
    "white": (225.0, 225.0, 225.0),
}


def _crop_matches_color_rgb(pil_img: Any, box: Sequence[float], target_color: str) -> bool:
    import numpy as np

    color = target_color.strip().lower()
    if color not in _CANONICAL_RGB_PROTOTYPES:
        return False
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    x1 = max(0, min(w - 1, int(round(float(box[0])))))
    y1 = max(0, min(h - 1, int(round(float(box[1])))))
    x2 = max(x1 + 1, min(w, int(round(float(box[2])))))
    y2 = max(y1 + 1, min(h, int(round(float(box[3])))))
    # Focus on center 60% of bounding box to reduce background bleed
    cw = max(1, int((x2 - x1) * 0.2))
    ch = max(1, int((y2 - y1) * 0.2))
    patch = arr[y1 + ch : max(y1 + ch + 1, y2 - ch), x1 + cw : max(x1 + cw + 1, x2 - cw)]
    if patch.size == 0:
        patch = arr[y1:y2, x1:x2]
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    best_name = min(
        _CANONICAL_RGB_PROTOTYPES.keys(),
        key=lambda cname: sum((float(mean_rgb[k]) - _CANONICAL_RGB_PROTOTYPES[cname][k]) ** 2 for k in range(3)),
    )
    return best_name == color


def _check_spatial_relation(rel: str, subj_box: Sequence[float], ref_box: Sequence[float]) -> bool:
    sx = 0.5 * (float(subj_box[0]) + float(subj_box[2]))
    sy = 0.5 * (float(subj_box[1]) + float(subj_box[3]))
    rx = 0.5 * (float(ref_box[0]) + float(ref_box[2]))
    ry = 0.5 * (float(ref_box[1]) + float(ref_box[3]))
    r = rel.strip().lower()
    if r == "left of":
        return sx < rx
    if r == "right of":
        return sx > rx
    if r == "above":
        return sy < ry
    if r == "below":
        return sy > ry
    return False


class OwlV2GenEvalScorer:
    """Open-vocabulary GenEval object/count/color/position scorer (`google/owlv2-base-patch16-ensemble`)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        allow_hf_download: bool = False,
        staging_root: Path | None = None,
        score_threshold: float = 0.20,
        iou_threshold: float = 0.50,
        device: str | None = None,
        detector_fn: Callable[[Any, list[str]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self._detector_fn = detector_fn
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None
        self.backend = "unavailable_offline"

        if detector_fn is not None:
            self.backend = "custom_detector"
            return

        resolved = resolve_evaluator_model_dir(
            model_dir,
            "MORPHSERVE_OWLV2_DIR",
            DEFAULT_OWLV2_REPO,
            allow_hf_download=allow_hf_download,
            staging_root=staging_root,
        )
        if resolved is None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Owlv2ForObjectDetection

            dev_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev_str)
            self._processor = AutoProcessor.from_pretrained(str(resolved), local_files_only=True)
            self._model = Owlv2ForObjectDetection.from_pretrained(str(resolved), local_files_only=True).to(self._device)
            self._model.eval()
            self.backend = "transformers_owlv2"
        except Exception:
            self._processor = None
            self._model = None
            self.backend = "unavailable_offline"

    @property
    def available(self) -> bool:
        return self._detector_fn is not None or (self._model is not None and self._processor is not None)

    def detect_queries(self, pil_img: Any, queries: list[str]) -> list[dict[str, Any]]:
        if not queries or not self.available:
            return []
        if self._detector_fn is not None:
            raw = self._detector_fn(pil_img, queries)
            return [d for d in raw if float(d.get("score", 0.0)) >= self.score_threshold]

        import inspect
        import torch

        rgb = pil_img.convert("RGB")
        w, h = rgb.size
        with torch.inference_mode():
            try:
                inputs = self._processor(
                    text=[queries],
                    images=rgb,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            except TypeError:
                inputs = self._processor(
                    text=[queries],
                    images=rgb,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            inputs = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            outputs = self._model(**inputs)

            post_fn = getattr(self._processor, "post_process_grounded_object_detection", None)
            if post_fn is None and hasattr(self._processor, "image_processor"):
                post_fn = getattr(self._processor.image_processor, "post_process_object_detection", None)
            if post_fn is None:
                post_fn = getattr(self._processor, "post_process_object_detection", None)

            dets: list[dict[str, Any]] = []
            if callable(post_fn):
                try:
                    target_sizes = torch.tensor([[h, w]], device=self._device)
                    sig_params = inspect.signature(post_fn).parameters
                    post_kw: dict[str, Any] = {"outputs": outputs, "target_sizes": target_sizes}
                    if "threshold" in sig_params:
                        post_kw["threshold"] = self.score_threshold
                    elif "box_threshold" in sig_params:
                        post_kw["box_threshold"] = self.score_threshold
                    if "text_labels" in sig_params:
                        post_kw["text_labels"] = [queries]
                    processed = post_fn(**post_kw)[0]
                    boxes = processed["boxes"].detach().cpu().tolist()
                    scores = processed["scores"].detach().cpu().tolist()
                    raw_labels = processed.get("labels")
                    txt_labels = processed.get("text_labels")
                    if raw_labels is not None:
                        lbl_list = raw_labels.detach().cpu().tolist() if torch.is_tensor(raw_labels) else list(raw_labels)
                        for bx, sc, lb in zip(boxes, scores, lbl_list):
                            idx = int(lb)
                            if 0 <= idx < len(queries):
                                dets.append({"query": queries[idx], "score": float(sc), "box": [float(v) for v in bx]})
                        return dets
                    if txt_labels is not None:
                        for bx, sc, tl in zip(boxes, scores, list(txt_labels)):
                            q_str = str(tl)
                            if q_str in queries:
                                dets.append({"query": q_str, "score": float(sc), "box": [float(v) for v in bx]})
                        return dets
                except Exception:
                    dets.clear()

            # Direct tensor fallback for forward compatibility across `transformers` major versions
            probs = torch.sigmoid(outputs.logits[0])
            boxes_cxcywh = outputs.pred_boxes[0]
            scale = float(max(h, w))
            scores_max, labels_idx = probs.max(dim=-1)
            keep = torch.nonzero(scores_max >= self.score_threshold, as_tuple=False).reshape(-1)
            for k_idx in keep.tolist():
                cx, cy, bw, bh = [float(x) * scale for x in boxes_cxcywh[k_idx].detach().cpu().tolist()]
                bx = [max(0.0, cx - 0.5 * bw), max(0.0, cy - 0.5 * bh), min(float(w), cx + 0.5 * bw), min(float(h), cy + 0.5 * bh)]
                q_idx = int(labels_idx[k_idx].item())
                dets.append({"query": queries[q_idx], "score": float(scores_max[k_idx].item()), "box": bx})
            return dets

    def score_image(self, pil_img: Any, geneval_meta: dict[str, Any]) -> dict[str, Any]:
        """Verify all `include` / `exclude` / `color` / `position` GenEval clauses on `pil_img`."""
        tag = str(geneval_meta.get("tag", "unknown"))
        if not self.available:
            return {"available": False, "backend": self.backend, "tag": tag, "score": None}

        includes = list(geneval_meta.get("include") or [])
        excludes = list(geneval_meta.get("exclude") or [])
        queries: list[str] = []
        for clause in includes + excludes:
            cls_name = str(clause["class"]).strip()
            q_base = f"a photo of a {cls_name}"
            if q_base not in queries:
                queries.append(q_base)
            if clause.get("color"):
                q_col = f"a photo of a {str(clause['color']).strip()} {cls_name}"
                if q_col not in queries:
                    queries.append(q_col)

        raw_dets = self.detect_queries(pil_img, queries)
        by_query: dict[str, list[dict[str, Any]]] = {}
        for q in queries:
            by_query[q] = _nms_boxes([d for d in raw_dets if d["query"] == q], iou_threshold=self.iou_threshold)

        matched_by_clause: list[list[dict[str, Any]]] = []
        clauses_ok = True

        for clause in includes:
            cls_name = str(clause["class"]).strip()
            req_count = int(clause.get("count", 1))
            base_boxes = list(by_query.get(f"a photo of a {cls_name}", []))

            color = clause.get("color")
            if color:
                col_q = f"a photo of a {str(color).strip()} {cls_name}"
                col_boxes = list(by_query.get(col_q, []))
                rgb_boxes = [b for b in base_boxes if _crop_matches_color_rgb(pil_img, b["box"], str(color))]
                cands = _nms_boxes(col_boxes + rgb_boxes, iou_threshold=self.iou_threshold)
            else:
                cands = base_boxes

            pos = clause.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                rel_str, ref_idx = str(pos[0]), int(pos[1])
                ref_boxes = matched_by_clause[ref_idx] if 0 <= ref_idx < len(matched_by_clause) else []
                if not ref_boxes:
                    cands = []
                else:
                    cands = [
                        b for b in cands if any(_check_spatial_relation(rel_str, b["box"], rb["box"]) for rb in ref_boxes)
                    ]

            matched_by_clause.append(cands)
            if len(cands) < req_count:
                clauses_ok = False

        for exc in excludes:
            cls_name = str(exc["class"]).strip()
            max_disallowed = int(exc.get("count", 1))
            found = by_query.get(f"a photo of a {cls_name}", [])
            if len(found) >= max_disallowed:
                clauses_ok = False

        return {
            "available": True,
            "backend": self.backend,
            "tag": tag,
            "score": 1.0 if clauses_ok else 0.0,
        }


class ClipEditingScorer:
    """Prompt-image (`CLIP-T`) & source-image (`CLIP-I`) alignment scorer (`openai/clip-vit-large-patch14`)."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        allow_hf_download: bool = False,
        staging_root: Path | None = None,
        device: str | None = None,
        image_embed_fn: Callable[[Any], Any] | None = None,
        text_embed_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._image_embed_fn = image_embed_fn
        self._text_embed_fn = text_embed_fn
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None
        self.backend = "unavailable_offline"

        if image_embed_fn is not None and text_embed_fn is not None:
            self.backend = "custom_clip"
            return

        resolved = resolve_evaluator_model_dir(
            model_dir,
            "MORPHSERVE_CLIP_DIR",
            DEFAULT_CLIP_REPO,
            allow_hf_download=allow_hf_download,
            staging_root=staging_root,
        )
        if resolved is None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            dev_str = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev_str)
            self._processor = CLIPProcessor.from_pretrained(str(resolved), local_files_only=True)
            self._model = CLIPModel.from_pretrained(str(resolved), local_files_only=True).to(self._device)
            self._model.eval()
            self.backend = "transformers_clip"
        except Exception:
            self._processor = None
            self._model = None
            self.backend = "unavailable_offline"

    @property
    def available(self) -> bool:
        return (self._image_embed_fn is not None and self._text_embed_fn is not None) or (
            self._model is not None and self._processor is not None
        )

    def _l2_norm_np(self, vec: Any) -> Any:
        import numpy as np

        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(arr))
        return arr / nrm if nrm > 1e-12 else arr

    def embed_image(self, pil_img: Any) -> Any | None:
        if not self.available:
            return None
        rgb = pil_img.convert("RGB")
        if self._image_embed_fn is not None:
            return self._l2_norm_np(self._image_embed_fn(rgb))

        import torch

        with torch.inference_mode():
            inp = self._processor(images=rgb, return_tensors="pt")
            inp = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            raw = self._model.get_image_features(**inp)
            feat = raw if torch.is_tensor(raw) else getattr(raw, "image_embeds", getattr(raw, "pooler_output", raw[0]))
            return self._l2_norm_np(feat[0].detach().float().cpu().numpy())

    def embed_text(self, prompt: str) -> Any | None:
        if not self.available:
            return None
        if self._text_embed_fn is not None:
            return self._l2_norm_np(self._text_embed_fn(prompt))

        import torch

        with torch.inference_mode():
            inp = self._processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt")
            inp = {k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            raw = self._model.get_text_features(**inp)
            feat = raw if torch.is_tensor(raw) else getattr(raw, "text_embeds", getattr(raw, "pooler_output", raw[0]))
            return self._l2_norm_np(feat[0].detach().float().cpu().numpy())

    def score_edit_turn(
        self,
        gen_img: Any,
        prompt: str,
        source_img: Any | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "backend": self.backend, "clip_t_pct": None, "clip_i_pct": None}
        import numpy as np

        e_img = self.embed_image(gen_img)
        e_txt = self.embed_text(prompt)
        clip_t = max(0.0, float(np.dot(e_img, e_txt))) * 100.0
        clip_i: float | None = None
        if source_img is not None:
            e_src = self.embed_image(source_img)
            clip_i = round(max(0.0, float(np.dot(e_img, e_src))) * 100.0, 4)
        return {
            "available": True,
            "backend": self.backend,
            "clip_t_pct": round(clip_t, 4),
            "clip_i_pct": clip_i,
        }

    def pair_similarity_pct(self, img_a: Any, img_b: Any) -> float | None:
        if not self.available:
            return None
        import numpy as np

        ea = self.embed_image(img_a)
        eb = self.embed_image(img_b)
        return round(max(0.0, float(np.dot(ea, eb))) * 100.0, 4)


def _resolve_bench_sources_dir(bench_root: Path) -> Path | None:
    for cand in (bench_root / "sources", bench_root.parent / "bench_sources"):
        if cand.is_dir():
            return cand
    return None


def build_m3_active_accuracy_suite(
    bench_root: Path,
    *,
    num_cot_geneval: int = 48,
    num_edit_4turn: int = 45,
    num_multiturn_mmvet: int = 48,
) -> WorkloadSuite:
    """Build the M3-active multi-turn downstream accuracy suite (`L_shared > 0` + 4-turn GPU loopback).

    Covers all three Module-3 active regimes (`D6`):
      1. `num_cot_geneval` Visual-CoT GenEval sessions (`cot_then_dit`, `<think>` shared causal prefix -> `L_shared > 0`);
      2. `num_edit_4turn` 4-turn MagicBrush / MSE-Bench editing sessions (`LECP CoW` + `Cascade-CFG` + `GpuInPlaceLoopback`);
      3. `num_multiturn_mmvet` multi-turn MMVet VQA sessions (`ar_only` across paired images & questions).
    """
    bench_root = Path(bench_root)
    base_acc_path = bench_root / "suites/accuracy_geneval_mmvet.json"
    base_acc = WorkloadSuite.load_json(base_acc_path) if base_acc_path.is_file() else None

    sessions: list[InterleavedSession] = []

    # 1. 2-Turn Visual-CoT GenEval:
    #    - Turn 0 (`ar_only`) decodes the `<think>` layout plan into `main_ctx` (`>= 48` causal tokens);
    #    - Turn 1 (`dit_t2i`) forks `cfg_text_ctx = main_ctx[:L_shared]` via CoW (`has_hist == True`, `L_shared >= 32`)
    #      and synthesizes the target GenEval scene!
    geneval_turns: list[InterleavedTurn] = []
    if base_acc is not None:
        for s in base_acc.sessions:
            for t in s.turns:
                if t.geneval_meta:
                    geneval_turns.append(t)
    for i in range(min(int(num_cot_geneval), len(geneval_turns))):
        gt = geneval_turns[i]
        plan_prompt = (
            "Think step by step to plan the exact object counts, colors, spatial positions, "
            f"and clean background layout before generating the image: {gt.prompt}"
        )
        sessions.append(
            InterleavedSession(
                session_id=f"acc_cot_geneval_{i:03d}",
                quadrant="visual_cot",
                source_dataset="geneval",
                arrival_offset_s=0.0,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=plan_prompt,
                        max_ar_tokens=48,
                    ),
                    InterleavedTurn(
                        turn_idx=1,
                        modality_plan="dit_t2i",
                        prompt=gt.prompt,
                        height=512,
                        width=512,
                        dit_steps=20,
                        geneval_meta=dict(gt.geneval_meta),
                    ),
                ],
            )
        )

    # 2. 4-Turn Chained Editing (`MSE-Bench` 4-turn templates + `MagicBrush` extended to 4 turns)
    edit_candidates: list[InterleavedSession] = []
    src_dir = _resolve_bench_sources_dir(bench_root)
    mse_json = (src_dir / "mse_editing_sessions.json") if src_dir is not None else None
    if mse_json is not None and mse_json.is_file():
        mse_doc = json.loads(mse_json.read_text(encoding="utf-8"))
        for idx, tpl in enumerate(mse_doc.get("templates", [])):
            img_name = Path(tpl["source_image"]["path"]).name
            img_rel = f"images/mse-bench/{img_name}"
            if not (bench_root / img_rel).is_file():
                continue
            raw_turns = tpl.get("turns", [])
            if len(raw_turns) < 4:
                continue
            edit_candidates.append(
                InterleavedSession(
                    session_id=f"acc_edit4_mse_{idx:03d}",
                    quadrant="multiturn_edit",
                    source_dataset="mse-bench",
                    arrival_offset_s=0.0,
                    turns=[
                        InterleavedTurn(
                            turn_idx=ti,
                            modality_plan="dit_edit",
                            prompt=str(rt["prompt"]),
                            image_paths=[img_rel] if ti == 0 else [],
                            height=512,
                            width=512,
                            dit_steps=20,
                        )
                        for ti, rt in enumerate(raw_turns[:4])
                    ],
                )
            )

    mb_json = bench_root / "magicbrush_sessions.json"
    fallback_refinements = (
        "Add soft natural warm lighting while preserving the edited objects.",
        "Enhance background depth and subtle realistic reflections.",
        "Adjust the lighting to golden hour while keeping the subject intact.",
    )
    mb_candidates: list[InterleavedSession] = []
    if mb_json.is_file():
        mb_items = json.loads(mb_json.read_text(encoding="utf-8"))
        for idx, mb in enumerate(mb_items):
            src_rel = str(mb.get("source_image", ""))
            if not src_rel or not (bench_root / src_rel).is_file():
                continue
            insts = [str(x["instruction"]).strip() for x in mb.get("turns", []) if str(x.get("instruction", "")).strip()]
            if not insts:
                continue
            while len(insts) < 4:
                insts.append(fallback_refinements[(idx + len(insts)) % len(fallback_refinements)])
            mb_candidates.append(
                InterleavedSession(
                    session_id=f"acc_edit4_mb_{idx:03d}",
                    quadrant="multiturn_edit",
                    source_dataset="magicbrush",
                    arrival_offset_s=0.0,
                    turns=[
                        InterleavedTurn(
                            turn_idx=ti,
                            modality_plan="dit_edit",
                            prompt=inst,
                            image_paths=[src_rel] if ti == 0 else [],
                            height=512,
                            width=512,
                            dit_steps=20,
                        )
                        for ti, inst in enumerate(insts[:4])
                    ],
                )
            )

    # Interleave MagicBrush and MSE-Bench 4-turn sessions evenly up to `num_edit_4turn`
    combined_edits: list[InterleavedSession] = []
    max_len = max(len(mb_candidates), len(edit_candidates))
    for i in range(max_len):
        if i < len(mb_candidates) and len(combined_edits) < int(num_edit_4turn):
            combined_edits.append(mb_candidates[i])
        if i < len(edit_candidates) and len(combined_edits) < int(num_edit_4turn):
            combined_edits.append(edit_candidates[i])
    sessions.extend(combined_edits[: int(num_edit_4turn)])

    # 3. Multi-Turn MMVet VQA (`2` chained VQA turns per session)
    mmvet_pool: list[dict[str, str]] = []
    mmvet_json = bench_root / "mmvet_questions.json"
    if mmvet_json.is_file():
        for item in json.loads(mmvet_json.read_text(encoding="utf-8")):
            if (bench_root / str(item.get("image_path", ""))).is_file():
                mmvet_pool.append(item)
    elif base_acc is not None:
        for s in base_acc.sessions:
            if s.source_dataset == "mmvet" and s.turns:
                t0 = s.turns[0]
                mmvet_pool.append(
                    {
                        "image_path": t0.image_paths[0] if t0.image_paths else "",
                        "question": t0.prompt,
                        "answer": t0.expected_answer,
                    }
                )

    if mmvet_pool:
        n_mv = min(int(num_multiturn_mmvet), len(mmvet_pool))
        for i in range(n_mv):
            a = mmvet_pool[i]
            b = mmvet_pool[(i + 1) % len(mmvet_pool)]
            sessions.append(
                InterleavedSession(
                    session_id=f"acc_mmvet_mt_{i:03d}",
                    quadrant="multi_image_vqa",
                    source_dataset="mmvet",
                    arrival_offset_s=0.0,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="ar_only",
                            prompt=str(a["question"]),
                            image_paths=[str(a["image_path"])] if a.get("image_path") else [],
                            max_ar_tokens=64,
                            expected_answer=str(a.get("answer", "")),
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="ar_only",
                            prompt=f"Now inspect the second image as well: {b['question']}",
                            image_paths=[str(b["image_path"])] if b.get("image_path") else [],
                            max_ar_tokens=64,
                            expected_answer=str(b.get("answer", "")),
                        ),
                    ],
                )
            )

    return WorkloadSuite(
        name="accuracy_m3_multiturn_suite",
        trace_source="deterministic_eval",
        base_qps=0.8,
        sessions=sessions,
    )


@dataclass
class ExactnessEvaluator:
    """Deep verification module enforcing Tier-1 Bit-Exact SHA256, Tier-2 Zero-Approx SSIM & task scores."""

    min_tier2_ssim: float = 0.99
    min_chained_ssim: float = 0.96
    owlv2_model_dir: Path | None = None
    clip_model_dir: Path | None = None
    allow_hf_download: bool = False
    staging_root: Path | None = None
    bench_root: Path | None = None
    owlv2_scorer: OwlV2GenEvalScorer | None = field(default=None, repr=False)
    clip_scorer: ClipEditingScorer | None = field(default=None, repr=False)

    def _get_owlv2(self) -> OwlV2GenEvalScorer:
        if self.owlv2_scorer is None:
            self.owlv2_scorer = OwlV2GenEvalScorer(
                model_dir=self.owlv2_model_dir,
                allow_hf_download=self.allow_hf_download,
                staging_root=self.staging_root,
            )
        return self.owlv2_scorer

    def _get_clip(self) -> ClipEditingScorer:
        if self.clip_scorer is None:
            self.clip_scorer = ClipEditingScorer(
                model_dir=self.clip_model_dir,
                allow_hf_download=self.allow_hf_download,
                staging_root=self.staging_root,
            )
        return self.clip_scorer

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

    def verify_tier2_image_pair(self, path_ref: Path, path_cand: Path, *, chained_turn: bool = False) -> dict[str, Any]:
        """Tier-2 Red Line (`Cascade-CFG`, `GpuInPlaceLoopback`): mathematical equivalence (`SSIM >= threshold`)."""
        m = image_pair_metrics(path_ref, path_cand)
        thresh = self.min_chained_ssim if chained_turn else self.min_tier2_ssim
        m["pass"] = bool(m["sha256_equal"] or float(m["ssim"]) >= thresh)
        return m

    def evaluate_parity_and_quality(
        self,
        sessions: Sequence[Any],
        ref_outcomes: Sequence[dict[str, Any]],
        cand_outcomes: Sequence[dict[str, Any]],
        *,
        bench_root: Path | None = None,
    ) -> dict[str, Any]:
        """Compare `monolithic_dp` (`ref`) vs `morphserve` (`cand`) across AR parity, SSIM, MMVet, OWLv2 & CLIP."""
        from PIL import Image

        eff_bench_root = Path(bench_root) if bench_root is not None else self.bench_root
        owlv2 = self._get_owlv2()
        clip = self._get_clip()

        ref_by_id = {r["session_id"]: r for r in ref_outcomes}
        cand_by_id = {c["session_id"]: c for c in cand_outcomes}

        ar_total = 0
        ar_exact = 0
        mmvet_ref_scores: list[float] = []
        mmvet_cand_scores: list[float] = []

        img_metrics: list[dict[str, Any]] = []
        turn0_ssims: list[float] = []
        chained_ssims: list[float] = []
        edit_turn_ssims: dict[int, list[float]] = {}

        geneval_ref_scores: list[float] = []
        geneval_cand_scores: list[float] = []
        geneval_by_tag: dict[str, dict[str, list[float]]] = {}

        clip_t_ref_list: list[float] = []
        clip_t_cand_list: list[float] = []
        clip_i_ref_list: list[float] = []
        clip_i_cand_list: list[float] = []
        clip_pair_sims: list[float] = []

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

            dit_turns = [t for t in sess.turns if t.modality_plan != "ar_only"]
            src_pil = None
            if eff_bench_root is not None and sess.turns and sess.turns[0].image_paths:
                src_p = eff_bench_root / sess.turns[0].image_paths[0]
                if src_p.is_file():
                    with Image.open(src_p) as raw_src:
                        src_pil = raw_src.convert("RGB")
                        src_pil.load()

            r_imgs = ro.get("image_paths_out") or []
            c_imgs = co.get("image_paths_out") or []
            for img_idx, (rp_str, cp_str) in enumerate(zip(r_imgs, c_imgs)):
                rp, cp = Path(rp_str), Path(cp_str)
                if not (rp.is_file() and cp.is_file()):
                    continue
                d_turn = dit_turns[img_idx] if img_idx < len(dit_turns) else None
                has_shared_or_loopback = bool(
                    img_idx > 0
                    or (d_turn is not None and (d_turn.modality_plan == "cot_then_dit" or int(d_turn.turn_idx) > 0))
                )
                pair_m = self.verify_tier2_image_pair(rp, cp, chained_turn=has_shared_or_loopback)
                img_metrics.append(pair_m)
                if has_shared_or_loopback:
                    chained_ssims.append(float(pair_m["ssim"]))
                else:
                    turn0_ssims.append(float(pair_m["ssim"]))

                if d_turn is not None and d_turn.modality_plan == "dit_edit":
                    edit_turn_ssims.setdefault(int(d_turn.turn_idx), []).append(float(pair_m["ssim"]))

                if (d_turn is not None and d_turn.geneval_meta and owlv2.available) or (
                    d_turn is not None and d_turn.modality_plan == "dit_edit" and clip.available
                ):
                    with Image.open(rp) as rim, Image.open(cp) as cim:
                        r_rgb = rim.convert("RGB")
                        c_rgb = cim.convert("RGB")
                        if d_turn.geneval_meta and owlv2.available:
                            gr = owlv2.score_image(r_rgb, d_turn.geneval_meta)
                            gc = owlv2.score_image(c_rgb, d_turn.geneval_meta)
                            if gr["score"] is not None and gc["score"] is not None:
                                geneval_ref_scores.append(float(gr["score"]))
                                geneval_cand_scores.append(float(gc["score"]))
                                tg = str(gr["tag"])
                                slot = geneval_by_tag.setdefault(tg, {"ref": [], "cand": []})
                                slot["ref"].append(float(gr["score"]))
                                slot["cand"].append(float(gc["score"]))
                        if d_turn.modality_plan == "dit_edit" and clip.available:
                            cr = clip.score_edit_turn(r_rgb, d_turn.prompt, source_img=src_pil)
                            cc = clip.score_edit_turn(c_rgb, d_turn.prompt, source_img=src_pil)
                            if cr["clip_t_pct"] is not None and cc["clip_t_pct"] is not None:
                                clip_t_ref_list.append(float(cr["clip_t_pct"]))
                                clip_t_cand_list.append(float(cc["clip_t_pct"]))
                            if cr["clip_i_pct"] is not None and cc["clip_i_pct"] is not None:
                                clip_i_ref_list.append(float(cr["clip_i_pct"]))
                                clip_i_cand_list.append(float(cc["clip_i_pct"]))
                            ps = clip.pair_similarity_pct(r_rgb, c_rgb)
                            if ps is not None:
                                clip_pair_sims.append(ps)

        exact_pct = round(100.0 * ar_exact / max(1, ar_total), 2) if ar_total > 0 else 100.0
        mean_ssim = round(sum(m["ssim"] for m in img_metrics) / max(1, len(img_metrics)), 6) if img_metrics else 1.0
        min_ssim = round(min((m["ssim"] for m in img_metrics), default=1.0), 6)
        min_t0_ssim = round(min(turn0_ssims, default=1.0), 6)
        min_chained_ssim_obs = round(min(chained_ssims, default=1.0), 6)

        ref_mmvet = round(100.0 * sum(mmvet_ref_scores) / max(1, len(mmvet_ref_scores)), 2) if mmvet_ref_scores else 0.0
        cand_mmvet = (
            round(100.0 * sum(mmvet_cand_scores) / max(1, len(mmvet_cand_scores)), 2) if mmvet_cand_scores else 0.0
        )
        delta_mmvet = round(cand_mmvet - ref_mmvet, 2)

        ref_gv = (
            round(100.0 * sum(geneval_ref_scores) / len(geneval_ref_scores), 2) if geneval_ref_scores else None
        )
        cand_gv = (
            round(100.0 * sum(geneval_cand_scores) / len(geneval_cand_scores), 2) if geneval_cand_scores else None
        )
        delta_gv = round(cand_gv - ref_gv, 2) if (ref_gv is not None and cand_gv is not None) else None
        gv_tags_summary = {
            tg: {
                "count": len(vals["ref"]),
                "ref_pct": round(100.0 * sum(vals["ref"]) / max(1, len(vals["ref"])), 2),
                "morphserve_pct": round(100.0 * sum(vals["cand"]) / max(1, len(vals["cand"])), 2),
            }
            for tg, vals in sorted(geneval_by_tag.items())
        }

        clip_t_ref = round(sum(clip_t_ref_list) / len(clip_t_ref_list), 3) if clip_t_ref_list else None
        clip_t_cand = round(sum(clip_t_cand_list) / len(clip_t_cand_list), 3) if clip_t_cand_list else None
        clip_t_delta = round(clip_t_cand - clip_t_ref, 3) if (clip_t_ref is not None and clip_t_cand is not None) else None
        clip_i_ref = round(sum(clip_i_ref_list) / len(clip_i_ref_list), 3) if clip_i_ref_list else None
        clip_i_cand = round(sum(clip_i_cand_list) / len(clip_i_cand_list), 3) if clip_i_cand_list else None
        clip_i_delta = round(clip_i_cand - clip_i_ref, 3) if (clip_i_ref is not None and clip_i_cand is not None) else None
        clip_pair_mean = round(sum(clip_pair_sims) / len(clip_pair_sims), 3) if clip_pair_sims else None

        ssim_ok = (min_t0_ssim >= self.min_tier2_ssim) and (min_chained_ssim_obs >= self.min_chained_ssim)
        gv_ok = True if delta_gv is None else (cand_gv >= ref_gv - 2.5)
        clip_ok = True if clip_t_delta is None else (abs(clip_t_delta) <= 0.5)

        return {
            "ar_turns_compared": ar_total,
            "ar_greedy_exact_match_pct": exact_pct,
            "images_compared": len(img_metrics),
            "image_mean_ssim": mean_ssim,
            "image_min_ssim": min_ssim,
            "turn0_min_ssim": min_t0_ssim,
            "chained_multiturn_min_ssim": min_chained_ssim_obs,
            "edit_per_turn_mean_ssim": {
                f"turn_{ti}": round(sum(vals) / len(vals), 6) for ti, vals in sorted(edit_turn_ssims.items())
            },
            "mmvet_ref_score_pct": ref_mmvet,
            "mmvet_morphserve_score_pct": cand_mmvet,
            "mmvet_delta_pct": delta_mmvet,
            "owlv2_backend": owlv2.backend,
            "geneval_evaluated": len(geneval_ref_scores),
            "geneval_ref_score_pct": ref_gv,
            "geneval_morphserve_score_pct": cand_gv,
            "geneval_delta_pct": delta_gv,
            "geneval_by_tag": gv_tags_summary,
            "clip_backend": clip.backend,
            "edit_turns_evaluated": len(clip_t_ref_list),
            "clip_t_ref_pct": clip_t_ref,
            "clip_t_morphserve_pct": clip_t_cand,
            "clip_t_delta_pct": clip_t_delta,
            "clip_i_ref_pct": clip_i_ref,
            "clip_i_morphserve_pct": clip_i_cand,
            "clip_i_delta_pct": clip_i_delta,
            "clip_pair_mean_sim_pct": clip_pair_mean,
            "lossless_pass": bool(exact_pct == 100.0 and ssim_ok and abs(delta_mmvet) <= 0.2 and gv_ok and clip_ok),
        }


__all__ = [
    "ClipEditingScorer",
    "DEFAULT_CLIP_REPO",
    "DEFAULT_OWLV2_REPO",
    "ExactnessEvaluator",
    "MIB",
    "OwlV2GenEvalScorer",
    "build_m3_active_accuracy_suite",
    "compute_ssim_rgb",
    "gpu_mem_info_mib",
    "image_pair_metrics",
    "percentile",
    "pil_sha256",
    "read_shmem_kib",
    "resolve_evaluator_model_dir",
    "score_mmvet_answer",
    "tensor_sha256",
]
