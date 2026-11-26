"""Interleaved-Bench v2: 4 orthogonal multi-turn multimodal quadrants & BurstGPT/GenTD26 dual-trace scheduler."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

from morphserve.engine.workload import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    assign_dual_trace_arrivals,
    build_time_window_suite,
    scale_arrivals_to_qps,
)

MMVET_PARQUET_URL = "https://hf-mirror.com/datasets/lmms-lab/MMVet/resolve/main/data/test-00000-of-00001.parquet"
MAGICBRUSH_DEV0_URL = (
    "https://hf-mirror.com/datasets/osunlp/MagicBrush/resolve/main/data/dev-00000-of-00004-f147d414270a90e1.parquet"
)

V2_QUADRANTS = (
    "interleaved_co_creation",
    "multiturn_edit",
    "multi_image_vqa",
    "visual_cot_highres",
)

POST_DIT_INSPECT_PROMPT = (
    "Inspect the generated image carefully: verify object placement, lighting, "
    "and suggest one concrete refinement."
)

REFINEMENT_TEMPLATES = (
    "Add soft natural warm lighting while preserving the main subject and object placement.",
    "Enhance background depth and subtle realistic reflections without altering the foreground composition.",
    "Adjust the scene illumination toward golden hour while keeping fine textures sharp.",
    "Refine color harmony and subtle shadows to make the edited elements blend seamlessly.",
    "Increase foreground detail clarity and balance ambient contrast across the entire frame.",
)


def _normalize_interarrivals(deltas: list[float]) -> list[float]:
    positive = [max(0.0, float(d)) for d in deltas]
    if not positive or sum(positive) <= 0.0:
        return [1.0] * max(1, len(positive))
    mean_d = sum(positive) / float(len(positive))
    return [d / mean_d for d in positive]


def load_burstgpt_interarrivals(csv_path: Path, max_events: int = 2000) -> list[float]:
    """Parse normalized inter-arrival deltas (mean=1.0) from BurstGPT CSV."""
    timestamps: list[float] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_ts = row.get("Timestamp")
            if raw_ts is None:
                continue
            try:
                timestamps.append(float(raw_ts))
            except ValueError:
                continue
            if len(timestamps) >= max_events + 1:
                break
    if len(timestamps) < 2:
        return [1.0]
    deltas = [max(0.0, timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
    return _normalize_interarrivals(deltas)


def load_gentd26_interarrivals(csv_path: Path, max_events: int = 2000) -> list[float]:
    """Parse normalized inter-arrival deltas (mean=1.0) from GenTD26 successful request trace."""
    timestamps: list[float] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("predict_status") not in (None, "", "SUCCEED"):
                continue
            raw_ts = row.get("gmt_create")
            if not raw_ts:
                continue
            try:
                dt = datetime.strptime(raw_ts.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            timestamps.append(dt.timestamp())
            if len(timestamps) >= max_events + 1:
                break
    if len(timestamps) < 2:
        return [1.0]
    deltas = [max(0.0, timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
    return _normalize_interarrivals(deltas)


def _ensure_pyarrow(tmp_dir: Path) -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq
    except ImportError:
        pydeps = tmp_dir / "pydeps"
        pydeps.mkdir(parents=True, exist_ok=True)
        if str(pydeps) not in sys.path:
            sys.path.insert(0, str(pydeps))
        try:
            import pyarrow.parquet as pq  # type: ignore

            return pq
        except ImportError:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--target",
                str(pydeps),
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
                "pyarrow",
            ]
            subprocess.run(cmd, check=True)
            import pyarrow.parquet as pq  # type: ignore

            return pq


def _download_if_missing(url: str, dst: Path) -> Path:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MorphServe-BenchBuilder/2.0"})
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    tmp.replace(dst)
    return dst


def _save_png_capped(raw_cell: Any, dst_path: Path, max_edge: int = 1024) -> bool:
    from PIL import Image

    raw_bytes: bytes | None = None
    if isinstance(raw_cell, dict):
        b = raw_cell.get("bytes")
        if isinstance(b, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(b)
    elif isinstance(raw_cell, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(raw_cell)
    if not raw_bytes:
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw_bytes)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            rgb = rgb.resize((max(16, int(round(w * scale))), max(16, int(round(h * scale)))), Image.BICUBIC)
        rgb.save(dst_path, format="PNG")
    return True


def _extract_mmvet(pq: Any, parquet_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    img_dir = out_dir / "images/mmvet"
    img_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    idx = 0
    with parquet_path.open("rb") as fh:
        pf = pq.ParquetFile(fh)
        cols = [c for c in ("question_id", "image", "question", "answer", "capability") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            for row in batch.to_pylist():
                qid = str(row.get("question_id") or f"mmvet_{idx:04d}")
                rel = f"images/mmvet/{idx:04d}.png"
                idx += 1
                if not _save_png_capped(row.get("image"), out_dir / rel, max_edge=1024):
                    continue
                items.append(
                    {
                        "question_id": qid,
                        "image_path": rel,
                        "question": str(row.get("question") or "").strip(),
                        "answer": str(row.get("answer") or "").strip(),
                        "capability": str(row.get("capability") or "").strip(),
                    }
                )
    (out_dir / "mmvet_questions.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return items


def _extract_magicbrush(pq: Any, parquet_path: Path, out_dir: Path, max_sessions: int = 45) -> list[dict[str, Any]]:
    img_dir = out_dir / "images/magicbrush"
    img_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    src_bytes_by_id: dict[str, bytes] = {}
    with parquet_path.open("rb") as fh:
        pf = pq.ParquetFile(fh)
        cols = [c for c in ("img_id", "turn_index", "source_img", "instruction") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            for row in batch.to_pylist():
                img_id = str(row.get("img_id") or "")
                if not img_id:
                    continue
                t_idx = int(row.get("turn_index") or 1)
                if img_id not in src_bytes_by_id or t_idx == 1:
                    sc = row.get("source_img")
                    b = sc.get("bytes") if isinstance(sc, dict) else sc
                    if isinstance(b, (bytes, bytearray, memoryview)) and img_id not in src_bytes_by_id:
                        src_bytes_by_id[img_id] = bytes(b)
                grouped.setdefault(img_id, []).append(
                    {"turn_index": t_idx, "instruction": str(row.get("instruction") or "").strip()}
                )
            if len(grouped) >= max_sessions * 3:
                break

    sessions: list[dict[str, Any]] = []
    ordered_ids = sorted(grouped.keys(), key=lambda k: (-len(grouped[k]), k))
    for img_id in ordered_ids:
        turns_raw = sorted(grouped[img_id], key=lambda r: int(r["turn_index"]))
        rel_src = f"images/magicbrush/{img_id}_src.png"
        if not _save_png_capped(src_bytes_by_id.get(img_id), out_dir / rel_src, max_edge=1024):
            continue
        turns = [
            {"turn_index": int(r["turn_index"]), "instruction": r["instruction"]}
            for r in turns_raw
            if r["instruction"]
        ]
        if not turns:
            continue
        sessions.append({"img_id": img_id, "source_image": rel_src, "turns": turns})
        if len(sessions) >= max_sessions:
            break

    (out_dir / "magicbrush_sessions.json").write_text(
        json.dumps(sessions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sessions


def _copy_mse_and_load(sources_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mse_img_src = sources_dir / "mse-bench/images"
    mse_img_dst = out_dir / "images/mse-bench"
    if mse_img_src.is_dir() and mse_img_src.resolve() != mse_img_dst.resolve():
        if mse_img_dst.exists():
            shutil.rmtree(mse_img_dst)
        shutil.copytree(mse_img_src, mse_img_dst)

    mse_doc = json.loads((sources_dir / "mse_editing_sessions.json").read_text(encoding="utf-8"))
    templates = list(mse_doc.get("templates", []))

    geneval_items = []
    for idx, line in enumerate((sources_dir / "geneval_evaluation_metadata.jsonl").read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj["index"] = idx
        geneval_items.append(obj)
    return templates, geneval_items


def _allocate_weighted_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Hamilton largest-remainder apportionment of `total` sessions across quadrants."""
    n_tot = max(1, int(total))
    w_sum = sum(max(0.0, float(v)) for v in weights.values())
    if w_sum <= 0.0:
        keys = list(weights.keys())
        base = n_tot // max(1, len(keys))
        counts = {k: base for k in keys}
        for k in keys[: n_tot - sum(counts.values())]:
            counts[k] += 1
        return counts

    quotas: list[tuple[str, int, float]] = []
    used = 0
    for q, w in weights.items():
        exact = n_tot * (max(0.0, float(w)) / w_sum)
        c = int(math.floor(exact))
        quotas.append((q, c, exact - math.floor(exact)))
        used += c
    quotas.sort(key=lambda x: (-x[2], x[0]))
    counts = {q: c for q, c, _ in quotas}
    rem = n_tot - used
    idx = 0
    while rem > 0 and quotas:
        counts[quotas[idx % len(quotas)][0]] += 1
        rem -= 1
        idx += 1
    return counts


def _build_quadrant_pools(
    mmvet_items: list[dict[str, Any]],
    magicbrush_items: list[dict[str, Any]],
    mse_templates: list[dict[str, Any]],
    geneval_items: list[dict[str, Any]],
    seed: int = 2026,
) -> dict[str, list[InterleavedSession]]:
    """Build Interleaved-Bench v2 pools across 4 clean orthogonal multi-turn quadrants."""
    rng = random.Random(seed)
    gv_shuffled = list(geneval_items)
    rng.shuffle(gv_shuffled)

    # Q1: `interleaved_co_creation` (30% of main mix)
    # 4-turn true bidirectional cross-paradigm sessions:
    #   Turn 0: `ar_only` (analyze source image / plan visual scene, max_ar_tokens=48)
    #   Turn 1: `dit_edit` (if Turn 0 had image, `use_prev_image=False` since already in ctx) OR `dit_t2i` (if text-only)
    #   Turn 2: `ar_only` (`use_prev_image=True`, inspect generated/edited image, max_ar_tokens=48)
    #   Turn 3: `dit_edit` (`use_prev_image=True`, concrete visual refinement, same resolution, dit_steps=20)
    co_creation_pool: list[InterleavedSession] = []
    for i in range(60):
        # 25% 1024px / 75% 512px rotated across source types
        hw = 1024 if (((i // 4) + i) % 4 == 3) else 512
        mode = i % 4
        ref_tpl = REFINEMENT_TEMPLATES[i % len(REFINEMENT_TEMPLATES)]

        if mode == 0 and mse_templates:
            tpl = mse_templates[(i // 4) % len(mse_templates)]
            img_name = Path(tpl["source_image"]["path"]).name
            img_rel = f"images/mse-bench/{img_name}"
            raw_turns = tpl.get("turns", [])
            t1_inst = str(raw_turns[0]["prompt"]) if raw_turns else ref_tpl
            t3_inst = str(raw_turns[1]["prompt"]) if len(raw_turns) > 1 else ref_tpl
            t0_prompt = (
                "Analyze the source image composition, lighting, and primary subjects, "
                f"and plan a clean visual edit for: {t1_inst}"
            )
            t0_imgs = [img_rel]
            t1_plan = "dit_edit"
            t1_meta: dict[str, Any] = {}
            src = "mse-bench"
        elif mode == 1 and magicbrush_items:
            mb = magicbrush_items[(i // 4) % len(magicbrush_items)]
            mb_turns = mb.get("turns", [])
            t1_inst = str(mb_turns[0]["instruction"]) if mb_turns else ref_tpl
            t3_inst = str(mb_turns[1]["instruction"]) if len(mb_turns) > 1 else ref_tpl
            t0_prompt = (
                "Analyze the source image layout and foreground objects, "
                f"and plan how to execute this edit cleanly: {t1_inst}"
            )
            t0_imgs = [str(mb["source_image"])]
            t1_plan = "dit_edit"
            t1_meta = {}
            src = "magicbrush"
        elif mode == 2 and mmvet_items:
            mv = mmvet_items[(i // 4) % len(mmvet_items)]
            t0_prompt = (
                f"Analyze the source image carefully to answer '{mv['question']}' "
                "and propose one visual enhancement to clarify the scene."
            )
            t0_imgs = [str(mv["image_path"])] if mv.get("image_path") else []
            t1_plan = "dit_edit" if t0_imgs else "dit_t2i"
            t1_inst = REFINEMENT_TEMPLATES[(i + 1) % len(REFINEMENT_TEMPLATES)]
            t3_inst = REFINEMENT_TEMPLATES[(i + 2) % len(REFINEMENT_TEMPLATES)]
            t1_meta = {}
            src = "mmvet"
        else:
            gv = gv_shuffled[i % len(gv_shuffled)] if gv_shuffled else {"prompt": "a red apple on a wooden table"}
            gv_prompt = str(gv["prompt"])
            t0_prompt = (
                "Plan the exact object placement, perspective, color harmony, and lighting "
                f"before synthesizing the target scene: {gv_prompt}"
            )
            t0_imgs = []
            t1_plan = "dit_t2i"
            t1_inst = gv_prompt
            t3_inst = ref_tpl
            t1_meta = dict(gv) if "include" in gv else {}
            src = "geneval"

        co_creation_pool.append(
            InterleavedSession(
                session_id=f"cocreate_{i:03d}",
                quadrant="interleaved_co_creation",
                source_dataset=src,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=t0_prompt,
                        image_paths=t0_imgs,
                        max_ar_tokens=48,
                        use_prev_image=False,
                    ),
                    InterleavedTurn(
                        turn_idx=1,
                        modality_plan=t1_plan,
                        prompt=t1_inst,
                        image_paths=[],
                        height=hw,
                        width=hw,
                        dit_steps=20,
                        geneval_meta=t1_meta,
                        use_prev_image=False,
                    ),
                    InterleavedTurn(
                        turn_idx=2,
                        modality_plan="ar_only",
                        prompt=POST_DIT_INSPECT_PROMPT,
                        image_paths=[],
                        max_ar_tokens=48,
                        use_prev_image=True,
                    ),
                    InterleavedTurn(
                        turn_idx=3,
                        modality_plan="dit_edit",
                        prompt=t3_inst,
                        image_paths=[],
                        height=hw,
                        width=hw,
                        dit_steps=20,
                        use_prev_image=True,
                    ),
                ],
            )
        )

    # Q2: `multiturn_edit` (25% of main mix)
    # 4-turn chained editing (`dit_edit` x 4, Turn 1..3 `use_prev_image=True`),
    # MagicBrush (extended to 4 turns if shorter) + MSE-Bench, 25% 1024 / 75% 512, dit_steps=20
    edit_pool: list[InterleavedSession] = []
    for i, mb in enumerate(magicbrush_items):
        hw = 1024 if (i % 4 == 3) else 512
        insts = [str(t["instruction"]).strip() for t in mb.get("turns", []) if str(t.get("instruction", "")).strip()]
        while len(insts) < 4:
            insts.append(REFINEMENT_TEMPLATES[(i + len(insts)) % len(REFINEMENT_TEMPLATES)])
        turns = [
            InterleavedTurn(
                turn_idx=ti,
                modality_plan="dit_edit",
                prompt=inst,
                image_paths=[str(mb["source_image"])] if ti == 0 else [],
                height=hw,
                width=hw,
                dit_steps=20,
                use_prev_image=(ti > 0),
            )
            for ti, inst in enumerate(insts[:4])
        ]
        edit_pool.append(
            InterleavedSession(
                session_id=f"edit_mb_{i:03d}",
                quadrant="multiturn_edit",
                source_dataset="magicbrush",
                turns=turns,
            )
        )
    for i, tpl in enumerate(mse_templates[:45]):
        img_name = Path(tpl["source_image"]["path"]).name
        img_rel = f"images/mse-bench/{img_name}"
        hw = 1024 if (i % 4 == 3) else 512
        prompts = [str(t["prompt"]).strip() for t in tpl.get("turns", []) if str(t.get("prompt", "")).strip()]
        while len(prompts) < 4:
            prompts.append(REFINEMENT_TEMPLATES[(i + len(prompts)) % len(REFINEMENT_TEMPLATES)])
        turns = [
            InterleavedTurn(
                turn_idx=ti,
                modality_plan="dit_edit",
                prompt=p_txt,
                image_paths=[img_rel] if ti == 0 else [],
                height=hw,
                width=hw,
                dit_steps=20,
                use_prev_image=(ti > 0),
            )
            for ti, p_txt in enumerate(prompts[:4])
        ]
        edit_pool.append(
            InterleavedSession(
                session_id=f"edit_mse_{i:03d}",
                quadrant="multiturn_edit",
                source_dataset="mse-bench",
                turns=turns,
            )
        )
    rng.shuffle(edit_pool)

    # Q3: `multi_image_vqa` (25% of main mix)
    # 2-3 turn interactive VQA (`ar_only` x 2 or 3, alternating 2-turn and 3-turn across MM-Vet images, max_ar_tokens=64)
    vqa_pool: list[InterleavedSession] = []
    pos = 0
    vqa_idx = 0
    n_mv = len(mmvet_items)
    while n_mv >= 2 and pos + 2 <= n_mv:
        n_turns = 2 if (vqa_idx % 2 == 0) else 3
        if pos + n_turns > n_mv:
            n_turns = 2
        chunk = mmvet_items[pos : pos + n_turns]
        pos += n_turns
        turns_vqa: list[InterleavedTurn] = []
        for ti, item in enumerate(chunk):
            if ti == 0:
                q_prompt = str(item["question"])
            elif ti == 1:
                q_prompt = f"Now inspect the second image as well: {item['question']}"
            else:
                q_prompt = f"Finally, inspect the third image and answer: {item['question']}"
            turns_vqa.append(
                InterleavedTurn(
                    turn_idx=ti,
                    modality_plan="ar_only",
                    prompt=q_prompt,
                    image_paths=[str(item["image_path"])] if item.get("image_path") else [],
                    max_ar_tokens=64,
                    expected_answer=str(item.get("answer", "")),
                    use_prev_image=False,
                )
            )
        vqa_pool.append(
            InterleavedSession(
                session_id=f"vqa_{vqa_idx:03d}",
                quadrant="multi_image_vqa",
                source_dataset="mmvet",
                turns=turns_vqa,
            )
        )
        vqa_idx += 1

    # Q4: `visual_cot_highres` (20% of main mix)
    # 50% `cot_then_dit` (`AR <think> 48 tok -> DiT`) + 50% `dit_t2i` from GenEval, 50% 1024 / 50% 512, dit_steps=20
    vcot_hires_pool: list[InterleavedSession] = []
    for i, gv in enumerate(gv_shuffled[:60]):
        plan = "cot_then_dit" if (i % 2 == 0) else "dit_t2i"
        hw = 1024 if ((i // 2) % 2 == 1) else 512
        vcot_hires_pool.append(
            InterleavedSession(
                session_id=f"vcot_hires_{i:03d}",
                quadrant="visual_cot_highres",
                source_dataset="geneval",
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan=plan,
                        prompt=str(gv["prompt"]),
                        image_paths=[],
                        height=hw,
                        width=hw,
                        dit_steps=20,
                        max_ar_tokens=48,
                        geneval_meta=dict(gv),
                        use_prev_image=False,
                    )
                ],
            )
        )

    return {
        "interleaved_co_creation": co_creation_pool,
        "multiturn_edit": edit_pool,
        "multi_image_vqa": vqa_pool,
        "visual_cot_highres": vcot_hires_pool,
    }


def compose_weighted_suite(
    name: str,
    pools: dict[str, list[InterleavedSession]],
    counts: dict[str, int],
    burstgpt_deltas: list[float],
    gentd26_deltas: list[float],
    target_qps: float = 0.6,
    seed: int = 2026,
) -> WorkloadSuite:
    rng = random.Random(seed)
    chosen: list[InterleavedSession] = []
    for quad, cnt in counts.items():
        pool = pools[quad]
        for i in range(cnt):
            src = pool[i % len(pool)]
            copied_turns = [InterleavedTurn(**asdict(t)) for t in src.turns]
            chosen.append(
                InterleavedSession(
                    session_id=f"{name}_{quad}_{i:03d}",
                    quadrant=src.quadrant,
                    source_dataset=src.source_dataset,
                    turns=copied_turns,
                    phase_label=src.phase_label,
                )
            )
    rng.shuffle(chosen)
    assign_dual_trace_arrivals(chosen, burstgpt_deltas, gentd26_deltas, target_qps=target_qps)
    span_s = round(max((float(s.arrival_offset_s) for s in chosen), default=0.0), 3)
    return WorkloadSuite(
        name=name,
        trace_source="burstgpt+gentd26",
        base_qps=target_qps,
        sessions=chosen,
        schema="morphserve.interleaved_bench.v2",
        arrival_duration_s=span_s if span_s > 0.0 else 240.0,
    )


def build_online_phase_shift_suite(
    pools: dict[str, list[InterleavedSession]],
    burstgpt_deltas: list[float],
    gentd26_deltas: list[float],
    *,
    phase_duration_s: float = 120.0,
    target_qps: float = 0.8,
    seed: int = 2026,
) -> WorkloadSuite:
    """Build a single continuous `3 x phase_duration_s` (`360 s`) online modality phase-shift suite.

    Phases:
      - Phase 0 `[0, 120s]`: `ar80_dit20` (55% multi_image_vqa, 25% interleaved_co_creation, 10% multiturn_edit, 10% visual_cot_highres)
      - Phase 1 `[120, 240s]`: `ar20_dit80` (10% multi_image_vqa, 20% interleaved_co_creation, 40% multiturn_edit, 30% visual_cot_highres)
      - Phase 2 `[240, 360s]`: `ar50_dit50` (30% interleaved_co_creation, 25% multiturn_edit, 25% multi_image_vqa, 20% visual_cot_highres)
    """
    phase_specs = (
        (
            "ar80_dit20",
            {
                "multi_image_vqa": 0.55,
                "interleaved_co_creation": 0.25,
                "multiturn_edit": 0.10,
                "visual_cot_highres": 0.10,
            },
        ),
        (
            "ar20_dit80",
            {
                "multi_image_vqa": 0.10,
                "interleaved_co_creation": 0.20,
                "multiturn_edit": 0.40,
                "visual_cot_highres": 0.30,
            },
        ),
        (
            "ar50_dit50",
            {
                "interleaved_co_creation": 0.30,
                "multiturn_edit": 0.25,
                "multi_image_vqa": 0.25,
                "visual_cot_highres": 0.20,
            },
        ),
    )

    rng = random.Random(seed)
    p_dur = max(1.0, float(phase_duration_s))
    qps = max(1e-4, float(target_qps))
    n_per_phase = max(4, int(round(p_dur * qps)))
    dt_tail = min(0.5 * p_dur, 1.0 / qps)
    target_span = max(0.0, p_dur - dt_tail)

    cursors: dict[str, int] = {q: 0 for q in pools}
    all_sessions: list[InterleavedSession] = []
    phase_schedule: list[dict[str, Any]] = []

    b_base = list(burstgpt_deltas) if burstgpt_deltas else [1.0]
    g_base = list(gentd26_deltas) if gentd26_deltas else [1.0]

    for p_idx, (p_label, weights) in enumerate(phase_specs):
        start_s = round(p_idx * p_dur, 3)
        end_s = round((p_idx + 1) * p_dur, 3)
        active_weights = {q: w for q, w in weights.items() if pools.get(q)}
        counts = _allocate_weighted_counts(active_weights, n_per_phase)

        phase_sessions: list[InterleavedSession] = []
        for quad, cnt in counts.items():
            pool = pools[quad]
            cur = cursors.get(quad, 0)
            for j in range(cnt):
                src = pool[(cur + j) % len(pool)]
                copied_turns = [InterleavedTurn(**asdict(t)) for t in src.turns]
                phase_sessions.append(
                    InterleavedSession(
                        session_id=f"ps_p{p_idx}_{p_label}_{quad}_{j:03d}",
                        quadrant=src.quadrant,
                        source_dataset=src.source_dataset,
                        turns=copied_turns,
                        phase_label=p_label,
                    )
                )
            cursors[quad] = cur + cnt

        rng.shuffle(phase_sessions)
        b_rot = (p_idx * 97) % len(b_base)
        g_rot = (p_idx * 89) % len(g_base)
        b_phase = b_base[b_rot:] + b_base[:b_rot]
        g_phase = g_base[g_rot:] + g_base[:g_rot]
        assign_dual_trace_arrivals(phase_sessions, b_phase, g_phase, target_qps=qps)

        max_raw = max((float(s.arrival_offset_s) for s in phase_sessions), default=0.0)
        scale = (target_span / max_raw) if (max_raw > 0.0 and target_span > 0.0) else 1.0
        for s in phase_sessions:
            s.arrival_offset_s = round(start_s + float(s.arrival_offset_s) * scale, 4)

        phase_sessions.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
        all_sessions.extend(phase_sessions)
        phase_schedule.append(
            {
                "phase_idx": p_idx,
                "phase_label": p_label,
                "start_s": start_s,
                "end_s": end_s,
                "weights": dict(weights),
                "quadrant_counts": counts,
                "num_sessions": len(phase_sessions),
            }
        )

    total_duration_s = round(len(phase_specs) * p_dur, 3)
    warmup_s = 20.0 if total_duration_s > 20.0 else 0.0
    steady_s = round(max(0.0, total_duration_s - warmup_s), 3)

    return WorkloadSuite(
        name=f"phase_shift_{int(round(total_duration_s))}s",
        trace_source="burstgpt+gentd26_online_phase_shift",
        base_qps=qps,
        sessions=all_sessions,
        schema="morphserve.interleaved_bench.v2",
        warmup_s=warmup_s,
        steady_s=steady_s,
        arrival_duration_s=total_duration_s,
        phase_schedule=phase_schedule,
    )


def build_interleaved_bench(sources_dir: Path, output_dir: Path, tmp_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mmvet_json = output_dir / "mmvet_questions.json"
    mb_json = output_dir / "magicbrush_sessions.json"
    if (
        mmvet_json.is_file()
        and mb_json.is_file()
        and (output_dir / "images/mmvet").is_dir()
        and (output_dir / "images/magicbrush").is_dir()
    ):
        mmvet_items = json.loads(mmvet_json.read_text(encoding="utf-8"))
        mb_items = json.loads(mb_json.read_text(encoding="utf-8"))
    else:
        pq = _ensure_pyarrow(tmp_dir)
        mmvet_pq = _download_if_missing(MMVET_PARQUET_URL, tmp_dir / "mmvet_test.parquet")
        mb_pq = _download_if_missing(MAGICBRUSH_DEV0_URL, tmp_dir / "magicbrush_dev0.parquet")
        mmvet_items = _extract_mmvet(pq, mmvet_pq, output_dir)
        mb_items = _extract_magicbrush(pq, mb_pq, output_dir, max_sessions=45)

    mse_templates, geneval_items = _copy_mse_and_load(sources_dir, output_dir)

    burstgpt_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv", max_events=2000)
    gentd26_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv", max_events=2000)

    pools = _build_quadrant_pools(mmvet_items, mb_items, mse_templates, geneval_items, seed=2026)

    suites_dir = output_dir / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)

    # Main v2 4-quadrant mix (80 sessions: 30/25/25/20%)
    main_mix = compose_weighted_suite(
        "main_mix_v2_80",
        pools,
        {
            "interleaved_co_creation": 24,
            "multiturn_edit": 20,
            "multi_image_vqa": 20,
            "visual_cot_highres": 16,
        },
        burstgpt_deltas,
        gentd26_deltas,
        target_qps=0.6,
        seed=2026,
    )
    main_mix.save_json(suites_dir / "main_mix_v2_80.json")
    main_mix.save_json(suites_dir / "main_mix_80.json")

    # Modality-skew suites using the 4 v2 quadrants (60 sessions each)
    for sname, counts in (
        (
            "skew_ar80_dit20",
            {
                "multi_image_vqa": 33,
                "interleaved_co_creation": 15,
                "multiturn_edit": 6,
                "visual_cot_highres": 6,
            },
        ),
        (
            "skew_ar50_dit50",
            {
                "interleaved_co_creation": 18,
                "multiturn_edit": 15,
                "multi_image_vqa": 15,
                "visual_cot_highres": 12,
            },
        ),
        (
            "skew_ar20_dit80",
            {
                "multi_image_vqa": 6,
                "interleaved_co_creation": 12,
                "multiturn_edit": 24,
                "visual_cot_highres": 18,
            },
        ),
    ):
        compose_weighted_suite(sname, pools, counts, burstgpt_deltas, gentd26_deltas, target_qps=0.6, seed=2026).save_json(
            suites_dir / f"{sname}.json"
        )

    # Continuous 360s Online Modality Phase-Shift suite (`3 x 120s`)
    phase_shift_suite = build_online_phase_shift_suite(
        pools,
        burstgpt_deltas,
        gentd26_deltas,
        phase_duration_s=120.0,
        target_qps=0.8,
        seed=2026,
    )
    phase_shift_suite.save_json(suites_dir / "phase_shift_360s.json")

    # Compact 12-session 8-GPU smoke suite covering all 4 v2 quadrants (including full 4-turn Q1 bidirectional!)
    smoke = compose_weighted_suite(
        "smoke_8gpu_12",
        pools,
        {
            "interleaved_co_creation": 3,
            "multiturn_edit": 3,
            "multi_image_vqa": 3,
            "visual_cot_highres": 3,
        },
        burstgpt_deltas,
        gentd26_deltas,
        target_qps=0.8,
        seed=2027,
    )
    for s in smoke.sessions:
        if s.quadrant == "multiturn_edit":
            s.turns = s.turns[:2]
        elif s.quadrant == "multi_image_vqa":
            s.turns = s.turns[:2]
        for t in s.turns:
            if t.modality_plan != "ar_only":
                t.dit_steps = 8
            t.max_ar_tokens = min(t.max_ar_tokens, 24)
    # Guarantee at least one 1024px turn and one 512px turn in smoke suite
    dit_turns = [t for s in smoke.sessions for t in s.turns if t.modality_plan != "ar_only"]
    if dit_turns:
        dit_turns[-1].height = 1024
        dit_turns[-1].width = 1024
        dit_turns[0].height = 512
        dit_turns[0].width = 512
    smoke.save_json(suites_dir / "smoke_8gpu_12.json")

    # Base deterministic GenEval + MM-Vet pool used by downstream accuracy builders
    acc_sessions: list[InterleavedSession] = []
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for gv in geneval_items:
        by_tag.setdefault(str(gv.get("tag", "other")), []).append(gv)
    picked_gv: list[dict[str, Any]] = []
    tags = sorted(by_tag.keys())
    idx_per_tag = 0
    while len(picked_gv) < 48 and tags:
        added = False
        for tag in tags:
            if idx_per_tag < len(by_tag[tag]) and len(picked_gv) < 48:
                picked_gv.append(by_tag[tag][idx_per_tag])
                added = True
        if not added:
            break
        idx_per_tag += 1
    for i, gv in enumerate(picked_gv):
        acc_sessions.append(
            InterleavedSession(
                session_id=f"acc_geneval_{i:03d}",
                quadrant="visual_cot_highres",
                source_dataset="geneval",
                arrival_offset_s=0.0,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="dit_t2i",
                        prompt=gv["prompt"],
                        height=512,
                        width=512,
                        dit_steps=20,
                        geneval_meta=gv,
                        use_prev_image=False,
                    )
                ],
            )
        )
    for i, mv in enumerate(mmvet_items[:48]):
        acc_sessions.append(
            InterleavedSession(
                session_id=f"acc_mmvet_{i:03d}",
                quadrant="multi_image_vqa",
                source_dataset="mmvet",
                arrival_offset_s=0.0,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=mv["question"],
                        image_paths=[mv["image_path"]],
                        max_ar_tokens=64,
                        expected_answer=mv["answer"],
                        use_prev_image=False,
                    )
                ],
            )
        )
    WorkloadSuite(
        name="accuracy_geneval_mmvet",
        trace_source="deterministic_eval",
        base_qps=1.0,
        sessions=acc_sessions,
        schema="morphserve.interleaved_bench.v2",
        warmup_s=0.0,
        steady_s=0.0,
        arrival_duration_s=0.0,
    ).save_json(suites_dir / "accuracy_geneval_mmvet.json")

    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    total_mib = round(total_bytes / (1024**2), 2)
    if total_mib >= 1024.0:
        raise RuntimeError(f"Interleaved-Bench exceeded <1 GiB budget: {total_mib} MiB")

    manifest = {
        "schema": "morphserve.interleaved_bench.manifest.v2",
        "total_bytes": total_bytes,
        "total_mib": total_mib,
        "quadrants": list(V2_QUADRANTS),
        "counts": {
            "mmvet_items": len(mmvet_items),
            "magicbrush_sessions": len(mb_items),
            "mse_templates": len(mse_templates),
            "geneval_prompts": len(geneval_items),
            "burstgpt_deltas": len(burstgpt_deltas),
            "gentd26_deltas": len(gentd26_deltas),
            "phase_shift_360s_sessions": len(phase_shift_suite.sessions),
        },
        "suites": sorted(p.name for p in suites_dir.glob("*.json")),
        "status": "ok",
    }
    (output_dir / "BENCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MorphServe Interleaved-Bench v2 dataset & traces")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Unpack parquets & compose v2 suites (<1 GiB)")
    b.add_argument("--sources-dir", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--tmp-dir", type=Path, default=Path("/tmp/interleaved_bench_build"))
    args = parser.parse_args(argv)
    if args.cmd == "build":
        rep = build_interleaved_bench(args.sources_dir, args.output_dir, args.tmp_dir)
        print(json.dumps(rep, indent=2), flush=True)
        return 0
    return 1


__all__ = [
    "InterleavedSession",
    "InterleavedTurn",
    "POST_DIT_INSPECT_PROMPT",
    "REFINEMENT_TEMPLATES",
    "V2_QUADRANTS",
    "WorkloadSuite",
    "assign_dual_trace_arrivals",
    "build_interleaved_bench",
    "build_online_phase_shift_suite",
    "build_time_window_suite",
    "compose_weighted_suite",
    "load_burstgpt_interarrivals",
    "load_gentd26_interarrivals",
    "scale_arrivals_to_qps",
]


if __name__ == "__main__":
    raise SystemExit(main())
