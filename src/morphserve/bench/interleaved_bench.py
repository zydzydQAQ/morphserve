"""Interleaved-Bench: 4-quadrant multimodal dataset builder & BurstGPT/GenTD26 dual-trace scheduler."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

MMVET_PARQUET_URL = "https://hf-mirror.com/datasets/lmms-lab/MMVet/resolve/main/data/test-00000-of-00001.parquet"
MAGICBRUSH_DEV0_URL = (
    "https://hf-mirror.com/datasets/osunlp/MagicBrush/resolve/main/data/dev-00000-of-00004-f147d414270a90e1.parquet"
)


@dataclass
class InterleavedTurn:
    """Single conversational turn inside an Interleaved-Bench session."""

    turn_idx: int
    modality_plan: str  # "ar_only" | "cot_then_dit" | "dit_edit" | "dit_t2i"
    prompt: str
    image_paths: list[str] = field(default_factory=list)
    height: int = 512
    width: int = 512
    dit_steps: int = 20
    max_ar_tokens: int = 48
    expected_answer: str = ""
    geneval_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterleavedSession:
    """One multi-turn or single-turn multimodal request session."""

    session_id: str
    quadrant: str  # "visual_cot" | "multiturn_edit" | "multi_image_vqa" | "highres_t2i"
    source_dataset: str
    arrival_offset_s: float = 0.0
    turns: list[InterleavedTurn] = field(default_factory=list)


@dataclass
class WorkloadSuite:
    """Serializable benchmark suite consumed by MorphServeClusterEngine."""

    name: str
    trace_source: str
    base_qps: float
    sessions: list[InterleavedSession]
    schema: str = "morphserve.interleaved_bench.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "trace_source": self.trace_source,
            "base_qps": self.base_qps,
            "num_sessions": len(self.sessions),
            "sessions": [asdict(s) for s in self.sessions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadSuite":
        sessions: list[InterleavedSession] = []
        for row in data.get("sessions", []):
            turns = [InterleavedTurn(**t) for t in row.get("turns", [])]
            sessions.append(
                InterleavedSession(
                    session_id=row["session_id"],
                    quadrant=row["quadrant"],
                    source_dataset=row["source_dataset"],
                    arrival_offset_s=float(row.get("arrival_offset_s", 0.0)),
                    turns=turns,
                )
            )
        return cls(
            name=data["name"],
            trace_source=data.get("trace_source", "burstgpt+gentd26"),
            base_qps=float(data.get("base_qps", 1.0)),
            sessions=sessions,
            schema=data.get("schema", "morphserve.interleaved_bench.v1"),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "WorkloadSuite":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


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


def scale_arrivals_to_qps(norm_deltas: list[float], target_qps: float, count: int) -> list[float]:
    """Convert normalized inter-arrival deltas (mean=1.0) into absolute arrival offsets in seconds."""
    if count <= 0:
        return []
    qps = max(float(target_qps), 1e-6)
    offsets = [0.0]
    for i in range(1, count):
        d = norm_deltas[(i - 1) % len(norm_deltas)]
        offsets.append(round(offsets[-1] + d / qps, 4))
    return offsets


def assign_dual_trace_arrivals(
    sessions: list[InterleavedSession],
    burstgpt_deltas: list[float],
    gentd26_deltas: list[float],
    target_qps: float,
) -> list[InterleavedSession]:
    """Assign BurstGPT offsets to AR/CoT sessions and GenTD26 offsets to DiT sessions."""
    ar_indices = [i for i, s in enumerate(sessions) if s.quadrant in ("multi_image_vqa", "visual_cot")]
    dit_indices = [i for i, s in enumerate(sessions) if s.quadrant not in ("multi_image_vqa", "visual_cot")]
    total = max(1, len(sessions))
    ar_qps = max(1e-4, target_qps * (len(ar_indices) / float(total)))
    dit_qps = max(1e-4, target_qps * (len(dit_indices) / float(total)))

    ar_offs = scale_arrivals_to_qps(burstgpt_deltas, ar_qps, len(ar_indices))
    dit_offs = scale_arrivals_to_qps(gentd26_deltas, dit_qps, len(dit_indices))

    for pos, idx in enumerate(ar_indices):
        sessions[idx].arrival_offset_s = ar_offs[pos]
    for pos, idx in enumerate(dit_indices):
        sessions[idx].arrival_offset_s = dit_offs[pos]
    sessions.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return sessions


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
    req = urllib.request.Request(url, headers={"User-Agent": "MorphServe-BenchBuilder/1.0"})
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


def _build_quadrant_pools(
    mmvet_items: list[dict[str, Any]],
    magicbrush_items: list[dict[str, Any]],
    mse_templates: list[dict[str, Any]],
    geneval_items: list[dict[str, Any]],
    seed: int = 2026,
) -> dict[str, list[InterleavedSession]]:
    rng = random.Random(seed)
    gv_shuffled = list(geneval_items)
    rng.shuffle(gv_shuffled)

    # 1. Visual-CoT (bagel_think style: AR <think> -> DiT synthesis/edit)
    visual_cot_pool: list[InterleavedSession] = []
    for i in range(48):
        if i % 3 == 0 and i < len(mse_templates):
            tpl = mse_templates[i]
            img_name = Path(tpl["source_image"]["path"]).name
            img_rel = f"images/mse-bench/{img_name}"
            prompt = tpl["turns"][0]["prompt"]
            src = "mse-bench"
            imgs = [img_rel]
        else:
            gv = gv_shuffled[i % len(gv_shuffled)]
            prompt = gv["prompt"]
            src = "geneval"
            imgs = []
        hw = 1024 if (i % 4 == 3) else 512
        visual_cot_pool.append(
            InterleavedSession(
                session_id=f"cot_{i:03d}",
                quadrant="visual_cot",
                source_dataset=src,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="cot_then_dit",
                        prompt=prompt,
                        image_paths=imgs,
                        height=hw,
                        width=hw,
                        dit_steps=20,
                        max_ar_tokens=48,
                    )
                ],
            )
        )

    # 2. Multi-Turn Editing (MagicBrush + MSE-Bench)
    edit_pool: list[InterleavedSession] = []
    for i, mb in enumerate(magicbrush_items):
        hw = 1024 if (i % 4 == 3) else 512
        turns = [
            InterleavedTurn(
                turn_idx=ti,
                modality_plan="dit_edit",
                prompt=t["instruction"],
                image_paths=[mb["source_image"]] if ti == 0 else [],
                height=hw,
                width=hw,
                dit_steps=20,
            )
            for ti, t in enumerate(mb["turns"][:4])
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
        turns = [
            InterleavedTurn(
                turn_idx=ti,
                modality_plan="dit_edit",
                prompt=t["prompt"],
                image_paths=[img_rel] if ti == 0 else [],
                height=hw,
                width=hw,
                dit_steps=20,
            )
            for ti, t in enumerate(tpl["turns"][:4])
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

    # 3. Interactive Multi-Image VQA (MMVet pairs/singles)
    vqa_pool: list[InterleavedSession] = []
    for i in range(0, len(mmvet_items) - 1, 2):
        a, b = mmvet_items[i], mmvet_items[i + 1]
        vqa_pool.append(
            InterleavedSession(
                session_id=f"vqa_{i // 2:03d}",
                quadrant="multi_image_vqa",
                source_dataset="mmvet",
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=a["question"],
                        image_paths=[a["image_path"]],
                        max_ar_tokens=64,
                        expected_answer=a["answer"],
                    ),
                    InterleavedTurn(
                        turn_idx=1,
                        modality_plan="ar_only",
                        prompt=f"Now inspect the second image as well: {b['question']}",
                        image_paths=[b["image_path"]],
                        max_ar_tokens=64,
                        expected_answer=b["answer"],
                    ),
                ],
            )
        )

    # 4. High-Res Text-to-Image (GenEval 512^2 + 1024^2)
    t2i_pool: list[InterleavedSession] = []
    for i, gv in enumerate(gv_shuffled[:60]):
        hw = 1024 if (i % 2 == 1) else 512
        t2i_pool.append(
            InterleavedSession(
                session_id=f"t2i_{i:03d}",
                quadrant="highres_t2i",
                source_dataset="geneval",
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="dit_t2i",
                        prompt=gv["prompt"],
                        height=hw,
                        width=hw,
                        dit_steps=20,
                        geneval_meta=gv,
                    )
                ],
            )
        )

    return {
        "visual_cot": visual_cot_pool,
        "multiturn_edit": edit_pool,
        "multi_image_vqa": vqa_pool,
        "highres_t2i": t2i_pool,
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
            # Deep-copy turns so arrival/step edits stay isolated per suite
            copied_turns = [InterleavedTurn(**asdict(t)) for t in src.turns]
            chosen.append(
                InterleavedSession(
                    session_id=f"{name}_{quad}_{i:03d}",
                    quadrant=src.quadrant,
                    source_dataset=src.source_dataset,
                    turns=copied_turns,
                )
            )
    rng.shuffle(chosen)
    assign_dual_trace_arrivals(chosen, burstgpt_deltas, gentd26_deltas, target_qps=target_qps)
    return WorkloadSuite(
        name=name,
        trace_source="burstgpt+gentd26",
        base_qps=target_qps,
        sessions=chosen,
    )


def build_interleaved_bench(sources_dir: Path, output_dir: Path, tmp_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
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

    # Main 4-quadrant mix (80 sessions: 30% visual_cot, 30% multiturn_edit, 25% multi_image_vqa, 15% highres_t2i)
    main_mix = compose_weighted_suite(
        "main_mix_80",
        pools,
        {"visual_cot": 24, "multiturn_edit": 24, "multi_image_vqa": 20, "highres_t2i": 12},
        burstgpt_deltas,
        gentd26_deltas,
        target_qps=0.6,
        seed=2026,
    )
    main_mix.save_json(suites_dir / "main_mix_80.json")

    # Modality-skew suites (80:20 <-> 50:50 <-> 20:80)
    for sname, counts in (
        ("skew_ar80_dit20", {"multi_image_vqa": 36, "visual_cot": 12, "multiturn_edit": 6, "highres_t2i": 6}),
        ("skew_ar50_dit50", {"multi_image_vqa": 18, "visual_cot": 12, "multiturn_edit": 18, "highres_t2i": 12}),
        ("skew_ar20_dit80", {"multi_image_vqa": 6, "visual_cot": 6, "multiturn_edit": 30, "highres_t2i": 18}),
    ):
        compose_weighted_suite(sname, pools, counts, burstgpt_deltas, gentd26_deltas, target_qps=0.6, seed=2026).save_json(
            suites_dir / f"{sname}.json"
        )

    # Compact 12-session 8-GPU smoke suite (covers all 4 quadrants + 1024px + multi-turn edit)
    smoke = compose_weighted_suite(
        "smoke_8gpu_12",
        pools,
        {"visual_cot": 4, "multiturn_edit": 3, "multi_image_vqa": 3, "highres_t2i": 2},
        burstgpt_deltas,
        gentd26_deltas,
        target_qps=0.8,
        seed=2027,
    )
    for s in smoke.sessions:
        if s.quadrant == "multiturn_edit":
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

    # Accuracy suite: 48 stratified GenEval prompts + 48 MMVet VQA samples
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
                quadrant="highres_t2i",
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
                    )
                ],
            )
        )
    WorkloadSuite(
        name="accuracy_geneval_mmvet",
        trace_source="deterministic_eval",
        base_qps=1.0,
        sessions=acc_sessions,
    ).save_json(suites_dir / "accuracy_geneval_mmvet.json")

    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    total_mib = round(total_bytes / (1024**2), 2)
    if total_mib >= 1024.0:
        raise RuntimeError(f"Interleaved-Bench exceeded <1 GiB budget: {total_mib} MiB")

    manifest = {
        "schema": "morphserve.interleaved_bench.manifest.v1",
        "total_bytes": total_bytes,
        "total_mib": total_mib,
        "counts": {
            "mmvet_items": len(mmvet_items),
            "magicbrush_sessions": len(mb_items),
            "mse_templates": len(mse_templates),
            "geneval_prompts": len(geneval_items),
            "burstgpt_deltas": len(burstgpt_deltas),
            "gentd26_deltas": len(gentd26_deltas),
        },
        "suites": sorted(p.name for p in suites_dir.glob("*.json")),
        "status": "ok",
    }
    (output_dir / "BENCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MorphServe Interleaved-Bench dataset & traces")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Unpack parquets & compose suites (<1 GiB)")
    b.add_argument("--sources-dir", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--tmp-dir", type=Path, default=Path("/tmp/interleaved_bench_build"))
    args = parser.parse_args(argv)
    if args.cmd == "build":
        rep = build_interleaved_bench(args.sources_dir, args.output_dir, args.tmp_dir)
        print(json.dumps(rep, indent=2), flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
