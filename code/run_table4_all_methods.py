"""
Table IV reproduction: 6 methods x 7 datasets cross-dataset comparison.

Methods: Ours(V6), ByteTrack, BotSort, StrongSORT, DeepSORT, OC-SORT
Datasets: UAVSwarm-39, UAVSwarm-25, UAVSwarm-61, Watercraft, Court-3, Court-2, Anti-UAV

Usage (from handoff_bundle root):
    python code/run_table4_all_methods.py
    python code/run_table4_all_methods.py --skip-run
    python code/run_table4_all_methods.py --dataset uavswarm25
    python code/run_table4_all_methods.py --dataset uavswarm25 --method ByteTrack

Environment variables (optional):
    TRACKEVAL_ROOT  — path to TrackEval (default: ../TrackEval relative to bundle)
"""

from __future__ import annotations
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# ── Paths (all bundle-relative) ───────────────────────────────────
BUNDLE = Path(__file__).resolve().parent.parent          # handoff_bundle/
CODE = BUNDLE / "code"
ALG = CODE / "algorithms"
WEIGHTS = BUNDLE / "weights"

# TrackEval: bundle-internal first, then env var override
TRACKEVAL_ROOT = Path(os.environ.get(
    "TRACKEVAL_ROOT", str(BUNDLE / "deps" / "TrackEval")))
GT_ROOT = TRACKEVAL_ROOT / "data" / "gt" / "mot_challenge"
TRK_ROOT = TRACKEVAL_ROOT / "data" / "trackers" / "mot_challenge"
LOG_DIR = BUNDLE / "logs" / "table4"

# Python executable: use current interpreter
PYTHON_EXE = Path(sys.executable)

# ── sys.path: bundle-internal only ────────────────────────────────
# ByteTrack last so its yolox.tracker takes priority over ocsort's yolox/
for p in [
    str(CODE),
    str(ALG / "deepsort"),
    str(ALG / "ocsort"),
    str(ALG / "bytetrack"),  # last insert = first on sys.path
]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Prefer the vendored BoxMOT implementation over any user-site installation.
# This keeps BotSort/StrongSORT reproducible across machines.
VENDORED_BOXMOT = str(ALG / "boxmot")
if VENDORED_BOXMOT not in sys.path:
    sys.path.insert(0, VENDORED_BOXMOT)

# ── Dataset definitions ───────────────────────────────────────────
DATASETS = [
    {
        "name": "uavswarm39",
        "paper_name": "UAVSwarm-39",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "uavswarm39_k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm39_k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm39_k3" / "gt.txt",
        "benchmark": "UAVSwarm39SubK3DetShared",
        "seq": "UAVSwarm39SubK3-01",
        "img_w": 1919, "img_h": 1079,
    },
    {
        "name": "uavswarm25",
        "paper_name": "UAVSwarm-25",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "gt.txt",
        "benchmark": "UAVSwarm25AblK3DetShared",
        "seq": "UAVSwarm25AblK3-01",
        "img_w": 863, "img_h": 411,
    },
    {
        "name": "uavswarm61",
        "paper_name": "UAVSwarm-61",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "uavswarm61_k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm61_k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "uavswarm61_k3" / "gt.txt",
        "benchmark": "UAVSwarm61SubK3DetShared",
        "seq": "UAVSwarm61SubK3-01",
        "img_w": 1279, "img_h": 655,
    },
    {
        "name": "watercraft",
        "paper_name": "Watercraft",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "watercraft_k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "watercraft_k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "watercraft_k3" / "gt.txt",
        "benchmark": "WatercraftSubK3DetShared",
        "seq": "WatercraftSubK3-01",
        "img_w": 3840, "img_h": 2160,
    },
    {
        "name": "count3",
        "paper_name": "Court-3",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "count3_k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "count3_k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "count3_k3" / "gt.txt",
        "benchmark": "Count3SubK3DetShared",
        "seq": "Count3SubK3-01",
        "img_w": 1280, "img_h": 720,
    },
    {
        "name": "count2",
        "paper_name": "Court-2",
        "img_dir": BUNDLE / "data" / "k3_sequences" / "k3" / "img1",
        "det_txt": BUNDLE / "data" / "k3_sequences" / "k3" / "det.txt",
        "gt_txt": BUNDLE / "data" / "k3_sequences" / "k3" / "gt.txt",
        "benchmark": "Count2SubK3DetShared",
        "seq": "Count2SubK3-01",
        "img_w": 1280, "img_h": 720,
    },
    # {
    #     "name": "antiuav",
    #     "paper_name": "Anti-UAV",
    #     "img_dir": BUNDLE / "data" / "antiuav" / "img1",
    #     "det_txt": BUNDLE / "data" / "antiuav" / "det_gt_as_det.txt",
    #     "gt_txt": BUNDLE / "data" / "antiuav" / "gt" / "gt.txt",
    #     "benchmark": "MultiUAV002GTDetShared",
    #     "seq": "MultiUAV002-01",
    #     "img_w": 640, "img_h": 512,
    # },
]

METHODS = ["Ours", "ByteTrack", "BotSort", "StrongSORT", "DeepSORT", "OC-SORT"]

# ── Image / detection loading ─────────────────────────────────────

def parse_frame_id(p: Path) -> int | None:
    stem = p.stem
    if stem.isdigit():
        return int(stem)
    tail = stem.split("_")[-1]
    if tail.isdigit():
        return int(tail)
    return None


def list_images(img_dir: Path) -> List[Tuple[int, Path]]:
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    by_fid: Dict[int, Path] = {}
    for p in files:
        fid = parse_frame_id(p)
        if fid is None:
            continue
        if fid not in by_fid:
            by_fid[fid] = p
    if not by_fid:
        raise RuntimeError(f"No images in {img_dir}")
    return [(fid, by_fid[fid]) for fid in sorted(by_fid.keys())]


def load_det_map(det_txt: Path) -> Dict[int, np.ndarray]:
    dets: defaultdict = defaultdict(list)
    with det_txt.open("r", encoding="ascii", errors="ignore") as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 7:
                continue
            try:
                fid = int(float(p[0]))
                x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
                conf = float(p[6])
            except ValueError:
                continue
            dets[fid].append([x, y, x + w, y + h, conf, 0.0])
    return {k: np.asarray(v, dtype=np.float32) for k, v in dets.items()}


# ── TrackEval setup ───────────────────────────────────────────────

def setup_trackeval(benchmark: str, seq: str, gt_txt: Path,
                    n_frames: int, img_w: int, img_h: int):
    seq_dir = GT_ROOT / f"{benchmark}-train" / seq
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text(
        gt_txt.read_text(encoding="ascii", errors="ignore"), encoding="ascii")
    si = [
        "[Sequence]", f"name={seq}", "imDir=img1", "frameRate=30",
        f"seqLength={n_frames}", f"imWidth={img_w}", f"imHeight={img_h}",
        "imExt=.jpg",
    ]
    (seq_dir / "seqinfo.ini").write_text("\n".join(si) + "\n", encoding="ascii")
    seqmap = GT_ROOT / "seqmaps" / f"{benchmark}-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text(f"name\n{seq}\n", encoding="ascii")


def run_trackeval(benchmark: str, seq: str, tracker_name: str,
                  log_file: Path) -> int:
    cmd = [
        str(PYTHON_EXE),
        str(TRACKEVAL_ROOT / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", "train",
        "--DO_PREPROC", "False",
        "--GT_FOLDER", str(GT_ROOT),
        "--TRACKERS_FOLDER", str(TRK_ROOT),
        "--TRACKER_SUB_FOLDER", "data",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--USE_PARALLEL", "False",
        "--SEQ_INFO", seq,
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(TRACKEVAL_ROOT),
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def setup_combined_trackeval():
    """Create a combined benchmark with all 6 UAV sequences for TrackEval official COMBINED."""
    benchmark = "CombinedUAV6"
    bench_gt = GT_ROOT / f"{benchmark}-train"
    bench_trk = TRK_ROOT / f"{benchmark}-train"

    # Collect all sequence names and build seqmap
    seq_names = []
    for ds in DATASETS[:6]:
        seq = ds["seq"]
        seq_names.append(seq)

        # Copy GT
        src_seq = GT_ROOT / f"{ds['benchmark']}-train" / seq
        dst_seq = bench_gt / seq
        if not dst_seq.exists():
            shutil.copytree(src_seq, dst_seq)

        # Copy tracker outputs for ALL methods under unified names
        for method in METHODS:
            combined_tname = f"combined_{method.lower().replace('-', '')}"
            src_tname = tracker_name_for(method, ds["name"])
            src_trk = TRK_ROOT / f"{ds['benchmark']}-train" / src_tname / "data" / f"{seq}.txt"
            dst_trk = bench_trk / combined_tname / "data" / f"{seq}.txt"
            # Always refresh the combined input. Keeping an existing file here
            # silently mixes tracker configurations across runs.
            if src_trk.exists():
                dst_trk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_trk, dst_trk)

    # Write seqmap (TrackEval looks in GT_ROOT/seqmaps/)
    seqmap = GT_ROOT / "seqmaps" / f"{benchmark}-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seq_names) + "\n", encoding="ascii")


def run_combined_trackeval(log_file: Path) -> int:
    """Run TrackEval on the combined 6-sequence benchmark for all methods."""
    tracker_names = [f"combined_{m.lower().replace('-', '')}" for m in METHODS]
    cmd = [
        str(PYTHON_EXE),
        str(TRACKEVAL_ROOT / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", "CombinedUAV6",
        "--SPLIT_TO_EVAL", "train",
        "--DO_PREPROC", "False",
        "--GT_FOLDER", str(GT_ROOT),
        "--TRACKERS_FOLDER", str(TRK_ROOT),
        "--TRACKER_SUB_FOLDER", "data",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
    ] + ["--TRACKERS_TO_EVAL"] + tracker_names
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(TRACKEVAL_ROOT),
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def parse_combined_metrics() -> dict:
    """Parse TrackEval COMBINED row for all methods. Returns {method: metrics_dict}."""
    results = {}
    for method in METHODS:
        combined_tname = f"combined_{method.lower().replace('-', '')}"
        csv_path = TRK_ROOT / "CombinedUAV6-train" / combined_tname / "pedestrian_detailed.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        row = next((r for r in rows if r.get("seq") in ("COMBINED", "COMBINED_SEQ")), None)
        if row is None:
            continue
        results[method] = {
            "HOTA": float(row["HOTA___AUC"]) * 100.0,
            "MOTA": float(row["MOTA"]) * 100.0,
            "IDF1": float(row["IDF1"]) * 100.0,
            "IDSW": int(float(row["IDSW"])),
        }
    return results


def parse_metrics(benchmark: str, seq: str, tracker_name: str) -> dict | None:
    csv_path = (TRK_ROOT / f"{benchmark}-train" / tracker_name
                / "pedestrian_detailed.csv")
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r.get("seq") == seq), None)
    if row is None:
        return None
    return {
        "HOTA": float(row["HOTA___AUC"]) * 100.0,
        "MOTA": float(row["MOTA"]) * 100.0,
        "IDF1": float(row["IDF1"]) * 100.0,
        "IDSW": int(float(row["IDSW"])),
    }


# ── Tracker runners ───────────────────────────────────────────────

def run_v6(images, det_map, out_txt):
    from V6 import RobustMultiObjectTracker, TrackState
    tracker = RobustMultiObjectTracker(maxDisappeared=30, maxDistance=150)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            objects = tracker.update(img, dets.tolist())
            for oid, obj in objects.items():
                if obj.state == TrackState.LOST:
                    continue
                x1, y1, x2, y2 = map(float, obj.bbox)
                tid = int(oid) + 1
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [V6] {i}/{len(images)}")


def run_v7(images, det_map, out_txt):
    from v7 import RobustMultiObjectTracker
    tracker = RobustMultiObjectTracker(
        maxDisappeared=30, maxDistance=150,
        watermark_mode="temporal",
        feature_weights=[0.25, 0.25, 0.25, 0.25],
    )
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            bboxes = [list(d[:4]) for d in dets]
            objects = tracker.update(img, bboxes)
            for oid, obj in objects.items():
                x1, y1, x2, y2 = map(float, obj.bbox)
                tid = int(oid) + 1
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [V7] {i}/{len(images)}")


def run_bytetrack(images, det_map, out_txt):
    from yolox.tracker.byte_tracker import BYTETracker
    class Args:
        track_thresh = 0.5; track_buffer = 30; match_thresh = 0.8; mot20 = False
    tracker = BYTETracker(Args(), frame_rate=30)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            dets5 = dets[:, :5] if len(dets) else np.zeros((0, 5), dtype=np.float32)
            tracks = tracker.update(dets5, img_info=(h, w), img_size=(h, w))
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                x, y, bw, bh = map(float, t.tlwh)
                tid = int(t.track_id)
                conf = float(getattr(t, "score", 1.0))
                g.write(f"{fid},{tid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{conf:.6f},-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [ByteTrack] {i}/{len(images)}")


def run_botsort(images, det_map, out_txt):
    from boxmot import BotSort
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type == "cuda"
    reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"
    tracker = BotSort(
        reid_weights=reid_weights, device=device, half=half,
        track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
        track_buffer=30, match_thresh=0.8, proximity_thresh=0.5,
        appearance_thresh=0.25, cmc_method="ecc", frame_rate=30,
        # The locked Table IV baseline uses the appearance/ReID branch.
        fuse_first_associate=False, with_reid=True,
    )
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            tracks = tracker.update(dets, img)
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                x1, y1, x2, y2 = map(float, t[0:4])
                tid = int(t[4])
                conf = float(t[5]) if len(t) > 5 else 1.0
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.6f},-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [BotSort] {i}/{len(images)}")


def run_strongsort(images, det_map, out_txt):
    from boxmot import StrongSort
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type == "cuda"
    reid_weights = WEIGHTS / "osnet_x0_25_msmt17.pt"
    tracker = StrongSort(reid_weights=reid_weights, device=device, half=half)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            tracks = tracker.update(dets, img)
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                x1, y1, x2, y2 = map(float, t[0:4])
                tid = int(t[4])
                conf = float(t[5]) if len(t) > 5 else 1.0
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.6f},-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [StrongSORT] {i}/{len(images)}")


def run_deepsort(images, det_map, out_txt):
    from trackers.ocsort_tracker.ocsort import OCSort  # noqa: ensure path
    from deep_sort.deep_sort import DeepSort
    # The locked Table IV baseline uses the downloaded pretrained checkpoint.
    ds_root = ALG / "deepsort"
    ckpt = ds_root / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing locked DeepSORT checkpoint: {ckpt}. "
            "Do not replace it with ckpt_autogen.t7."
        )
    tracker = DeepSort(model_path=str(ckpt), min_confidence=0.1,
                       use_cuda=torch.cuda.is_available())
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets6 = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            if len(dets6):
                x1, y1, x2, y2 = dets6[:, 0], dets6[:, 1], dets6[:, 2], dets6[:, 3]
                bbox_xywh = np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=1)
                confs = dets6[:, 4]
                classes = np.zeros((len(dets6),), dtype=np.int32)
            else:
                bbox_xywh = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int32)
            tracks, _ = tracker.update(bbox_xywh, confs, classes, img)
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                x1, y1, x2, y2 = map(float, t[:4])
                tid = int(t[5])
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [DeepSORT] {i}/{len(images)}")


def run_ocsort(images, det_map, out_txt):
    from trackers.ocsort_tracker.ocsort import OCSort
    tracker = OCSort(det_thresh=0.1, iou_threshold=0.3)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as g:
        for i, (fid, img_path) in enumerate(images, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets6 = det_map.get(fid, np.zeros((0, 6), dtype=np.float32))
            dets5 = dets6[:, :5] if len(dets6) else np.zeros((0, 5), dtype=np.float32)
            tracks = tracker.update(dets5, img_info=(img.shape[0], img.shape[1]),
                                    img_size=(img.shape[0], img.shape[1]))
            if tracks is None or len(tracks) == 0:
                continue
            for t in tracks:
                x1, y1, x2, y2 = map(float, t[:4])
                tid = int(t[4])
                g.write(f"{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")
            if i % 200 == 0:
                print(f"  [OC-SORT] {i}/{len(images)}")


TRACKER_RUNNERS = {
    "Ours": run_v6,
    "V7": run_v7,
    "ByteTrack": run_bytetrack,
    "BotSort": run_botsort,
    "StrongSORT": run_strongsort,
    "DeepSORT": run_deepsort,
    "OC-SORT": run_ocsort,
}


# ── Main ──────────────────────────────────────────────────────────

def tracker_name_for(method: str, ds_name: str) -> str:
    return f"repro_{method.lower().replace('-','')}_{ds_name}"


def run_single_dataset(ds: dict, methods_to_run: List[str], skip_run: bool, force: bool = False):
    ds_name = ds["name"]
    benchmark = ds["benchmark"]
    seq = ds["seq"]
    print(f"\n{'='*60}")
    print(f"Dataset: {ds['paper_name']} ({ds_name})")
    print(f"  benchmark={benchmark}  seq={seq}")
    print(f"{'='*60}")

    # Setup TrackEval GT
    images = list_images(ds["img_dir"])
    n_frames = len(images)
    setup_trackeval(benchmark, seq, ds["gt_txt"], n_frames,
                    ds["img_w"], ds["img_h"])
    print(f"  GT: {n_frames} frames, {ds['img_w']}x{ds['img_h']}")

    det_map = load_det_map(ds["det_txt"])
    print(f"  Detections: {sum(len(v) for v in det_map.values())} total")

    if not skip_run:
        for method in methods_to_run:
            tname = tracker_name_for(method, ds_name)
            out_txt = TRK_ROOT / f"{benchmark}-train" / tname / "data" / f"{seq}.txt"
            if out_txt.exists():
                if force:
                    out_txt.unlink()
                    csv_path = out_txt.parent.parent / "pedestrian_detailed.csv"
                    if csv_path.exists():
                        csv_path.unlink()
                    print(f"  [FORCE] {method} deleted existing output, re-running")
                else:
                    print(f"  [SKIP] {method} already exists: {out_txt}")
                    continue

            print(f"  [RUN] {method} ...")
            t0 = time.time()
            try:
                TRACKER_RUNNERS[method](images, det_map, out_txt)
                elapsed = time.time() - t0
                print(f"  [DONE] {method} in {elapsed:.1f}s")
            except Exception as e:
                print(f"  [ERROR] {method}: {e}")
                continue

            # Run TrackEval
            log_file = LOG_DIR / f"{ds_name}_{tname}.log"
            print(f"  [EVAL] TrackEval for {method} ...")
            rc = run_trackeval(benchmark, seq, tname, log_file)
            if rc != 0:
                print(f"  [WARN] TrackEval rc={rc}, see {log_file}")

    # Collect metrics
    results = {}
    for method in methods_to_run:
        tname = tracker_name_for(method, ds_name)
        m = parse_metrics(benchmark, seq, tname)
        if m is None:
            print(f"  [WARN] No metrics for {method}")
        else:
            print(f"  [METRIC] {method}: HOTA={m['HOTA']:.2f}  MOTA={m['MOTA']:.2f}  "
                  f"IDF1={m['IDF1']:.2f}  IDSW={m['IDSW']}")
        results[method] = m
    return results


def build_table(all_results: dict, combined_metrics: dict) -> str:
    """Build markdown table matching paper Table IV format."""
    dataset_columns = [(ds["paper_name"], ds["name"]) for ds in DATASETS]
    dataset_columns.append(("COMBINED", "COMBINED"))
    datasets_ordered = [label for label, _ in dataset_columns]
    metrics = ["HOTA", "MOTA", "IDF1", "IDSW"]

    lines = []
    lines.append("# Table IV: Cross-Dataset Comparison (Reproduction)\n")
    lines.append("| Method | Metric | " + " | ".join(datasets_ordered) + " |")
    lines.append("|---|---|" + "|".join(["---:"] * len(dataset_columns)) + "|")

    for method in METHODS:
        for mi, metric in enumerate(metrics):
            row = f"| {'**' + method + '**' if mi == 0 else ''} | {metric} |"
            for paper_name, ds_name in dataset_columns:
                m = (combined_metrics.get(method) if ds_name == "COMBINED"
                     else all_results.get(paper_name, {}).get(method))
                if m is None:
                    row += " — |"
                else:
                    val = m[metric]
                    if metric == "IDSW":
                        row += f" {val} |"
                    else:
                        row += f" {val:.2f} |"
            lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table IV")
    parser.add_argument("--skip-run", action="store_true",
                        help="Only rebuild tables from existing results")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset (name)")
    parser.add_argument("--method", type=str, default=None,
                        help="Run only this method")
    parser.add_argument("--paper-only", action="store_true",
                        help="Only show paper values (no run)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    if args.dataset:
        datasets = [ds for ds in DATASETS if ds["name"] == args.dataset]
        if not datasets:
            print(f"Unknown dataset: {args.dataset}")
            return

    methods = METHODS
    if args.method:
        methods = [m for m in METHODS if m.lower() == args.method.lower()]
        if not methods:
            print(f"Unknown method: {args.method}")
            return

    all_results = {}
    for ds in datasets:
        paper_name = ds["paper_name"]
        all_results[paper_name] = run_single_dataset(ds, methods, args.skip_run, args.force)

    # TrackEval official COMBINED (6 UAV sequences)
    combined_metrics = {}
    if len(datasets) >= 6:
        print(f"\n{'='*60}")
        print("TrackEval Official COMBINED (6 UAV sequences)")
        print(f"{'='*60}")
        setup_combined_trackeval()
        combined_log = LOG_DIR / "combined_uav6.log"
        print("  [EVAL] Running TrackEval on CombinedUAV6 ...")
        rc = run_combined_trackeval(combined_log)
        if rc != 0:
            print(f"  [WARN] Combined TrackEval rc={rc}, see {combined_log}")
        combined_metrics = parse_combined_metrics()
        for method in METHODS:
            m = combined_metrics.get(method)
            if m:
                print(f"  [COMBINED] {method}: HOTA={m['HOTA']:.2f}  "
                      f"MOTA={m['MOTA']:.2f}  IDF1={m['IDF1']:.2f}  IDSW={m['IDSW']}")

    # A complete no-filter run must produce every cell in the paper table.
    if not args.skip_run and not args.dataset and not args.method:
        missing = []
        for ds in DATASETS:
            for method in METHODS:
                if all_results.get(ds["paper_name"], {}).get(method) is None:
                    missing.append(f"{method}/{ds['paper_name']}")
        missing.extend(method for method in METHODS if combined_metrics.get(method) is None)
        if missing:
            raise RuntimeError("Incomplete Table IV results: " + ", ".join(missing))

    # Build and save tables
    table_md = build_table(all_results, combined_metrics)

    out_dir = BUNDLE
    (out_dir / "table4_reproduction.md").write_text(
        table_md, encoding="utf-8")
    print(f"\n[DONE] Tables saved to {out_dir / 'table4_reproduction.md'}")

    # Also save CSV
    csv_path = out_dir / "table4_reproduction.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Dataset", "HOTA", "MOTA", "IDF1", "IDSW"])
        for ds_spec in DATASETS:
            pn = ds_spec["paper_name"]
            for method in METHODS:
                m = all_results.get(pn, {}).get(method)
                if m:
                    w.writerow([method, pn,
                                f"{m['HOTA']:.2f}", f"{m['MOTA']:.2f}",
                                f"{m['IDF1']:.2f}", m["IDSW"]])
        # COMBINED rows from TrackEval official evaluation
        for method in METHODS:
            m = combined_metrics.get(method)
            if m:
                w.writerow([method, "COMBINED",
                            f"{m['HOTA']:.2f}", f"{m['MOTA']:.2f}",
                            f"{m['IDF1']:.2f}", m["IDSW"]])
    print(f"[DONE] CSV saved to {csv_path}")

    # Console summary (6 UAV datasets + TrackEval COMBINED)
    uav_datasets = [ds["paper_name"] for ds in DATASETS[:6]]
    print(f"\n{'='*95}")
    print(f"{'Method':12s} {'Metric':6s}", end="")
    for pn in uav_datasets:
        print(f" {pn:>10s}", end="")
    print(f" {'COMBINED':>10s}")
    print("-" * 95)
    for method in METHODS:
        for mi, metric in enumerate(["HOTA", "MOTA", "IDF1", "IDSW"]):
            label = method if mi == 0 else ""
            print(f"{label:12s} {metric:6s}", end="")
            for pn in uav_datasets:
                m = all_results.get(pn, {}).get(method)
                if m:
                    val = m[metric]
                    print(f" {val:>10.2f}" if metric != "IDSW" else f" {val:>10d}", end="")
                else:
                    print(f" {'—':>10s}", end="")
            # TrackEval official COMBINED
            cm = combined_metrics.get(method)
            if cm:
                val = cm[metric]
                print(f" {val:>10.2f}" if metric != "IDSW" else f" {val:>10d}", end="")
            else:
                print(f" {'—':>10s}", end="")
            print()
    print("=" * 95)


if __name__ == "__main__":
    main()
