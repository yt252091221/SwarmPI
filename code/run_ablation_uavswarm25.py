"""
Ablation reproduction for UAVSwarm-25.

Runs V6 with different feature combinations (H/M/S/Z leave-one-out)
via subprocess to reset env vars per run, then evaluates with TrackEval.

Usage (from handoff_bundle root):
    python code/run_ablation_uavswarm25.py
    python code/run_ablation_uavswarm25.py --skip-run

Environment variables (optional):
    TRACKEVAL_ROOT  — path to TrackEval (default: ../TrackEval relative to bundle)
"""

from __future__ import annotations
import csv
import os
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
CODE = BUNDLE / "code"
TRACKEVAL_ROOT = Path(os.environ.get(
    "TRACKEVAL_ROOT", str(BUNDLE / "deps" / "TrackEval")))
GT_ROOT = TRACKEVAL_ROOT / "data" / "gt" / "mot_challenge"
TRK_ROOT = TRACKEVAL_ROOT / "data" / "trackers" / "mot_challenge"
PYTHON_EXE = Path(sys.executable)
LOG_DIR = BUNDLE / "logs" / "ablation"

# UAVSwarm-25 k3 (now in bundle)
DS_IMG = BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "img1"
DS_DET = BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "det.txt"
DS_GT = BUNDLE / "data" / "k3_sequences" / "uavswarm25_k3" / "gt.txt"
BENCHMARK = "UAVSwarm25AblK3DetShared"
SEQ = "UAVSwarm25AblK3-01"
IMG_W, IMG_H, N_FRAMES = 863, 411, 82

# ── Ablation configs ──────────────────────────────────────────────
# H=A(histogram), M=B(motion), S=C(spatial), Z=D(watermark), P=prototype, G=gating
ABLATION_CONFIGS = [
    # Conditional Gain (leave-one-out for H/M/S/Z)
    {"name": "full_HMSZ",   "label": "H+M+S+Z (Full)",   "A": 1, "B": 1, "C": 1, "D": 1, "P": 0, "G": 0},
    {"name": "woH_MSZ",     "label": "M+S+Z (w/o H)",     "A": 0, "B": 1, "C": 1, "D": 1, "P": 0, "G": 0},
    {"name": "woM_HSZZ",    "label": "H+S+Z (w/o M)",     "A": 1, "B": 0, "C": 1, "D": 1, "P": 0, "G": 0},
    {"name": "woS_HMZ",     "label": "H+M+Z (w/o S)",     "A": 1, "B": 1, "C": 0, "D": 1, "P": 0, "G": 0},
    {"name": "woZ_HMS",     "label": "H+M+S (w/o Z)",     "A": 1, "B": 1, "C": 1, "D": 0, "P": 0, "G": 0},
    # Additive chain
    {"name": "only_H",      "label": "H only",            "A": 1, "B": 0, "C": 0, "D": 0, "P": 0, "G": 0},
    {"name": "HM",          "label": "H+M",               "A": 1, "B": 1, "C": 0, "D": 0, "P": 0, "G": 0},
    {"name": "HMS",         "label": "H+M+S",             "A": 1, "B": 1, "C": 1, "D": 0, "P": 0, "G": 0},
    {"name": "HMSZ",        "label": "H+M+S+Z",           "A": 1, "B": 1, "C": 1, "D": 1, "P": 0, "G": 0},
    {"name": "HMSZP",       "label": "H+M+S+Z+P",         "A": 1, "B": 1, "C": 1, "D": 1, "P": 1, "G": 0},
    {"name": "HMSZPG",      "label": "H+M+S+Z+P+G (All)", "A": 1, "B": 1, "C": 1, "D": 1, "P": 1, "G": 1},
]

# Paper reference values for comparison
PAPER_VALUES = {
    "full_HMSZ":  {"HOTA": 91.30, "MOTA": 99.56, "IDF1": 89.91, "IDSW": 6},
    "woH_MSZ":    {"HOTA": 88.28, "MOTA": 99.20, "IDF1": 86.35, "IDSW": 11},
    "woM_HSZZ":   {"HOTA": 80.82, "MOTA": 97.68, "IDF1": 79.59, "IDSW": 32},
    "woS_HMZ":    {"HOTA": 77.43, "MOTA": 97.82, "IDF1": 73.93, "IDSW": 30},
    "woZ_HMS":    {"HOTA": 77.96, "MOTA": 97.02, "IDF1": 74.80, "IDSW": 41},
}


def setup_trackeval():
    seq_dir = GT_ROOT / f"{BENCHMARK}-train" / SEQ
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "gt.txt").write_text(
        DS_GT.read_text(encoding="ascii", errors="ignore"), encoding="ascii")
    si = [
        "[Sequence]", f"name={SEQ}", "imDir=img1", "frameRate=30",
        f"seqLength={N_FRAMES}", f"imWidth={IMG_W}", f"imHeight={IMG_H}",
        "imExt=.jpg",
    ]
    (seq_dir / "seqinfo.ini").write_text("\n".join(si) + "\n", encoding="ascii")
    seqmap = GT_ROOT / "seqmaps" / f"{BENCHMARK}-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text(f"name\n{SEQ}\n", encoding="ascii")


def run_trackeval(tracker_name: str, log_file: Path) -> int:
    cmd = [
        str(PYTHON_EXE),
        str(TRACKEVAL_ROOT / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", BENCHMARK,
        "--SPLIT_TO_EVAL", "train",
        "--DO_PREPROC", "False",
        "--GT_FOLDER", str(GT_ROOT),
        "--TRACKERS_FOLDER", str(TRK_ROOT),
        "--TRACKER_SUB_FOLDER", "data",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--USE_PARALLEL", "False",
        "--SEQ_INFO", SEQ,
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(TRACKEVAL_ROOT),
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def parse_metrics(tracker_name: str) -> dict | None:
    csv_path = TRK_ROOT / f"{BENCHMARK}-train" / tracker_name / "pedestrian_detailed.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r.get("seq") == SEQ), None)
    if row is None:
        return None
    return {
        "HOTA": float(row["HOTA___AUC"]) * 100.0,
        "MOTA": float(row["MOTA"]) * 100.0,
        "IDF1": float(row["IDF1"]) * 100.0,
        "IDSW": int(float(row["IDSW"])),
    }


def run_v6_ablation_subprocess(cfg: dict, force: bool = False) -> bool:
    """Run V6 with specific env vars in a subprocess, save to TrackEval."""
    import tempfile
    tracker_name = f"abl_repro_{cfg['name']}"
    out_txt = TRK_ROOT / f"{BENCHMARK}-train" / tracker_name / "data" / f"{SEQ}.txt"

    if out_txt.exists():
        if not force:
            print(f"  [SKIP] {cfg['label']} already exists")
            return True
        out_txt.unlink()
        print(f"  [FORCE] removed existing output for {cfg['label']}")

    runner_code = (
        "import sys, os, glob\n"
        "sys.path.insert(0, r'" + str(BUNDLE / "code") + "')\n"
        "import cv2, numpy as np\n"
        "from collections import defaultdict\n"
        "from V6 import RobustMultiObjectTracker, TrackState\n"
        "\n"
        "def load_det(det_path):\n"
        "    dets = defaultdict(list)\n"
        "    with open(det_path) as f:\n"
        "        for ln in f:\n"
        "            p = ln.strip().split(',')\n"
        "            if len(p) < 7: continue\n"
        "            try:\n"
        "                fid = int(float(p[0]))\n"
        "                x, y, w, h, c = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])\n"
        "            except: continue\n"
        "            dets[fid].append([x, y, x+w, y+h, c, 0.0])\n"
        "    return {k: np.asarray(v, dtype=np.float32) for k, v in dets.items()}\n"
        "\n"
        "det_map = load_det(r'" + str(DS_DET) + "')\n"
        "img_dir = r'" + str(DS_IMG) + "'\n"
        "out_path = r'" + str(out_txt) + "'\n"
        "os.makedirs(os.path.dirname(out_path), exist_ok=True)\n"
        "tracker = RobustMultiObjectTracker(maxDisappeared=30, maxDistance=150)\n"
        "imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))\n"
        "with open(out_path, 'w') as g:\n"
        "    for i, ip in enumerate(imgs, 1):\n"
        "        img = cv2.imread(ip)\n"
        "        if img is None: continue\n"
        "        stem = os.path.splitext(os.path.basename(ip))[0]\n"
        "        tail = stem.split('_')[-1]\n"
        "        fid = int(tail) if tail.isdigit() else int(stem)\n"
        "        d = det_map.get(fid, np.zeros((0,6), dtype=np.float32))\n"
        "        objs = tracker.update(img, d.tolist())\n"
        "        for oid, obj in objs.items():\n"
        "            if obj.state == TrackState.LOST: continue\n"
        "            x1, y1, x2, y2 = map(float, obj.bbox)\n"
        "            tid = int(oid) + 1\n"
        "            g.write(f'{fid},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\\n')\n"
        "        if i % 50 == 0: print(f'  frame {i}/{len(imgs)}')\n"
        "print('DONE')\n"
    )

    # Write runner to temp file
    tmp = Path(tempfile.mktemp(suffix=".py"))
    tmp.write_text(runner_code, encoding="utf-8")

    env = os.environ.copy()
    env["V6_USE_A"] = str(cfg["A"])
    env["V6_USE_B"] = str(cfg["B"])
    env["V6_USE_C"] = str(cfg["C"])
    env["V6_USE_D"] = str(cfg["D"])
    env["V6_USE_P"] = str(cfg["P"])
    env["V6_USE_G"] = str(cfg["G"])

    print(f"  [RUN] {cfg['label']} ...")
    try:
        result = subprocess.run(
            [str(PYTHON_EXE), str(tmp)],
            capture_output=True, text=True, env=env,
            cwd=str(BUNDLE / "code"),
        )
        if result.returncode != 0:
            print(f"  [ERROR] {cfg['label']}: {result.stderr[-500:]}")
            return False
    finally:
        tmp.unlink(missing_ok=True)

    # Run TrackEval
    log_file = LOG_DIR / f"abl_{cfg['name']}.log"
    rc = run_trackeval(tracker_name, log_file)
    if rc != 0:
        print(f"  [WARN] TrackEval rc={rc}")
    return True


def build_ablation_table(results: dict) -> str:
    lines = []
    lines.append("# Ablation Reproduction — UAVSwarm-25\n")
    lines.append("## Conditional Gain (Leave-one-out)\n")
    lines.append("| Feature Combination | HOTA(%) | IDF1(%) | MOTA(%) | IDSW |")
    lines.append("|---|---:|---:|---:|---:|")

    for cfg in ABLATION_CONFIGS[:5]:
        name = cfg["name"]
        label = cfg["label"]
        m = results.get(name)
        p = PAPER_VALUES.get(name)
        if m:
            hota_diff = f" (Δ{m['HOTA']-p['HOTA']:+.2f})" if p else ""
            lines.append(
                f"| **{label}** | **{m['HOTA']:.2f}**{hota_diff} "
                f"| {m['IDF1']:.2f} | {m['MOTA']:.2f} | {m['IDSW']} |")
        else:
            lines.append(f"| {label} | — | — | — | — |")

    lines.append("")
    lines.append("## Additive Chain\n")
    lines.append("| Stage | Features | HOTA(%) | MOTA(%) | IDF1(%) | IDSW |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cfg in ABLATION_CONFIGS[5:]:
        m = results.get(cfg["name"])
        if m:
            lines.append(
                f"| {cfg['name']} | {cfg['label']} "
                f"| {m['HOTA']:.2f} | {m['MOTA']:.2f} | {m['IDF1']:.2f} | {m['IDSW']} |")
        else:
            lines.append(f"| {cfg['name']} | {cfg['label']} | — | — | — | — |")

    # Paper comparison
    lines.append("")
    lines.append("## Paper Comparison\n")
    lines.append("| Config | Paper HOTA | Repro HOTA | Δ | Paper IDSW | Repro IDSW |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cfg in ABLATION_CONFIGS[:5]:
        name = cfg["name"]
        m = results.get(name)
        p = PAPER_VALUES.get(name)
        if m and p:
            lines.append(
                f"| {cfg['label']} | {p['HOTA']:.2f} | {m['HOTA']:.2f} "
                f"| {m['HOTA']-p['HOTA']:+.2f} | {p['IDSW']} | {m['IDSW']} |")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every ablation configuration even if output exists")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    setup_trackeval()

    results = {}
    for cfg in ABLATION_CONFIGS:
        tracker_name = f"abl_repro_{cfg['name']}"
        if not args.skip_run:
            run_v6_ablation_subprocess(cfg, force=args.force)
        m = parse_metrics(tracker_name)
        if m:
            results[cfg["name"]] = m
            print(f"  [METRIC] {cfg['label']}: HOTA={m['HOTA']:.2f} MOTA={m['MOTA']:.2f} "
                  f"IDF1={m['IDF1']:.2f} IDSW={m['IDSW']}")
        else:
            print(f"  [WARN] No metrics for {cfg['label']}")

    table = build_ablation_table(results)
    out_path = BUNDLE / "ablation_reproduction.md"
    out_path.write_text(table, encoding="utf-8")
    print(f"\n[DONE] Ablation table: {out_path}")

    # CSV
    csv_path = BUNDLE / "ablation_reproduction.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config", "HOTA", "MOTA", "IDF1", "IDSW"])
        for cfg in ABLATION_CONFIGS:
            m = results.get(cfg["name"])
            if m:
                w.writerow([cfg["label"], f"{m['HOTA']:.2f}",
                            f"{m['MOTA']:.2f}", f"{m['IDF1']:.2f}", m["IDSW"]])
    print(f"[DONE] CSV: {csv_path}")


if __name__ == "__main__":
    main()
