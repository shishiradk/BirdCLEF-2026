"""
Z9: Y0 base (Anthony 0.950) + NOVEL adaptive per-file temporal smoothing.

The idea NOBODY has tried publicly:
- Each test file has 12 windows of predictions.
- Compute per-window prediction entropy.
- For high-confidence windows: keep predictions sharp (no smoothing).
- For low-confidence windows: smooth from neighbors (gain from temporal context).
- Adaptive Gaussian kernel bandwidth based on entropy.

This is custom logic appended AFTER Y0's submission.csv is generated.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")

SRC_NB = r"c:\BirdCLEF+ 2026\anthony_950\birdclef-2026-ensemble-0-950.ipynb"
OUT_DIR = r"c:\BirdCLEF+ 2026\variant_z9_adaptive_smooth"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NB = os.path.join(OUT_DIR, "birdclef-z9-adaptive-smooth.ipynb")

with open(SRC_NB, encoding="utf-8") as f:
    nb = json.load(f)

# Custom adaptive smoothing cell to append at the END
NOVEL_CELL = (
    "# ============================================================\n"
    "# NOVEL POSTPROC: Adaptive per-file temporal smoothing\n"
    "# (not present in any public notebook)\n"
    "# ============================================================\n"
    "import pandas as pd, numpy as np\n"
    "\n"
    "_sub_path = 'submission.csv'\n"
    "if os.path.exists(_sub_path):\n"
    "    _df = pd.read_csv(_sub_path)\n"
    "    _cols = [c for c in _df.columns if c != 'row_id']\n"
    "    _arr = _df[_cols].to_numpy(np.float32)\n"
    "    _row_ids = _df['row_id'].astype(str).to_numpy()\n"
    "    _file_ids = np.array(['_'.join(r.split('_')[:-1]) for r in _row_ids])\n"
    "\n"
    "    # Per-row entropy (uncertainty)\n"
    "    _p = np.clip(_arr, 1e-6, 1 - 1e-6)\n"
    "    _ent = -(_p * np.log(_p) + (1-_p) * np.log(1-_p)).mean(axis=1)\n"
    "    # Normalize entropy to [0, 1] per-file\n"
    "    _ent_norm = _ent.copy()\n"
    "    for _fid in pd.unique(_file_ids):\n"
    "        _m = _file_ids == _fid\n"
    "        if _m.sum() > 1:\n"
    "            _e = _ent[_m]\n"
    "            _emin, _emax = _e.min(), _e.max()\n"
    "            if _emax > _emin:\n"
    "                _ent_norm[_m] = (_e - _emin) / (_emax - _emin)\n"
    "    print(f'Adaptive smoothing: entropy range {_ent.min():.3f}-{_ent.max():.3f}')\n"
    "\n"
    "    # Adaptive Gaussian kernel: bandwidth scales with normalized entropy\n"
    "    # Confident windows (low entropy): bandwidth ~ 0.3 (almost no smoothing)\n"
    "    # Uncertain windows (high entropy): bandwidth ~ 1.5 (significant smoothing)\n"
    "    _BW_MIN = 0.3\n"
    "    _BW_MAX = 1.5\n"
    "    _smoothed = _arr.copy()\n"
    "    _file_count = 0\n"
    "    for _fid in pd.unique(_file_ids):\n"
    "        _m = _file_ids == _fid\n"
    "        if _m.sum() < 3:\n"
    "            continue\n"
    "        _idx = np.where(_m)[0]\n"
    "        _n = len(_idx)\n"
    "        _x = _arr[_idx]\n"
    "        _ent_local = _ent_norm[_idx]\n"
    "        for _i in range(_n):\n"
    "            _bw = _BW_MIN + (_BW_MAX - _BW_MIN) * _ent_local[_i]\n"
    "            _w = np.exp(-((np.arange(_n) - _i) ** 2) / (2 * _bw ** 2))\n"
    "            _w = _w / _w.sum()\n"
    "            _smoothed[_idx[_i]] = (_w[:, None] * _x).sum(axis=0)\n"
    "        _file_count += 1\n"
    "    print(f'Adaptive smoothing applied to {_file_count} files')\n"
    "\n"
    "    # Blend smoothed with original (conservative: 70% original + 30% smoothed)\n"
    "    _BLEND = 0.30\n"
    "    _final = (1 - _BLEND) * _arr + _BLEND * _smoothed\n"
    "    _df[_cols] = _final.astype(np.float32)\n"
    "    _df.to_csv(_sub_path, index=False)\n"
    "    print(f'submission.csv updated with adaptive smoothing (blend={_BLEND})')\n"
    "else:\n"
    "    print('submission.csv not found - skipping adaptive smoothing')\n"
)

nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": NOVEL_CELL,
})

with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

meta = {
    "id": "shishiradhikari11/birdclef-z9-adaptive-smooth",
    "title": "BirdCLEF z9 adaptive smooth",
    "code_file": "birdclef-z9-adaptive-smooth.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [
        "tuckerarrants/bc2026-distilled-sed-public",
        "tuckerarrants/birdclef-2026-waveform-cache",
        "jaejohn/perch-meta",
        "tuckerarrants/perch-v2-no-dft-onnx",
        "rishikeshjani/perch-onnx-for-birdclef-2026",
        "hideyukizushi/sgkfk-202604041716",
    ],
    "kernel_sources": [
        "ashok205/tf-wheels",
        "hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177",
    ],
    "competition_sources": ["birdclef-2026"],
    "model_sources": [
        "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1"
    ],
    "docker_image": "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c",
    "machine_shape": "None",
}
with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"Z9 saved -> {OUT_DIR}")
