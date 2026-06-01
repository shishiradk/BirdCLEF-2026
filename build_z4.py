"""
Z4: Anthony 3-way ensemble PROPERLY configured.
- Uncomment _runSED_once = ... line in cell 3
- Set task1 = 'run SED once' in solutions
- Add Model_73 to active ensemble
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")

SRC_NB = r"c:\BirdCLEF+ 2026\anthony_950\birdclef-2026-ensemble-0-950.ipynb"
OUT_DIR = r"c:\BirdCLEF+ 2026\variant_z4_3way_fixed"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NB = os.path.join(OUT_DIR, "birdclef-z4-3way-fixed.ipynb")

with open(SRC_NB, encoding="utf-8") as f:
    nb = json.load(f)

# Patch 1: Uncomment _runSED_once line in cell 3
old_line = "# _runSED_once = True if 'task1' in solutions and solutions['task1']=='run SED once' else False"
new_line = "_runSED_once = True if 'task1' in solutions and solutions['task1']=='run SED once' else False"

p1 = 0
for c in nb["cells"]:
    s = "".join(c.get("source", []))
    if old_line in s:
        c["source"] = s.replace(old_line, new_line)
        p1 += 1
        print(f"Uncommented _runSED_once in cell.")
        break

# Patch 2: Update solutions dict - enable task1, add Model_73
old_sol_markers = [
    "#'task1'    : 'run SED once',",
    "{'Model':'Model_22','subm':'subm_22.csv','weight':0.0305,'xSED':[],          'LB':'0.928'},",
    "{'Model':'Model_51','subm':'subm_51.csv','weight':0.9695,'xSED':[0.60, 0.40],'LB':'0.949'},",
]
new_task1 = "'task1'     : 'run SED once',"
new_models = (
    "{'Model':'Model_22','subm':'subm_22.csv','weight':0.0290,'xSED':[],          'LB':'0.928'},\n"
    "  {'Model':'Model_51','subm':'subm_51.csv','weight':0.9410,'xSED':[0.60, 0.40],'LB':'0.949'},\n"
    "  {'Model':'Model_73','subm':'subm_73.csv','weight':0.0300,'xSED':[],          'LB':'0.949'},"
)

p2 = 0
for c in nb["cells"]:
    s = "".join(c.get("source", []))
    if all(m in s for m in old_sol_markers):
        new_s = s.replace(old_sol_markers[0], new_task1)
        new_s = new_s.replace(old_sol_markers[1] + "\n", "")
        new_s = new_s.replace(old_sol_markers[2], new_models)
        c["source"] = new_s
        p2 += 1
        check = "".join(c["source"])
        assert "task1" in check and "Model_73" in check
        print("Solutions patched with task1 + 3 models.")
        break

assert p1 == 1 and p2 == 1, f"p1={p1}, p2={p2}"

with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

meta = {
    "id": "shishiradhikari11/birdclef-z4-3way-fixed",
    "title": "BirdCLEF z4 3way fixed",
    "code_file": "birdclef-z4-3way-fixed.ipynb",
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
print(f"Z4 saved -> {OUT_DIR}")
