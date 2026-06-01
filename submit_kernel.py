"""
Submit a completed Kaggle kernel version to a code competition.
"""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")

import requests
from requests.auth import HTTPBasicAuth
from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = sys.argv[1]
DESC   = sys.argv[2] if len(sys.argv) > 2 else "auto-submit"
COMP   = "birdclef-2026"
FILE   = "submission.csv"

with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    creds = json.load(f)
auth = HTTPBasicAuth(creds["username"], creds["key"])

# Get latest version number
r = requests.get(f"https://www.kaggle.com/api/v1/kernels/pull/{KERNEL}", auth=auth, timeout=30)
if r.status_code != 200:
    print(f"ERR pull: {r.status_code} {r.text[:200]}")
    sys.exit(1)
md = r.json().get("metadata", {})
version = md.get("currentVersionNumber") or md.get("currentVersionNumberNullable")
print(f"Kernel {KERNEL} version: {version}")

api = KaggleApi()
api.authenticate()

print(f"Submitting v{version} of {KERNEL} to {COMP}...")
try:
    result = api.competition_submit_cli(
        file_name=FILE,
        message=DESC,
        competition=COMP,
        kernel=KERNEL,
        version=str(version),
    )
    print("OK:", result)
except Exception as e:
    print(f"ERR: {type(e).__name__}: {e}")
    sys.exit(1)
