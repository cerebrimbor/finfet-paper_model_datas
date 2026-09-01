from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
scripts = sorted(root.glob("verify_*.py"))
failed = []
for script in scripts:
    print(f"\n=== {script.name} ===")
    rc = subprocess.call([sys.executable, str(script)])
    if rc:
        failed.append(script.name)

if failed:
    print("\nFAILED:", ", ".join(failed))
    raise SystemExit(1)
print("\nALL VERIFICATIONS PASSED")
