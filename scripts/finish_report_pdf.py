"""Finish the report PDF swap once the old PDF is no longer locked.

The PDF is produced by Word COM in ``make_report.py``.  If a PDF viewer
(Foxit / Acrobat / Edge) has the report open, Windows refuses to replace the
file, so the freshly generated PDF is left as ``_new_report.pdf``.  Close the
viewer and run this script -- it retries for a while, then swaps the file in.

    python scripts/finish_report_pdf.py --run runs/ppo_v3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replace the report PDF once unlocked.")
    ap.add_argument("--run", type=str, default="runs/ppo_v3")
    ap.add_argument("--wait", type=int, default=300,
                    help="how long to keep retrying, in seconds")
    args = ap.parse_args(argv)

    run = Path(args.run)
    target = run / "课程报告_专利格式.pdf"
    fresh = run / "_new_report.pdf"
    if not fresh.exists():
        print(f"no freshly generated PDF at {fresh}; run make_report.py first")
        return 1

    deadline = time.time() + args.wait
    attempt = 0
    while True:
        attempt += 1
        try:
            shutil.copyfile(fresh, target)
            fresh.unlink()
            print(f"replaced {target} on attempt {attempt}")
            return 0
        except PermissionError as exc:
            if time.time() >= deadline:
                print(f"still locked after {attempt} attempts: {exc}")
                print(f"close the PDF viewer and re-run; the new file is at {fresh}")
                return 1
            print(f"attempt {attempt}: locked, retrying in 10 s ...")
            time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
