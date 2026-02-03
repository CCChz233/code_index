"""
Run dense retrieval locator (non-overlapping blocks) with a chosen model.
"""
import sys
from method.RepoCoder import run_locator


if __name__ == "__main__":
    # Force dense mode; other args are passed through CLI
    sys.argv.extend(["--mode", "dense"])
    sys.exit(run_locator.run(run_locator.parse_args()))
