#!/usr/bin/env python3
"""Re-run Phase 7 synthesis.
This package ships with the generated tables. To rebuild from raw Phase 1-6 outputs,
run /mnt/data/build_phase7.py in the ChatGPT sandbox, or adapt its paths locally.
"""
from phase7_publication_synthesis import safe_claim_ledger

if __name__ == "__main__":
    print(safe_claim_ledger().to_string(index=False))
