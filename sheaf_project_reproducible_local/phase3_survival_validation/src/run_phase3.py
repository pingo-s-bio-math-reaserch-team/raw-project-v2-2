#!/usr/bin/env python3
from pathlib import Path
from phase3_survival_analysis import run

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    run(root/'data', root/'results', root/'figures')
