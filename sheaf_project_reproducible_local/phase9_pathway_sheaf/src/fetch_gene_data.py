"""
fetch_gene_data.py -- Pull TCGA mutation and GISTIC CNV status for the driver genes
in the approved Bio Lead pathway graph, via the cBioPortal API.

This resolves the "missing gene-level data" constraint that blocked the pathway sheaf:
the team's clinical table doesn't have per-patient gene mutation or CNV status, but
cBioPortal has it for the same patients (study lgggbm_tcga_pub).

Output: tcga_gene_status.csv with columns
    patient_id, <GENE>_mutated (0/1), <GENE>_gistic (-2/-1/0/1/2), <GENE>_amplified (0/1), <GENE>_deleted (0/1)
for each driver gene in DRIVER_GENES.
"""
from __future__ import annotations
import json
import time
import urllib.request
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
STUDY = "lgggbm_tcga_pub"

# Driver genes with Entrez IDs that appear in the approved Bio Lead pathway graph
# (Bio Lead template: pathway_node_list.csv).
DRIVER_GENES = {
    "TP53": 7157, "ATRX": 546, "DAXX": 1616, "CIC": 23152, "FUBP1": 8880,
    "IDH1": 3417, "IDH2": 3418,
    "CDKN2A": 1029, "CDKN2B": 1030, "RB1": 5925, "PTEN": 5728, "NF1": 4763,
    "EGFR": 1956, "PDGFRA": 5156, "MET": 4233, "FGFR1": 2260,
    "PIK3CA": 5290, "PIK3R1": 5295,
    "MDM2": 4193, "MDM4": 4194, "PPM1D": 8493,
    "BRAF": 673, "KRAS": 3845,
    "CDK4": 1019, "CDK6": 1021,
    "NOTCH1": 4851, "SETD2": 29072, "SUZ12": 23512,
    "H3F3A": 3020,
}

UA = "Mozilla/5.0 (research-pull)"
def _get(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def fetch_mutations(entrez: int) -> set:
    url = (f"https://www.cbioportal.org/api/molecular-profiles/{STUDY}_mutations/mutations"
           f"?sampleListId={STUDY}_sequenced&entrezGeneId={entrez}&projection=SUMMARY")
    try:
        records = _get(url)
    except Exception as e:
        print(f"   mutation pull failed: {e}")
        return set()
    return {r["sampleId"] for r in records}

def fetch_gistic(entrez: int) -> dict:
    url = (f"https://www.cbioportal.org/api/molecular-profiles/{STUDY}_gistic/molecular-data"
           f"?sampleListId={STUDY}_cna&entrezGeneId={entrez}&projection=SUMMARY")
    try:
        records = _get(url)
    except Exception as e:
        print(f"   gistic pull failed: {e}")
        return {}
    return {r["sampleId"]: int(r["value"]) for r in records}

def sample_to_patient(s: str) -> str:
    return "-".join(s.split("-")[:3])  # TCGA-XX-XXXX-01 -> TCGA-XX-XXXX

def main():
    all_samples: set = set()
    mut_by_gene = {}
    gistic_by_gene = {}
    for gene, eid in DRIVER_GENES.items():
        print(f"pulling {gene} (Entrez {eid})...", end=" ", flush=True)
        muts = fetch_mutations(eid)
        gist = fetch_gistic(eid)
        mut_by_gene[gene] = muts
        gistic_by_gene[gene] = gist
        all_samples |= muts
        all_samples |= set(gist.keys())
        print(f"mut={len(muts):4d}  gistic={len(gist):4d}")
        time.sleep(0.1)
    # Build the patient-level table
    patient_to_samples: dict = {}
    for s in all_samples:
        p = sample_to_patient(s)
        patient_to_samples.setdefault(p, set()).add(s)
    rows = []
    for p, samples in sorted(patient_to_samples.items()):
        row = {"patient_id": p}
        for gene in DRIVER_GENES:
            row[f"{gene}_mutated"] = int(any(s in mut_by_gene[gene] for s in samples))
            gvals = [gistic_by_gene[gene].get(s) for s in samples if s in gistic_by_gene[gene]]
            gv = max(gvals, key=abs) if gvals else None
            row[f"{gene}_gistic"] = gv
            row[f"{gene}_amplified"] = int(gv == 2) if gv is not None else None
            row[f"{gene}_deleted"] = int(gv == -2) if gv is not None else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tcga_gene_status.csv", index=False)
    print(f"\nwrote {OUT/'tcga_gene_status.csv'}: {len(df)} patients x {len(df.columns)} cols")
    print(f"genes covered: {len(DRIVER_GENES)}")
    print(f"sample mutation rates:")
    for gene in list(DRIVER_GENES)[:8]:
        c = df[f"{gene}_mutated"].sum()
        print(f"  {gene}: {c}/{len(df)} mutated")

if __name__ == "__main__":
    main()
