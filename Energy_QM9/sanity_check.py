import csv
import pandas as pd
from rdkit import Chem

CSV_PATH = "qm9_valid_smiles_atomization_free_energy.csv"

def rdkit_roundtrip_ok(smiles: str) -> bool:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    s2 = Chem.MolToSmiles(m, isomericSmiles=True)
    return Chem.MolFromSmiles(s2) is not None

def main(in_csv=CSV_PATH, out_log="csv_rdkit_recheck_log.csv"):
    df = pd.read_csv(in_csv)
    ok = 0
    with open(out_log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "smiles", "A_G_eV", "status"])
        for _, r in df.iterrows():
            s = r["smiles"]
            if isinstance(s, str) and rdkit_roundtrip_ok(s):
                ok += 1
                w.writerow([r.get("i", ""), s, r["A_G_eV"], "OK"])
            else:
                w.writerow([r.get("i", ""), s, r.get("A_G_eV", ""), "FAIL"])
    print(f"OK={ok}/{len(df)} -> {out_log}")

if __name__ == "__main__":
    main()