# qm9_valid_smiles_atomization_free_energy.csv
# Save only RDKit-sane QM9 entries (SMILES + atomization free energy targets)

import csv
from rdkit import Chem
from torch_geometric.datasets import QM9

# PyG QM9 target indices (per PyG docs): 12..18 are atomization energies (eV)
ATOMIZATION_TARGETS = {
    "A_U0": 12,
    "A_U": 13,
    "A_H": 14,
    "A_G": 15,      # atomization free energy at 298.15 K (eV)
    "A_Cv": 16,
    "A_U0_ZPVE": 17,
    "A_U_ZPVE": 18,
}

def rdkit_sane(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    return can if Chem.MolFromSmiles(can) is not None else None

def get_target(d, idx: int) -> float:
    y = d.y
    return float(y[0, idx].item()) if y.dim() == 2 else float(y[idx].item())

def main(qm9_root="qm9_data", out_csv="qm9_valid_smiles_atomization_free_energy.csv", max_samples=None):
    ds = QM9(qm9_root)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    ok = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "smiles", "A_G_eV"])  # atomization free energy (eV)

        for i in range(n):
            d = ds[i]
            s = getattr(d, "smiles", None)
            if not isinstance(s, str) or not s:
                continue

            can = rdkit_sane(s)
            if can is None:
                continue

            a_g = get_target(d, ATOMIZATION_TARGETS["A_G"])
            w.writerow([i, can, a_g])
            ok += 1

    print(f"saved {ok}/{n} -> {out_csv}")

if __name__ == "__main__":
    main(qm9_root="qm9_data", out_csv="qm9_valid_smiles_atomization_free_energy.csv", max_samples=None)