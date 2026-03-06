from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CC1=CC(=N)N=CN1"

mol0 = Chem.MolFromSmiles(smiles, sanitize=False)
if mol0 is None:
    raise ValueError("Parse failed")

# sanitize가 되면 가장 좋고, 안 되면 가능한 상태에서 H를 줄여봄
try:
    Chem.SanitizeMol(mol0)
except Exception as e:
    print("SanitizeMol failed (will still try removeHs):", repr(e))

# explicit H 제거 시도
mol = Chem.RemoveHs(mol0, sanitize=False)

# 다시 sanitize 시도
try:
    Chem.SanitizeMol(mol)
except Exception as e:
    print(Chem.MolToSmiles(mol))
    print("SanitizeMol after RemoveHs failed:", repr(e))

img = Draw.MolToImage(mol, size=(400, 400))
img.save("example_smiles_2d_implicitH.png")
print("Saved 2D image -> example_smiles_2d_implicitH.png")