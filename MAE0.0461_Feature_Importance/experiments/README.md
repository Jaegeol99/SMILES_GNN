# Experiment Variants

Each subfolder is an independent training copy of the current project.

- `00_backup_baseline`
  - Unmodified backup of the current codebase.
- `01_3d_global`
  - Adds ETKDG-based 3D global summary features to `g`.
- `02_functional_group_redesign`
  - Replaces atom-level functional-group membership with role-centric features and adds graph-level count/density features.
- `03_topology_chirality`
  - Adds chirality and ring-size atom features plus topology counts to `g`.
- `04_pruned_line_features`
  - Removes weak line features: Laplacian PE, pseudo-angle, and conjugation-flow.

Run from any experiment folder:

```powershell
python train.py --no-plots
```
