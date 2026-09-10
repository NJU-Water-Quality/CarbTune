# Data dictionary

## `Input/curves_template.csv`

Each row is one point on a simulated growth curve.

| Column | Type | Unit | Description |
|---|---|---|---|
| `strain` | string | — | Strain identifier used consistently across all inputs |
| `carbon` | string | — | Carbon-source identifier |
| `uptake` | numeric | mmol gDW⁻¹ h⁻¹ | Carbon uptake constraint used for the simulation point |
| `growth` | numeric | h⁻¹ | Simulated specific growth rate |

## `Input/sequences.csv`

| Column | Type | Description |
|---|---|---|
| `strain` | string | Strain identifier matching `curves_template.csv` |
| `sequence` | string | Optional inline amino-acid sequence; blank when `fasta_path` is used |
| `fasta_path` | string | Path to a protein FASTA file; paths in the distributed table are relative to the repository root |

Each FASTA file may contain multiple proteins. Amino-acid composition is aggregated across all residues. ESM-2 produces one mean-pooled vector per protein and then averages protein vectors within each strain.

## `outputs_smiles/carbons_smiles.csv`

| Column | Type | Description |
|---|---|---|
| `carbon` | string | Carbon-source identifier matching `curves_template.csv` |
| `smiles` | string | Canonical SMILES used for RDKit feature calculation |
| `InChIKey` | string | Structure identifier |
| `MolWt` | numeric | Molecular weight |
| `LogP` | numeric | Crippen logP |
| `TPSA` | numeric | Topological polar surface area |
| `HBD` | integer | Hydrogen-bond donor count |
| `HBA` | integer | Hydrogen-bond acceptor count |
| `RotBonds` | integer | Rotatable-bond count |
| `RingCount` | integer | Ring count |
| `HeavyAtom` | integer | Heavy-atom count |

Descriptors used for modeling are recalculated from `smiles` by RDKit. `FractionCSP3` is calculated during feature construction.

## Derived training labels

| Field | Description |
|---|---|
| `max_growth_actual` | Plateau growth value determined using the configured plateau tolerance |
| `u_opt_actual` | First uptake value at the sustained plateau |
| `can_grow_actual` | 1 when plateau growth exceeds `growth_threshold`; otherwise 0 |
| `pauc_g_actual` | Normalized partial area under the growth curve up to `pauc_u_max` |

## Model feature names

| Prefix or name | Count | Description |
|---|---:|---|
| `aa_pca_0`–`aa_pca_10` | 11 | PCA scores from 20 amino-acid composition frequencies |
| `esm_0`–`esm_10` | 11 | PCA scores from 320-dimensional ESM-2 strain representations |
| `MolWt`, `LogP`, `TPSA`, `HBD`, `HBA`, `RotBonds`, `RingCount`, `HeavyAtom`, `FractionCSP3` | 9 | RDKit molecular descriptors |
| `fp_0`–`fp_12` | 13 | PCA scores from 2,048-bit Morgan fingerprints |
