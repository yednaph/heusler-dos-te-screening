#!/usr/bin/env python3
"""
dedup_twins.py -- count distinct crystals behind the computed and certified entries (paper decision 2026-10-08:
report ~80 distinct crystals, each confirmed by two DXMag entries, not 155 discoveries).

DXMag often lists one composition twice: the lowest-energy entry (training set, has a DFT DOS) and a second
configuration (prediction set). Two entries are counted as the SAME crystal ("twins") when they share composition,
family and space group and their DXMag energies differ by < 5 meV/atom and volumes by < 1 %.

INPUT   results/screening/ALL_DFT_3625_materials_DXMAG_style.xlsx (Materials: DXMag energy, cell, space group),
        results/paper_analysis/paper_dataset.pkl (certified flags)
OUTPUT  results/paper_analysis/twins.xlsx, twins.txt
"""
import os as _os_htr  # release v1: project paths come from the environment
_HTR = _os_htr.environ.get('HEUSLER_TE_ROOT', _os_htr.path.abspath(_os_htr.path.join(_os_htr.path.dirname(__file__), '../..')))
_HTE = _os_htr.environ.get('HEUSLER_TE_EXPORT', _os_htr.path.join(_HTR, 'DOS_EXPORT'))
import json
import numpy as np
import pandas as pd

P = _HTR
OUT = f"{P}/results/paper_analysis"
DE_MAX, DV_MAX = 0.005, 0.01                                              # twin tolerances: eV/atom, relative volume


def cell_volume(cell):
    """Volume (A^3) from the DXMag 'cell' column (3x3 lattice as a string or list)."""
    if not isinstance(cell, str):
        return np.nan                                                     # cell not recorded for this row
    a = np.array(json.loads(cell), float).reshape(3, 3)                   # JSON list-of-lists, parsed safely
    return abs(np.linalg.det(a))


def main():
    m = pd.read_excel(f"{P}/results/screening/ALL_DFT_3625_materials_DXMAG_style.xlsx", "Materials",
                      usecols=["UUID", "composition", "RUN_data_set", "RUN_family", "space group symbol",
                               "energy (eV/atom)", "cell", "number of atoms in the cell"])
    d = pd.read_pickle(f"{OUT}/paper_dataset.pkl")[["UUID", "certified", "winner"]].drop_duplicates("UUID")
    m = m.drop_duplicates("UUID").merge(d, on="UUID", how="left")
    m["vol_per_atom"] = m.cell.map(cell_volume) / m["number of atoms in the cell"]
    pr = m[m.RUN_data_set == "prediction"]; tr = m[m.RUN_data_set == "training"]
    j = pr.merge(tr, on=["composition", "RUN_family", "space group symbol"], suffixes=("_P", "_T"))   # candidate pairs
    j["dE_meV"] = (j["energy (eV/atom)_P"] - j["energy (eV/atom)_T"]).abs() * 1000
    j["dV_rel"] = (j.vol_per_atom_P - j.vol_per_atom_T).abs() / j.vol_per_atom_T
    j["twin"] = (j.dE_meV < DE_MAX * 1000) & (j.dV_rel < DV_MAX)
    lines = []
    for lab, sel in (("all computed", lambda x: True), ("certified", lambda x: x)):
        cp = m[m.certified == True] if lab == "certified" else m
        jj = j[j.certified_P == True] if lab == "certified" else j
        tw = jj[jj.twin]
        twinned_P = set(tw.UUID_P)
        n_entries = len(cp)
        # distinct crystals = connected groups of the entries in this set, linked by twin edges (union-find)
        parent = {u: u for u in cp.UUID.dropna()}                         # NaN != NaN would never find its root
        def root(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for a_, b_ in zip(tw.UUID_P, tw.UUID_T):
            if a_ in parent and b_ in parent:                             # both twins are in this set
                parent[root(a_)] = root(b_)
        n_crystals = len({root(u) for u in parent})
        lines.append(f"{lab}: entries {n_entries} -> distinct crystals ~{n_crystals} "
                     f"(prediction entries with a training twin: {len(twinned_P & set(cp.UUID))}; "
                     f"max dE among twins {tw.dE_meV.max():.2f} meV/atom, max dV {100*tw.dV_rel.max():.2f} %)")
    cert_comp = m[m.certified == True].groupby("composition").RUN_data_set.agg(lambda s: tuple(sorted(set(s))))
    lines.append("certified compounds by sets: " + str(cert_comp.value_counts().to_dict()))
    open(f"{OUT}/twins.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    j.sort_values(["certified_P", "composition"], ascending=[False, True]).to_excel(f"{OUT}/twins.xlsx", index=False)


if __name__ == "__main__":
    main()
