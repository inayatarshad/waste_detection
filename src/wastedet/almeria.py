"""Almeria labelled windows (from the labelling sheet) and mixing with AerialWaste.

Fine-tuning on Almeria uses BOTH sets: Almeria teaches the local look (greenhouse plastic is not
waste; greenhouse plastic dumped in a rambla is), AerialWaste keeps the general knowledge of what
dumps look like, so the model does not forget it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .aerialwaste import Record, WasteSceneDataset
from .scan import read_sheet


def _sheet_rows(sheet_paths):
    """One sheet or several (one per scanned area); a window labelled in two sheets is kept once."""
    if isinstance(sheet_paths, (str, Path)):
        sheet_paths = [sheet_paths]
    seen = set()
    for sp in sheet_paths:
        for r in read_sheet(sp):
            if r["file"] not in seen:
                seen.add(r["file"]); yield r


def records_from_sheet(sheet_path, n_types: int, plastic_index: int = 0) -> dict[str, list[Record]]:
    """Rows -> Records split by the sheet's spatial 'split' column (2 km blocks).
    Only the plastic type is labelled in Almeria; the other types are marked unknown.
    `sheet_path` may be one sheet or a list of sheets."""
    out = {"train": [], "val": [], "test": []}
    for r in _sheet_rows(sheet_path):
        types = np.zeros(n_types, np.float32)
        mask = np.zeros(n_types, np.float32)
        if r["plastic"] is not None:
            types[plastic_index] = float(r["plastic"]); mask[plastic_index] = 1.0
        if r["waste"] == 0:           # no waste here -> no waste type of any kind
            mask[:] = 1.0; types[:] = 0.0
        rec = Record(file_name=r["file"], waste=r["waste"], fine=bool(mask.any()), types=types,
                     source="almeria", type_mask=mask,
                     meta={"x": float(r["x"]), "y": float(r["y"]), "why": r.get("why_sampled", "")})
        out[r.get("split", "train") or "train"].append(rec)
    return out


class MixedDataset(Dataset):
    """Concatenation that still exposes `.records` and forwards `.epoch` (used by training)."""

    def __init__(self, parts: list[WasteSceneDataset], repeats: list[int] | None = None):
        self.parts = parts
        self.repeats = repeats or [1] * len(parts)
        self.index = [(p, i) for p, ds, rep in zip(range(len(parts)), parts, self.repeats)
                      for _ in range(rep) for i in range(len(ds))]
        self._epoch = 0

    @property
    def records(self):
        return [self.parts[p].records[i] for p, i in self.index]

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, e):
        self._epoch = e
        for ds in self.parts:
            ds.epoch = e

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        p, i = self.index[k]
        return self.parts[p][i]


def almeria_datasets(sheet_path, crops_dir, n_types, size=640, plastic_index=0):
    recs = records_from_sheet(sheet_path, n_types, plastic_index)
    return {k: WasteSceneDataset(v, Path(crops_dir), size=size, train=(k == "train")) for k, v in recs.items()}
