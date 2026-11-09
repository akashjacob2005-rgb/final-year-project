"""Brain parcellation used to turn voxels into graph nodes.

The atlas is configurable (GRAPH_ATLAS env var, or the `atlas` key a trained
model's graph_config.json carries) but a trained model is only valid with the
atlas it was trained on: node i must mean the same region at inference. The
backend therefore reads the atlas name from the artifact, never from a default.

Atlases are fetched by nilearn on first use and cached under NILEARN_DATA
(default ~/nilearn_data). Harvard-Oxford ships with FSL under a non-commercial
licence; it is downloaded, not redistributed in this repo.
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_ATLAS = os.getenv("GRAPH_ATLAS", "harvard_oxford_cort_2mm")

# name -> (nilearn fetcher, fetcher argument)
ATLASES = {
    "harvard_oxford_cort_2mm": ("fetch_atlas_harvard_oxford", "cort-maxprob-thr25-2mm"),
    "harvard_oxford_cort_1mm": ("fetch_atlas_harvard_oxford", "cort-maxprob-thr25-1mm"),
    "harvard_oxford_sub_2mm": ("fetch_atlas_harvard_oxford", "sub-maxprob-thr25-2mm"),
}


@lru_cache(maxsize=4)
def load_atlas(name: str = DEFAULT_ATLAS):
    """Return (labels_img, region_names) with background removed from the names.

    region_names[i] is the name of the label value i+1 in labels_img, which is
    the column order NiftiLabelsMasker produces.
    """
    import nibabel as nib
    from nilearn import datasets

    if name not in ATLASES:
        raise ValueError(f"unknown atlas {name!r}; choose one of {sorted(ATLASES)}")
    fetcher, arg = ATLASES[name]
    atlas = getattr(datasets, fetcher)(arg, data_dir=os.getenv("NILEARN_DATA"))
    img = atlas.maps if hasattr(atlas.maps, "shape") else nib.load(atlas.maps)
    names = [str(n) for n in atlas.labels]
    if names and names[0].lower() == "background":
        names = names[1:]
    return img, names
