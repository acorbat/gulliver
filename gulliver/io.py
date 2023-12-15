from pathlib import Path
from typing import Dict

from napari_czifile2.io import CZISceneFile
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
import zarr


def get_image(path: Path, scene: int = 0) -> np.ndarray:
    """Gets the selected scene from the CZI File as an array."""
    file = CZISceneFile(path, scene)
    img = np.squeeze(file.asarray())
    return img


def save_img(
    path: Path,
    original_image: np.ndarray,
    labeled_images: zarr.hierarchy.Group,
) -> None:
    """Saves the original image with the segmentation in multiscale OME Zarr
    format. Labeled images should be a dictionary where the keys are the names
    of the image."""
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    root.attrs["omero"] = {
        "channels": [
            {
                "color": "00FFFF",
                "label": "DAPI",
            },
            {
                "color": "FFFF00",
                "label": "Sox9",
            },
            {
                "color": "FF00FF",
                "label": "GS",
            },
        ]
    }
    write_image(
        image=original_image,
        group=root,
        axes="cyx",
        storage_options=dict(chunks=(1, 2046, 2046)),
    )

    for label_name, label_image in labeled_images.items():
        write_labels(
            labels=label_image["labels"][:],
            group=root,
            name=label_name,
            axes="yx",
        )
