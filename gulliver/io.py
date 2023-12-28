from pathlib import Path
from typing import Dict

from napari_czifile2.io import CZISceneFile
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
import zarr


def get_image(path: Path, scene: int = 0) -> zarr.hierarchy.Group:
    """Gets the selected scene from the CZI File as an array."""
    file = CZISceneFile(path, scene)
    img = np.squeeze(file.asarray())

    scale = {"z": file.scale_z_um, "y": file.scale_y_um, "x": file.scale_x_um}

    root = zarr.group()
    root.attrs["scale"] = scale
    for this_img, channel in zip(img, ["DAPI", "Sox9", "GS", "elastin"]):
        this_group = root.create_group(channel)
        this_group.create_dataset("image", data=this_img)
    return root


def save_img(
    path: Path,
    original_image: zarr.hierarchy.Group,
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
        image=np.stack(
            [
                original_image[channel]["image"]
                for channel in ("DAPI", "Sox9", "GS")
            ]
        ),
        group=root,
        axes="cyx",
        storage_options=dict(chunks=(1, 2046, 2046)),
        metadata=original_image.attrs["scale"],
    )

    for label_name, label_image in labeled_images.items():
        write_labels(
            labels=label_image["labels"][:],
            group=root,
            name=label_name,
            axes="yx",
        )
