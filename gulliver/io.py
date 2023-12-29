import logging
from pathlib import Path
from typing import Dict

import dask
from napari_czifile2.io import CZISceneFile
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from skimage.exposure import rescale_intensity
import zarr


logger = logging.getLogger(__name__)


def get_image(
    path: Path, scene: int = 0, remake: bool = False
) -> zarr.hierarchy.Group:
    """Gets the selected scene from the CZI File as an array and makes an
    OME Zarr, or loads it if it was already created."""
    file = CZISceneFile(path, scene)
    image = np.squeeze(file.asarray())

    scale = {"z": file.scale_z_um, "y": file.scale_y_um, "x": file.scale_x_um}

    savepath = path.with_suffix(".zarr")
    if savepath.exists():
        logger.warn("zarr image already exists")
        store = parse_url(savepath, mode="w").store
        return zarr.group(store=store)

    store = parse_url(savepath, mode="w").store
    root = zarr.group(store=store)
    root.attrs["omero"] = {
        "channels": [
            {
                "color": "FFFFFF",
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
            {
                "color": "00FFFF",
                "label": "elastin",
            },
        ]
    }

    write_image(
        image=np.stack([rescale_intensity(this_image) for this_image in image]),
        group=root,
        axes="cyx",
        storage_options=dict(chunks=(1, 2048, 2048)),
        metadata={"scale": scale},
    )
    return root


def get_channel_from_zarr(
    image: zarr.hierarchy.Group, channel: str
) -> Dict[str, dask.array.core.Array]:
    """Get's the requested channel with the highest resolution"""
    channels = [
        channel["label"] for channel in image.attrs["omero"]["channels"]
    ]
    channel_index = channels.index(channel)
    return image["0"][channel_index]


def get_dict_image(path: Path) -> Dict[str, dask.array.Array]:
    """Returns a dask array containing the different channels and highest
    resolution of an image to be processed"""
    location = parse_url(path)
    multiscale = location.root_attrs["multiscales"][0]
    channels = [
        channel["label"] for channel in location.root_attrs["omero"]["channels"]
    ]
    image = location.load(multiscale["datasets"][0]["path"])
    return {channel: this_image for channel, this_image in zip(channels, image)}


def add_labels(
    root: zarr.hierarchy.Group, label_image: np.ndarray, label_name: str
) -> None:
    """Add a labeled image to the OME Zarr object."""
    write_labels(
        labels=label_image,
        group=root,
        name=label_name,
        axes="yx",
    )


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
        storage_options=dict(chunks=(1, 2048, 2048)),
        metadata=original_image.attrs["scale"],
    )

    for label_name, label_image in labeled_images.items():
        write_labels(
            labels=label_image["labels"][:],
            group=root,
            name=label_name,
            axes="yx",
        )
