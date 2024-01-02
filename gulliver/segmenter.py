import logging
import os
from typing import Callable, Tuple, List

from apoc import PixelClassifier
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
    disk,
    label,
    remove_small_objects,
    remove_small_holes,
)
import zarr

from .quantify import relate_structures

MODULE_DIRECTORY = os.path.dirname(__file__)


def segment_liver(nuclei_channel: np.ndarray) -> np.ndarray:
    """Segments liver area from nuclei stained images."""
    if nuclei_channel.ndim > 2:
        nuclei_channel = np.mean(nuclei_channel, axis=0)
    nuclei_channel = gaussian(nuclei_channel, 10)
    threshold = threshold_otsu(nuclei_channel)
    liver_masks = nuclei_channel > threshold
    liver_masks = binary_opening(
        liver_masks, footprint=disk(15, decomposition="sequence")
    )
    liver_masks = remove_small_holes(liver_masks, area_threshold=10**6)
    liver_masks = label(liver_masks)
    liver_masks = remove_small_objects(liver_masks, min_size=40000)
    return liver_masks


def clean_channel(
    sox9_channel: np.ndarray, elastin_channel: np.ndarray
) -> np.ndarray:
    """Remove one channel from the other to clean blledthrough"""
    clean_sox9_channel = np.clip(
        sox9_channel.astype(float) - elastin_channel.astype(float),
        a_min=0,
        a_max=np.inf,
    ).astype("uint16")
    return clean_sox9_channel


def chunk_and_process_2d_array(
    input_array: np.ndarray, chunk_shape: Tuple, processing_function: Callable
) -> np.ndarray:
    """Applies the processing_function to the input array by chunks of
    chunk_shape."""
    # split the input array into subarrays of the specified shape
    sub_arrays = [
        [
            [
                processing_function(
                    input_array[i : i + chunk_shape[0], j : j + chunk_shape[1]]
                )
                for j in range(0, input_array.shape[1], chunk_shape[1])
            ]
            for i in range(0, input_array.shape[0], chunk_shape[0])
        ]
    ]

    # stitch the subarrays back together into a single numpy array
    output_array = np.block(sub_arrays)
    return output_array[0]


def relabel_image(
    label_image: np.ndarray,
    new_labels: List,
    chunk_shape: Tuple = (5 * 1024, 5 * 1024),
) -> np.ndarray:
    """Relabels the labels in label_image with the new labels. It assumes the
    first label in the new_labels corresponds to label 1 and onwards."""
    annotation = [0] + new_labels

    def my_replacer(image):
        return cle.replace_intensities(image, annotation)

    predicted_image = chunk_and_process_2d_array(
        label_image, chunk_shape=chunk_shape, processing_function=my_replacer
    )
    predicted_image = predicted_image.astype(int)
    return predicted_image


def predict(image: np.ndarray, segmenter: PixelClassifier) -> np.ndarray:
    """Uses the trained pixel classifier to perform a semantic segmentation of
    the image."""
    logging.info("processing image of size ", image.shape)
    semantic_segmentation = segmenter.predict(image=image)
    semantic_segmentation = cle.pull(semantic_segmentation)
    return semantic_segmentation


def predict_sox9(image: np.ndarray) -> np.ndarray:
    """Performs Sox9 positive semantic segmentation on the image."""
    sox9_segmenter = PixelClassifier(
        opencl_filename=os.path.join(MODULE_DIRECTORY, "Sox9PixelClassifier.cl")
    )
    semantic_segmentation = predict(image=image, segmenter=sox9_segmenter)

    semantic_segmentation = binary_opening(semantic_segmentation > 1, disk(1))
    semantic_segmentation = binary_closing(semantic_segmentation, disk(4))
    semantic_segmentation = binary_opening(semantic_segmentation, disk(1))
    return semantic_segmentation


def predict_holes(image: np.ndarray) -> np.ndarray:
    """Performs semantic segmentation for holes and debris on the image. Returns
    a stack where the first image corresponds to holes and the second one to
    labeled debris."""
    hole_segmenter = PixelClassifier(
        opencl_filename=os.path.join(MODULE_DIRECTORY, "HolePixelClassifier.cl")
    )
    semantic_segmentation = predict(image=image, segmenter=hole_segmenter)
    return semantic_segmentation


def separate_holes_and_debris(
    hole_segmentation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates first semantic segmentation of holes into holes and debris"""
    holes = hole_segmentation == 2
    not_well_stained = hole_segmentation == 3

    not_well_stained = binary_closing(not_well_stained, footprint=disk(3))
    return holes, not_well_stained


def predict_gs_positive(image: np.ndarray) -> np.ndarray:
    """Performs semantic segmentation for GS positive structures on the
    image."""
    gs_segmenter = PixelClassifier(
        opencl_filename=os.path.join(MODULE_DIRECTORY, "GSPixelClassifier.cl")
    )
    semantic_segmentation = predict(image=image, segmenter=gs_segmenter)

    semantic_segmentation = binary_opening(semantic_segmentation > 1, disk(5))
    return semantic_segmentation


def predict_elastin_positive(image: np.ndarray) -> np.ndarray:
    """Performs semantic segmentation for elastin positive structures on the
    image."""
    elastin_segmenter = PixelClassifier(
        opencl_filename=os.path.join(
            MODULE_DIRECTORY,
            "ElastinPixelClassifier.cl",
        )
    )
    semantic_segmentation = predict(image=image, segmenter=elastin_segmenter)

    semantic_segmentation = binary_closing(semantic_segmentation > 1, disk(5))
    return semantic_segmentation


def find_structures(
    sox9_channel: np.ndarray,
    gs_channel: np.ndarray,
    elastin_channel: np.ndarray,
    chunk_shape: Tuple = (5 * 1024, 5 * 1024),
) -> zarr.Array:
    """Finds bile ducts, holes and other things in the image"""
    segmentations = zarr.group()
    logging.info("Performing semantic segmentation of Sox9+ cells")
    sox9_positive = chunk_and_process_2d_array(
        clean_channel(
            sox9_channel=sox9_channel,
            elastin_channel=elastin_channel,
        ),
        chunk_shape=chunk_shape,
        processing_function=predict_sox9,
    )

    logging.info("Performing semantic segmentation of holes and debris")
    holes = chunk_and_process_2d_array(
        sox9_channel, chunk_shape=chunk_shape, processing_function=predict_holes
    )
    holes, not_well_stained = separate_holes_and_debris(holes)

    logging.info("Segmenting GS channel")
    gs_positive = chunk_and_process_2d_array(
        gs_channel,
        chunk_shape=chunk_shape,
        processing_function=predict_gs_positive,
    )

    logging.info("Segmenting elastin channel")
    elastin_positive = chunk_and_process_2d_array(
        elastin_channel,
        chunk_shape=chunk_shape,
        processing_function=predict_elastin_positive,
    )

    logging.info("Cleaning Sox9+ cells")
    sox9_positive[np.logical_or(holes.astype(bool), not_well_stained > 0)] = 0

    logging.info("Labelling Sox9+ cells")
    sox9_positive = label(sox9_positive)
    sox9_positive = remove_small_objects(sox9_positive, min_size=70)

    s9 = segmentations.create_group("sox9_positive")
    s9 = s9.create_dataset("labels", data=sox9_positive)
    h = segmentations.create_group("holes")
    h = h.create_dataset("labels", data=holes)
    n = segmentations.create_group("not_well_stained")
    n = n.create_dataset("labels", data=not_well_stained)
    gs = segmentations.create_group("gs_positive")
    gs = gs.create_dataset("labels", data=gs_positive)
    el = segmentations.create_group("elastin_positive")
    el = el.create_dataset("labels", data=elastin_positive)

    return segmentations


def clean_segmentations(segmentations: zarr.hierarchy.Group) -> None:
    """Uses liver segmentation to remove objects outside the liver inside the
    segmentations object"""
    # First remove signal from outside livers
    liver_mask = segmentations["liver"]["labels"][:] < 1

    for group in segmentations.groups():
        if group[0] == "liver":
            continue

        group[1]["labels"].set_mask_selection(liver_mask, 0)


def find_vessel_regions(
    holes: np.ndarray, not_well_stained: np.ndarray
) -> np.ndarray:
    """Finds big regions that could be portal triads, portal veins or central
    veins. It discards huge regions that could be not well stained regions."""
    veins = remove_small_objects(label(holes), min_size=250)
    mesenchyma = remove_small_objects(label(not_well_stained), min_size=750)
    structures = np.logical_or(veins > 0, mesenchyma > 0)
    structures = np.logical_xor(
        remove_small_objects(label(structures), min_size=1000000), structures
    )
    structures = label(binary_dilation(structures, disk(15)))
    return structures


def find_borders(label: np.ndarray, size: int = 20):
    """Creates a labeled mask of the surrounding region of each label."""
    borders = cle.dilate_labels(label, radius=size) - cle.erode_labels(
        label, radius=15
    )
    return borders.get().astype(int)


def classify_vein(gs_intensity: float, elastin_intensity: float) -> int:
    """Classifies veins according to the ratio of pixels where GS signal and
    elastin signal in borders is positive.

    1 -> Portal vein
    2 -> Central vein
    3 -> Unclassified"""
    if gs_intensity > 0.18:
        return 2
    if elastin_intensity > 0.05:
        return 1
    else:
        return 3


def add_veins(segmentations: zarr.hierarchy.Group) -> None:
    """First finds regions corresponding to portal triads, portal veins or
    central veins and then classifies into central vein regions considering
    GS positive region and portal regions considering elastin positive staining.

    Returns a tuple of arrays: (portal veins, central veins)"""
    regions = find_vessel_regions(
        segmentations["holes"]["labels"],
        segmentations["not_well_stained"]["labels"],
    )
    border_regions = find_borders(regions)
    gs_table = relate_structures(
        border_regions, segmentations["gs_positive"]["labels"]
    )
    elastin_table = relate_structures(
        border_regions, segmentations["elastin_positive"]["labels"]
    )
    full_table = pd.merge(
        gs_table, elastin_table, on="label", suffixes=["_gs", "_elastin"]
    )
    full_table["vein"] = full_table.apply(
        lambda x: classify_vein(
            x["intensity_mean_gs"], x["intensity_mean_elastin"]
        ),
        axis=1,
    )

    # To do: several steps here are probably innefficient
    veins = relabel_image(regions, list(full_table["vein"].values))
    cv = segmentations.create_group("central_veins")
    cv = cv.create_dataset("labels", data=regions.copy())
    cv.set_mask_selection(veins != 2, 0)

    pv = segmentations.create_group("portal_veins")
    pv = pv.create_dataset("labels", data=regions.copy())
    pv.set_mask_selection(veins != 1, 0)


def find_portal_regions(portal_veins: np.ndarray, radius: int) -> np.ndarray:
    """Exapnds the portal veins to define a region to be analyzed"""
    regions = cle.dilate_labels(portal_veins, radius=radius)
    return regions
