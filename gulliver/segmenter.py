import logging
import os
from typing import Callable, Tuple, List

from apoc import PixelClassifier
import numpy as np
import pyclesperanto_prototype as cle
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    binary_opening,
    binary_closing,
    disk,
    label,
    remove_small_objects,
    remove_small_holes,
)
import zarr

MODULE_DIRECTORY = os.path.dirname(__file__)


def segment_liver(nuclei_channel: np.ndarray) -> np.ndarray:
    """Segments liver area from nuclei stained images."""
    nuclei_channel = gaussian(nuclei_channel, 20)
    threshold = threshold_otsu(nuclei_channel)
    liver_masks = nuclei_channel > threshold
    liver_masks = binary_opening(
        liver_masks, footprint=disk(30, decomposition="sequence")
    )
    liver_masks = remove_small_holes(liver_masks, area_threshold=10**8)
    liver_masks = label(liver_masks)
    liver_masks = remove_small_objects(liver_masks, min_size=10000)
    return liver_masks


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

    semantic_segmentation = binary_opening(semantic_segmentation > 1, disk(3))
    semantic_segmentation = binary_closing(semantic_segmentation, disk(15))
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

    not_well_stained = binary_closing(not_well_stained, footprint=disk(10))
    not_well_stained = remove_small_objects(
        label(not_well_stained),
        min_size=200,
    )
    return holes, not_well_stained


def predict_gs_positive(image: np.ndarray) -> np.ndarray:
    """Performs semantic segmentation for GS positive structures on the
    image."""
    gs_segmenter = PixelClassifier(
        opencl_filename=os.path.join(MODULE_DIRECTORY, "GSPixelClassifier.cl")
    )
    semantic_segmentation = predict(image=image, segmenter=gs_segmenter)

    semantic_segmentation = binary_closing(semantic_segmentation > 1, disk(10))
    semantic_segmentation = remove_small_objects(
        label(semantic_segmentation), min_size=9000
    )
    return semantic_segmentation


def find_structures(
    image: np.ndarray, chunk_shape: Tuple = (5 * 1024, 5 * 1024)
) -> zarr.Array:
    """Finds bile ducts, holes and other things in the image"""
    segmentations = zarr.group()
    logging.info("Performing semantic segmentation of Sox9+ cells")
    sox9_positive = chunk_and_process_2d_array(
        image, chunk_shape=chunk_shape, processing_function=predict_sox9
    )

    logging.info("Performing semantic segmentation of holes and debris")
    holes = chunk_and_process_2d_array(
        image, chunk_shape=chunk_shape, processing_function=predict_holes
    )
    holes, not_well_stained = separate_holes_and_debris(holes)

    logging.info("Cleaning Sox9+ cells")
    sox9_positive[np.logical_or(holes, not_well_stained > 1)] = 0

    logging.info("Labelling Sox9+ cells")
    sox9_positive = label(sox9_positive)

    s9 = segmentations.create_group("sox9_positive")
    s9 = s9.create_dataset("labels", data=sox9_positive)
    h = segmentations.create_group("holes")
    h = h.create_dataset("labels", data=holes)
    n = segmentations.create_group("not_well_stained")
    n = n.create_dataset("labels", data=not_well_stained)

    return segmentations