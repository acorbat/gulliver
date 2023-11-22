import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_opening, disk, label, remove_small_objects, remove_small_holes
from skimage.measure import regionprops


def segment_liver(nuclei_channel: np.ndarray) -> np.ndarray:
    """Segments liver area from nuclei stained images."""
    nuclei_channel = gaussian(nuclei_channel, 20)
    threshold = threshold_otsu(nuclei_channel)
    liver_masks = nuclei_channel > threshold
    liver_masks = binary_opening(liver_masks, footprint=disk(30, decomposition="sequence"))
    liver_masks = remove_small_holes(liver_masks, area_threshold=10**8)
    liver_masks = label(liver_masks)
    liver_masks = remove_small_objects(liver_masks, min_size=10000)
    return liver_masks