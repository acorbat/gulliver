import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table


def get_properties(
    labels: np.ndarray, annotation: np.ndarray | None = None
) -> pd.DataFrame:
    """Generates a properties table including the annotated values from a
    labeled image."""
    properties = regionprops_table(
        labels,
        intensity_image=annotation,
        properties=(
            "area",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "intensity_max",
            "intensity_min",
            "label",
            "moments_hu",
            "perimeter",
            "solidity",
        ),
    )
    return pd.DataFrame.from_dict(properties)


def relate_structures(
    labels: np.ndarray, related_image: np.ndarray
) -> pd.DataFrame:
    """Generates a properties table containing the maximum, minimum and mean
    intensities of the related_image in the labeled image.

    This function can be used for example:
    - related labeled can be a manually painted image and the maximum will yield
    the value inside the label
    - related label can be a distance image and the minimum yields the minimum
    distance to the structure.
    - related label can be some mask and the mean yields the percentage of
    positive pixels"""
    properties = regionprops_table(
        labels,
        intensity_image=related_image,
        properties=(
            "area",
            "intensity_max",
            "intensity_min",
            "intensity_mean",
            "label",
        ),
    )
    return pd.DataFrame.from_dict(properties)


def find_distances(
    labels: np.ndarray,
    relative_to_mask: np.ndarray,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Calculates distance estimations between each labeled object and the
    mask provided."""
    distance_map = distance_transform_edt(~relative_to_mask.astype(bool))
    table = relate_structures(labels=labels, related_image=distance_map)
    table.drop(columns="area", inplace=True)
    table.rename(
        columns={
            column_name: column_name.replace("intensity", "distance")
            for column_name in table.columns
        },
        inplace=True,
    )
    if suffix is not None:
        table.rename(
            columns={
                column_name: "_".join([suffix, column_name])
                for column_name in table.columns
                if "distance" in column_name
            },
            inplace=True,
        )
    return table
