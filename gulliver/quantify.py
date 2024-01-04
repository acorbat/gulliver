import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table

from .classifier import (
    predict_bile_duct_classes_from_table,
    parse_bile_duct_classes,
)


def get_properties(
    labels: np.ndarray,
    scale: float | None = None,
    annotation: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generates a properties table including the annotated values from a
    labeled image."""
    properties_list = [
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
        "label",
        "moments_hu",
        "perimeter",
        "solidity",
    ]
    if annotation is not None:
        properties_list += ["intensity_max", "intensity_min", "intensity_mean"]

    properties = regionprops_table(
        labels,
        intensity_image=annotation,
        properties=properties_list,
    )

    if scale is not None:
        area_properties = [
            "area",
            "area_convex",
            "area_filled",
        ]
        for column in area_properties:
            properties[column] = properties[column] * scale * scale

        distance_properties = [
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "feret_diameter_max",
        ]
        for column in distance_properties:
            properties[column] = properties[column] * scale

    return pd.DataFrame.from_dict(properties)


def get_sox9_properties(
    sox9_positive: np.ndarray,
    lumen: np.ndarray,
    scale: float,
    annotation: np.ndarray | None = None,
) -> pd.DataFrame:
    """Builds the table with all Sox9+ descriptors, and distance to lumen. Can
    take an annotation image to add annotated information."""
    properties = get_properties(
        sox9_positive, scale=scale, annotation=annotation
    )
    lumen_distance_table = find_distances(
        sox9_positive, lumen > 0, suffix="lumen", scale=scale
    )
    properties.set_index("label", verify_integrity=True, inplace=True)
    properties = properties.join(lumen_distance_table.set_index("label"))
    return properties


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


def get_vein_properties(
    vein: np.ndarray,
    other_vein: np.ndarray,
    scale: float,
) -> pd.DataFrame:
    """Get a table with the properties describing all veins in a labeled
    image."""
    properties = get_properties(vein, scale=scale)
    properties = properties[["label", "area", "eccentricity"]]
    properties = properties.merge(
        find_distances(labels=vein, relative_to_mask=other_vein, scale=scale),
        on="label",
    )
    return properties.set_index("label", verify_integrity=True)


def find_distances(
    labels: np.ndarray,
    relative_to_mask: np.ndarray,
    suffix: str | None = None,
    scale: float | None = None,
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

    if scale is not None:
        for column in table.columns:
            if "distance" in column:
                table[column] = table[column] * scale
    return table


def find_portal_regions(portal_veins: np.ndarray, radius: int) -> np.ndarray:
    """Exapnds the portal veins to define a region to be analyzed"""
    regions = cle.dilate_labels(portal_veins, radius=radius)
    return regions.get()


def get_portal_region_description(
    sox9_positive: np.ndarray,
    lumen: np.ndarray,
    portal_veins: np.ndarray,
    scale: float,
):
    """Generates a table describing all the Sox9+ structures, to which class and
    portal vein they correspond to."""
    properties = get_sox9_properties(
        sox9_positive=sox9_positive, lumen=lumen, scale=scale
    )
    properties["class"] = predict_bile_duct_classes_from_table(properties)

    portal_regions = find_portal_regions(portal_veins, 50)
    portal_dependency = relate_structures(
        labels=sox9_positive, related_image=portal_regions
    )
    properties = properties.merge(
        portal_dependency[["label", "intensity_max"]].rename(
            columns={"intensity_max": "portal_vein"}
        ),
        on="label",
    )

    properties["class"] = properties["class"].apply(parse_bile_duct_classes)
    properties = properties.set_index(
        "label", verify_integrity=True
    ).sort_values("portal_vein", ascending=False)[
        ["area", "class", "portal_vein"]
    ]
    return properties
