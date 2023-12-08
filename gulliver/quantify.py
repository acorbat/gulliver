import numpy as np
import pandas as pd
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
