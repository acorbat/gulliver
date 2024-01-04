import logging
import os

import apoc
import numpy as np
import pandas as pd


MODULE_DIRECTORY = os.path.dirname(__file__)


def predict_bile_duct_classes_from_table(
    properties: pd.DataFrame,
) -> np.ndarray:
    """Predicts to which class of bile duct each row of properties coresponds
    to."""
    logging.info("Classifying table")
    cl_filename = os.path.join(
        MODULE_DIRECTORY,
        "Sox9ObjectClassifier.model.cl",
    )
    classifier = apoc.TableRowClassifier(cl_filename)

    properties_for_classification = [
        "area",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "perimeter",
        "solidity",
        "lumen_distance_max",
        "lumen_distance_min",
        "lumen_distance_mean",
    ]

    properties_for_classification += [
        "moments_hu-%s" % str(num) for num in range(7)
    ]

    prediction = classifier.predict(properties[properties_for_classification])
    return prediction


def parse_bile_duct_classes(class_number: int) -> str:
    """Parses bile duct class number into class name"""
    class_dictionary = {
        1: "Well Formed",
        2: "Functional",
        3: "Open Circle",
        4: "Cluster",
        5: "Single Cell",
    }
    return class_dictionary[class_number]
