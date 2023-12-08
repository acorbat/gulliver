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
    prediction = classifier.predict(properties)
    return prediction
