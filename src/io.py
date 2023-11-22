from pathlib import Path
from napari_czifile2.io import CZISceneFile
import numpy as np


def get_image(path: Path, scene: int = 0) -> np.ndarray:
    """Gets the selected scene from the CZI File as an array."""
    file = CZISceneFile(path, scene)
    img = np.squeeze(file.asarray())
    return img
