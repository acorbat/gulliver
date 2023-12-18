from argparse import ArgumentParser
import logging
from pathlib import Path

from napari_czifile2.io import CZISceneFile
from tqdm import tqdm

from .io import get_image, save_img
from .segmenter import find_structures, segment_liver, clean_segmentations

from . import __version__

logger = logging.getLogger(__name__)


def segment_file(
    filepath: Path,
    scene: int,
    savepath: Path | None = None,
    chunk_multiplier: int = 7,
) -> None:
    """Segments an image file and saves it into ome-zarr format."""
    logger.info("Loading image %s" % str(filepath))
    image = get_image(filepath, scene)

    logger.info("Looking for structures in Sox9 staining")
    segmentations = find_structures(
        image["Sox9"]["image"],
        chunk_shape=(chunk_multiplier * 1024, chunk_multiplier * 1024),
    )

    logger.info("Looking for liver pieces")
    liver = segmentations.create_group("liver")
    liver.create_dataset("labels", data=segment_liver(image["DAPI"]["image"]))

    logger.info("Cleaning up segmentations")
    clean_segmentations(segmentations)

    if savepath is None:
        savepath = filepath.with_suffix(f"_scene_{scene}.zarr")
    logger.info("Saving image at %s" % str(savepath))
    save_img(savepath, image, segmentations)


def segment_folder(
    folderpath: Path,
    chunk_multiplier: int = 7,
) -> None:
    """Runs segment file no a whole folder and subfolders."""
    filepaths = folderpath.rglob("*.czi")

    logger.info(f"{len(filepaths)} images were found.")
    for filepath in tqdm(filepaths):
        number_of_scenes = CZISceneFile(filepath).get_num_scenes()

        logger.info(
            f"Segmenting {number_of_scenes} scenes" + "from image {filepath}"
        )
        for scene in range(number_of_scenes):
            logger.info(f"Segmenting scene number {scene}")
            savepath = filepath.with_suffix(f"_scene_{scene}.zarr")
            segment_file(
                filepath=filepath,
                scene=scene,
                savepath=savepath,
                chunk_multiplier=chunk_multiplier,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "path", type=Path, help="Path to file or folder to process"
    )
    parser.add_argument(
        "--chunk-multiplier",
        type=int,
        help="multiplier to determine maximum size of chunk "
        + "to send to GPU",
    )
