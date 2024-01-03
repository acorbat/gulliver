from argparse import ArgumentParser
import logging
from pathlib import Path

from napari_czifile2.io import CZISceneFile
import pyclesperanto_prototype as cle
from tqdm import tqdm

from .io import get_image, add_labels, get_channel_from_zarr
from .segmenter import (
    find_structures,
    segment_liver,
    clean_segmentations,
    add_veins,
)

from . import __version__

cle.select_device("RTX")

logger = logging.getLogger(__name__)


def main():
    logger.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        choices=["segment", "quantify"],
        help="Command to run. You can choose between segment "
        + "and quantify.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to file or folder to process",
    )
    parser.add_argument(
        "--scene",
        type=int,
        help="Scene to process if analyze a single file was " + "chosen",
    )
    parser.add_argument(
        "--chunk-multiplier",
        type=int,
        default=7,
        help="multiplier to determine maximum size of chunk "
        + "to send to GPU. (chunk * (1024, 1024))",
    )
    args = parser.parse_args()

    is_folder = args.path.is_dir()

    if args.command == "segment":
        if is_folder:
            segment_folder(
                folderpath=args.path,
                chunk_multiplier=args.chunk_multiplier,
            )
        else:
            segment_file(
                filepath=args.path,
                scene=args.scene,
                chunk_multiplier=args.chunk_multiplier,
            )
    elif args.command == "quantify":
        raise NotImplementedError("Not implemented yet")


def segment_file(
    filepath: Path,
    scene: int,
    chunk_multiplier: int = 7,
) -> None:
    """Segments an image file and saves it into ome-zarr format."""
    logger.info("Loading image %s" % str(filepath))
    image = get_image(filepath, scene)

    if "labels" in list(image.group_keys()):
        logger.info("Already contains a labels and cannot overwrite")
        return

    logger.info("Looking for structures in Sox9 staining")
    segmentations = find_structures(
        sox9_channel=get_channel_from_zarr(image, "Sox9"),
        gs_channel=get_channel_from_zarr(image, "GS"),
        elastin_channel=get_channel_from_zarr(image, "elastin"),
        chunk_shape=(chunk_multiplier * 1024, chunk_multiplier * 1024),
    )

    logger.info("Looking for liver pieces")
    liver = segmentations.create_group("liver")
    liver.create_dataset(
        "labels", data=segment_liver(get_channel_from_zarr(image, "DAPI"))
    )

    logger.info("Cleaning up segmentations")
    clean_segmentations(segmentations)

    logger.info("Adding vein labeling")
    add_veins(segmentations)

    logger.info("Adding labels to zarr")
    for label_name, label_image in segmentations.items():
        add_labels(image, label_image["labels"][:], label_name)


def segment_folder(
    folderpath: Path,
    chunk_multiplier: int = 7,
) -> None:
    """Runs segment file no a whole folder and subfolders."""
    filepaths = list(folderpath.rglob("*.czi"))

    logger.info(f"{len(filepaths)} images were found.")
    for filepath in tqdm(filepaths):
        number_of_scenes = CZISceneFile(filepath, 0).get_num_scenes(filepath)

        logger.info(
            f"Segmenting {number_of_scenes} scenes" + "from image {filepath}"
        )
        for scene in range(number_of_scenes):
            logger.info(f"Segmenting scene number {scene}")
            segment_file(
                filepath=filepath,
                scene=scene,
                chunk_multiplier=chunk_multiplier,
            )


if __name__ == "__main__":
    main()
