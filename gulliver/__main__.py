from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Tuple

from napari_czifile2.io import CZISceneFile
import pandas as pd
import pyclesperanto_prototype as cle
from tqdm import tqdm

from .quantify import (
    get_vein_properties,
    get_portal_region_description,
    get_properties,
)
from .io import (
    get_image,
    add_labels,
    get_channel_from_zarr,
    get_labels_from_zarr,
)
from .segmenter import (
    find_structures,
    segment_liver,
    clean_segmentations,
    add_veins,
    relabel_by_values,
)
from .classifier import parse_to_bile_duct_numbers

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
        quantify_folder(folderpath=args.path)


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
        dapi_channel=get_channel_from_zarr(image, "DAPI"),
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
        scene = 0
        logger.info(f"Segmenting scene number {scene}")
        segment_file(
            filepath=filepath,
            scene=scene,
            chunk_multiplier=chunk_multiplier,
        )
        ### This code is for scenes
        # number_of_scenes = CZISceneFile(filepath, 0).get_num_scenes(filepath)

        # logger.info(
        #     f"Segmenting {number_of_scenes} scenes" + "from image {filepath}"
        # )
        # for scene in range(number_of_scenes):
        #     logger.info(f"Segmenting scene number {scene}")
        #     segment_file(
        #         filepath=filepath,
        #         scene=scene,
        #         chunk_multiplier=chunk_multiplier,
        #     )


def quantify_file(path: Path) -> Tuple[pd.DataFrame]:
    """Quantifies portal regions, portal veins, central veins and lumen in a
    single file and returns all four tables"""
    image = get_image(path, 0)
    x_scale = image.attrs["multiscales"][0]["metadata"]["scale"]["x"]

    sox9_positive = get_labels_from_zarr(image, "sox9_positive")
    lumen = get_labels_from_zarr(image, "lumen")
    portal_veins = get_labels_from_zarr(image, "portal_veins")
    central_veins = get_labels_from_zarr(image, "central_veins")

    logger.info("Quantifying Portal Vein properties")
    portal_vein_table = get_vein_properties(
        vein=portal_veins, other_vein=central_veins[:] > 1, scale=x_scale
    )

    logger.info("Quantifying Central Vein properties")
    central_vein_table = get_vein_properties(
        vein=central_veins, other_vein=portal_veins[:] > 1, scale=x_scale
    )

    logger.info("Quantifying Portal Region properties")
    portal_region_table = get_portal_region_description(
        sox9_positive=sox9_positive,
        lumen=lumen[:],
        portal_veins=portal_veins[:],
        scale=x_scale,
    )

    logger.info("Relabeling and saving bile duct classes")
    bile_duct_numbers = portal_region_table["class"].apply(
        parse_to_bile_duct_numbers
    )
    bile_duct_class_labeled = relabel_by_values(
        sox9_positive,
        labels=bile_duct_numbers.index.values,
        values=bile_duct_numbers.values,
    )
    add_labels(
        image,
        label_image=bile_duct_class_labeled,
        label_name="bile_duct_classes",
    )

    logger.info("Quantifying Lumen properties")
    lumen_table = get_properties(lumen, scale=x_scale)

    logger.info("Quantifying Liver properties")
    liver_labels = get_labels_from_zarr(image, "liver")
    liver_table = get_properties(liver_labels, scale=x_scale)

    return (
        portal_region_table,
        portal_vein_table,
        central_vein_table,
        lumen_table,
        liver_table,
    )


def quantify_folder(folderpath: Path) -> None:
    """Quantifies every zarr file ina folder and adds the result to different
    tabs in three Excel files"""
    filepaths = list(folderpath.rglob("*.zarr"))

    logger.info(f"{len(filepaths)} images were found.")
    with pd.ExcelWriter(
        folderpath / "sox9_positive.xlsx"
    ) as sox9_writer, pd.ExcelWriter(
        folderpath / "portal_veins.xlsx"
    ) as portal_vein_writer, pd.ExcelWriter(
        folderpath / "central_veins.xlsx"
    ) as central_vein_writer, pd.ExcelWriter(
        folderpath / "portal_regions.xlsx"
    ) as portal_region_writer, pd.ExcelWriter(
        folderpath / "lumen_regions.xlsx"
    ) as lumen_writer, pd.ExcelWriter(
        folderpath / "liver_parts.xlsx"
    ) as liver_writer:
        for filepath in tqdm(filepaths):
            logger.info(f"Quantifying file {filepath.stem}")
            (
                portal_region_table,
                portal_vein_table,
                central_vein_table,
                lumen_table,
                liver_table,
            ) = quantify_file(path=filepath)

            portal_region_table.to_excel(sox9_writer, sheet_name=filepath.stem)
            portal_vein_table.to_excel(
                portal_vein_writer, sheet_name=filepath.stem
            )
            central_vein_table.to_excel(
                central_vein_writer, sheet_name=filepath.stem
            )
            portal_region_table.query("portal_vein > 0").to_excel(
                portal_region_writer, sheet_name=filepath.stem
            )
            lumen_table.to_excel(lumen_writer, sheet_name=filepath.stem)
            liver_table.to_excel(liver_writer, sheet_name=filepath.stem)


if __name__ == "__main__":
    main()
