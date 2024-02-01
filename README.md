# GulLiver

An image analysis package for whole slide liver immunostainings.


# Installation


# Usage

Once installed in a specific environment, you should always first activate the environment and then run gulliver.

```
conda activate gulliver
gulliver
```

## Segmentation

The first module is in charge of processing scenes from CZI files and will output an OME Zarr file that contains all three channels, and labeled images corresponding to Sox9+ structures, liver regions, vessels or ducts and mesenchyma or not well stained areas.

You can either process a single scene from a single file.

```
gulliver segment path/to/file/czi --scene 0
```

Or you can point to a folder and gulliver will process every scene from every file.

```
gulliver segment path/to/folder
```

Another optional parameters is ```--chunk-multiplier``` which will select how big is the chunk to be sent to GPU in number of (1024, 1024) chunks. The default is 7 so $(7*1024, 7*1024)$ chunks will be used.


## Quantification

Once images are segmented, everything is saved in OME-Zarr files. The quantify command receives a folder where the zarr images reside.

```
gulliver quantify path/to/folder
```

It will generate six Excel files where each tab will correspond to each image found in the folder.

```portal_regions.xlsx``` contains all the Sox9+ structures detected in portal regions and information about their area (in $\mu ^2$), their class, to which portal vein they belong to and their labels in the segmented zarr image (see [Visualization](#visualization)).

```sox9_positive.xlsx``` contains all the Sox9+ structures detected, even if they are far away from portal veins, and information about their area (in $\mu ^2$), their class, to which portal vein they belong to (0 if the belong to no portal region) and their labels in the segmented zarr image (see [Visualization](#visualization)).

```portal_veins.xlsx``` contains all portal veins detected and information about their area (in $\mu ^2$), their distance to the nearest central vein (in $\mu$) and their labels in the segmented zarr image (see [Visualization](#visualization)).

```central_veins.xlsx``` contains all central veins detected and information about their area (in $\mu ^2$), their distance to the nearest portal vein  (in $\mu$) and their labels in the segmented zarr image (see [Visualization](#visualization)).

```lumen_regions.xlsx``` contains all lumens detected irrespective of size and information about their area (in $\mu ^2$), their morphometric properties and their labels in the segmented zarr image (see [Visualization](#visualization)).

```liver_parts.xlsx``` contains all liver regions detected and information about their area (in $\mu ^2$), their morphometric properties and their labels in the segmented zarr image (see [Visualization](#visualization)).


# Visualization

To visualize results you should first install [napari](https://napari.org/stable/) following [these instructions](https://napari.org/stable/tutorials/fundamentals/installation.html#install-as-python-package-recommended).

With napari installed, you can start it by activating the environment and running it.

```
conda activate napari-env
napari
```

Once opened, you should open the menu for ```Plugins\Install Plugins```. There you should search for [napari-ome-zarr](https://github.com/ome/napari-ome-zarr) and install it (and restart napari).

Now that napari with the napari-ome-zarr is running, you can drag and drop ```.zarr``` files (they look like folders) into napari. If a prompt appears asking what reader to use, choose napari-ome-zarr.


# Rationale

During Gulliver's first part of his trips, he arrives to Lilliput and finds himself a prisoner of a race of tiny people.
Whole slide imaging looks similar as one is prisoner of an image filled with small details and not too much memory to load everything at the same time.
Additionally, Gulliver can be separated into Gul and Liver, Gul in swedish means yellow and some of the diseases this pipeline was developed for end up in yellow livers.
