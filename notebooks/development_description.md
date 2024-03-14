# How was GulLiver Developed?

The end goal of the project was to analyze Sox9 positive structures surrounding portal veins.
Optionally, we would like to quantify area coverage and quantity of central and portal veins as well.
Samples consist of whole immunofluorescence slides of livers with stainings corresponding to DAPI, Sox9, elastin and GS.

With this in consideration, we worked in segmenting:

- [livers](#liver-segmentation) from DAPI staining;
- [Sox9 structures](#sox9-structure-segmentation) by removing autofluorescence from Sox9+ channel and training a Pixel Classifier;
- [lumina and not well stained areas](#lumina-segmentation) from autofluorescence and combination of channels by training another Pixel Classifier;
- [elastin and GS](#elastin-and-gs-segmentation) from their respective channels and more Pixel Classifiers.

And then classifying:

- [lumina](#vein-classification) according to elastin and GS segmentation.
- [Sox9 structures](#bile-duct-classification) with machine learning based classification of morphometric parameters and distance to nearest lumina.

Once segmentation and classification has been done, there are steps pertaining the quantification of the different objects:

- [vein quantification](#vein-quantification): distance between vein types, vein cross section areas and total area covered.
- [portal region quantification](#portal-region-quantification): number of Sox9+ structures of each type surrounding each portal vein.

Finally, the [whole pipeline](#whole-pipeline) was combined into single functions that are run with a command line interface (CLI).


## Segmentation

### Liver Segmentation

Uses DAPI channel to segment liver. Described in [liver_segmentation](https://github.com/acorbat/gulliver/blob/main/notebooks/liver_segmentation.ipynb) notebook.


### Sox9 structure segmentation

Combinations of Sox9, DAPI, elastin and GS channels were used to clean up Sox9 channel and then segment Sox9 structures as shown in [bile duct segmentation notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_segmentation.ipynb).


### Lumina segmentation

Combinations of Sox9, DAPI, elastin and GS channels were used again to increase contrast between lumina and tissue (making use of autofluorescence) to segment out lumina and not well stained regions as also shown in [bile duct segmentation notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_segmentation.ipynb).


### Elastin and GS segmentation

Elastin and GS signals were identfied from their respective channels as shown in [vein classification notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/vein_classification.ipynb).


## Classification

### Vein Classification

Information from segmented lumina and GS and Elastin signals were combined to identify veins and classify them into portal and central, as shown in [vein classification notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/vein_classification.ipynb).


### Bile duct classification

With the information from the lumina and morphometrics of the Sox9+ structures, we train a classifier to distinguish between each Sox9+ structure class as shown in [bile duct classification notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_classification.ipynb).


## Quantification

### Vein quantification

Quantification of cross-section area, total area covered   and distance between portal and central veins are addressed in [intervein distance notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/intervein_distance.ipynb).


### Portal region quantification

Portal regions are defined as the area surrounding the portal veins, and every Sox9+ structure in this area is then quantified as shown in [portal region description notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/portal_region_description.ipynb).


## Whole pipeline

In [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/whole_pipeline.ipynb) a complete run for a single file is shown.
These functions are later wrapped into commands that can be run over files or folders from the command line interface.
