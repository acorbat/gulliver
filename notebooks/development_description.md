# How was GulLiver Developed?

The end goal of the project was to analyze Sox9 positive structures surrounding portal veins.
Optionally, we would like to quantify area coverage and quantity of central and portal veins as well.
Samples consist of whole immunofluorescence slides of livers with stainings corresponding to DAPI, Sox9, elastin and GS.
With this in consideration, we worked in segmenting:

- [livers from DAPI staining;](#liver-segmentation)
- [Sox9 structures](#sox9-structure-segmentation) from Sox9 staining;
- [lumina](#lumina-segmentation) from autofluorescence and combination of channels;
- [elastin and GS](#elastin-and-gs-segmentation) from their respective channels.

And then classifying
- [lumina](#vein-classification) according to other stainings
- [Sox9 structures](#bile-duct-classification) with machine learning based classification

Finally, the whole pipeline was combined into single functions that are run with a command line interface (CLI).


## Liver Segmentation

Uses DAPI channel to segment liver. Described in [liver_segmentation](https://github.com/acorbat/gulliver/blob/main/notebooks/liver_segmentation.ipynb) notebook.


## Sox9 structure segmentation

Combinations of Sox9, DAPI, elastin and GS channels were used to clean up Sox9 channel and then segment Sox9 structures as shown in [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_segmentation.ipynb).


## Lumina segmentation

Combinations of Sox9, DAPI, elastin and GS channels were used again to increase contrast between lumina and tissue (making use of autofluorescence) as shown in [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_segmentation.ipynb).


## Elastin and GS segmentation

Elastin and GS signals were identfied from their respective channels as shown in [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/vein_classification.ipynb).


## Vein Classification

Information from segmented lumina and GS and Elastin signals were combined to identify veins and classify them into portal and central, as shown in [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/vein_classification.ipynb).


## Bile duct classification

With the information from the lumina and morphometrics of the Sox9+ structures, we train a classifier to distinguish between each bile duct class as shown in [this notebook](https://github.com/acorbat/gulliver/blob/main/notebooks/bile_duct_classification.ipynb)
