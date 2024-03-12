# How was GulLiver Developed?

The end goal of the project was to analyze Sox9 positive structures surrounding portal veins.
Optionally, we would like to quantify area coverage and quantity of central and portal veins as well.
Samples consist of whole immunofluorescence slides of livers with stainings corresponding to DAPI, Sox9, elastin and GS.
With this in consideration, we worked in segmenting:

- [livers from DAPI staining;](#liver-segmentation)
- Sox9 structures from Sox9 staining;
- lumina from autofluorescence and combination of channels;
- elastin and GS from their respective channels.

And then classifying
- lumina according to other stainings
- Sox9 structures with machine learning based classification

Finally, the whole pipeline was combined into single functions that are run with a command line interface (CLI).

## Liver Segmentation

Uses DAPI channel to segment liver. Described in [liver_segmentation](https://github.com/acorbat/gulliver/blob/main/notebooks/liver_segmentation.ipynb) notebook.
