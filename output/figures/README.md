# Plots 
This folder is subject to changes over different runs, as new plots are routinely produced and overwritten. Experiment-specific plots produced by ``visuals.py`` are located in ``processed/``, while more informative visualizations produced by the main scripts are in ``general``.

## ``general`` plots
Here we provide annotations and commentary on the types of plots in ``general/``.

### Sample vs Mean Divergence Plots
These plots compare a provided sample against the mean image of a class. All images were taken from the _compounded, raw_ data and are on a scale from -1 to 1 (unless otherwise stated). To compute the divergence the sample was subtracted from the mean class image; values > 0 indicate a higher value in the mean, while values < 1 indicate a higher value in the sample.
- Image-Space-Outlier-Divergences.png
- Image-Space-Outlier-Divergences-All.png
- Image-Space-Nonoutlier-Divergences.png

### Mean vs Mean Divergence Plots
Identical to the ``Sample vs Mean Divergence Plots``, however compares mean class images against one another in place of a given sample. Mean class images are plotted on the diagonal, with divergences mirrored on either side. To compute the divergence, the image on the x-axis was subtracted from that on the y-axis.
- Image-Space-Overview.png
- Image-Space-Overview-raw.png
- Image-Space-Variance-raw.png

### Dendrograms
Dendrograms are a clustering method which iteratively combine similar samples based on similarity. We are less interested in the final clusters than we are trends in which samples are combined. All dendrograms were produced using either the compounded data or PCA on the _raw_ data with 5 principal components; this is reflected in the names of files. Samples were labelled according to their index in the dataset and colored according to class.
- agglomerative_dendrogram.png
- agglomerative_dendrogram_compounded.png
- agglomerative_dendrogram_comp_pca.png

### Dimensionality Reduction Techniques
We use PCA and UMAP to produce 2-dimensional latent spaces for our original flattened data. By plotting the distribution of our samples, we can observe common trends across samples and classes. As in the dendrograms, samples are labelled according to their index and colored by class. Poor-performing outliers are named in red.
- PCA_2d.png
- UMAP_2d.png

### Misclassification Rates
One aspect of our analysis involves tracking the performance of particular samples. We measure the misclassification rates of all samples, displaying the top ~20 from our MLP and SVC models. All rates were obtained by averaging performance over 100 indepedent runs. As in previous plots, samples are labelled according to index and colored by class. All samples which do not appear in our figures were classified with 100% accuracy.
- MLP_Sample_misclassifications_comp_pca.png
- MLP_Sample_misclassifications_comp_pca_filtered.png
- SVC_Sample_misclassifications_comp_pca.png
- SVC_Sample_misclassifications_comp_pca_filtered.png





