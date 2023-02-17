# Plots 
This folder is subject to changes over different runs, as new plots are routinely produced and overwritten. Experiment-specific plots produced by ``visuals.py`` are located in ``processed/``, while more informative visualizations produced by the main scripts are in ``general/`` and ``CAMs/``.

## ``general`` plots
Here we provide annotations and commentary on the types of plots in ``general/``.

### Sample vs Mean Divergence Plots
These plots compare a provided sample against the mean image of a class. All images were taken from the _compounded, raw_ data and are on a scale from -1 to 1 (unless otherwise stated). To compute the divergence the sample was subtracted from the mean class image; values > 0 indicate a higher value in the mean, while values < 1 indicate a higher value in the sample.
- [Image-Space-Outlier-Divergences.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Outlier-Divergences.png) : Divergences between probable outliers and the corresponding mean class image. 
- [Image-Space-Outlier-Divergences-All.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Outlier-Divergences-All.png) : Divergences between probable outliers and **all** mean class images. 
- [Image-Space-Nonoutlier-Divergences.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Nonoutlier-Divergences.png) : Divergences between high-performance samples (which achieved 100% classification accuracy) and the corresponding mean class image, in addition to an example of a potential outlier.

### Mean vs Mean Divergence Plots
Identical to the ``Sample vs Mean Divergence Plots``, however compares mean class images against one another in place of a given sample. Mean class images are plotted on the diagonal, with divergences mirrored on either side. To compute the divergence, the image on the x-axis was subtracted from that on the y-axis.
- [Image-Space-Overview.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Overview.png) : Divergences on the compounded, processed data. On the scale (-1,1).
- [Image-Space-Overview-raw.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Overview-raw.png) : Divergences on the compounded, raw data. On the scale (-1,1).
- [Image-Space-Variance-raw.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/Image-Space-Variance-raw.png) : Variance divergences on the compounded, raw data. On the scale (-0.1,0.1).

### Dendrograms
Dendrograms are a clustering method which iteratively combine similar samples based on similarity. We are less interested in the final clusters than we are trends in which samples are combined. All dendrograms were produced using either the compounded data or PCA on the _raw_ data with 5 principal components; this is reflected in the names of files. Samples were labelled according to their index in the dataset and colored according to class.
- [agglomerative_dendrogram.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/agglomerative_dendrogram.png) : Clustering on the original full, raw dataset sans PCA. 
- [agglomerative_dendrogram_compounded.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/agglomerative_dendrogram_compounded.png) : Clustering on the compounded, raw dataset sans PCA.
- [agglomerative_dendrogram_comp_pca.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/agglomerative_dendrogram_comp_pca.png) : PCA clustering on the compounded, raw dataset.

### Dimensionality Reduction Techniques
We use PCA and UMAP to produce 2-dimensional latent spaces for our original flattened data. By plotting the distribution of our samples, we can observe common trends across samples and classes. As in the dendrograms, samples are labelled according to their index and colored by class. Poor-performing outliers are named in **red**.
- [PCA_2d.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/PCA_2d.png) : 2d PCA features. Produced from the compounded, raw data.
- [UMAP_2d.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/UMAP_2d.png) : 2d UMAP features. Produced from the compounded, raw data.

### Misclassification Rates
One aspect of our analysis involves tracking the performance of particular samples. We measure the misclassification rates of all samples, displaying the top ~20 from our MLP and SVC models. All rates were obtained by averaging performance over 100 indepedent runs. As in previous plots, samples are labelled according to index and colored by class. All samples which do not appear in our figures were classified with 100% accuracy.
- [MLP_Sample_misclassifications_comp_pca.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/MLP_Sample_misclassifications_comp_pca.png) : Samplewise misclassification rates for an MLP model.
- [MLP_Sample_misclassifications_comp_pca_filtered.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/MLP_Sample_misclassifications_comp_pca_filtered.png) : Samplewise misclassification rates for an MLP model with the worst-performing sample (probable outliers) of each class removed.
- [SVC_Sample_misclassifications_comp_pca.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/SVC_Sample_misclassifications_comp_pca.png) : Samplewise misclassification rates for an SVC model.
- [SVC_Sample_misclassifications_comp_pca_filtered.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/SVC_Sample_misclassifications_comp_pca_filtered.png) : Samplewise misclassification rates for an SVC model with the worst-performing sample (probable outliers) of each class removed.

### CNN Training vs Validation 
To properly analyze the variance of our CNN models, we plot the final training and validation accuracies of 30 identical CMapNN models with a kernel size of 3x3. We produce several plots of our proposed datasets for comparison.
- [DL-Overview_raw.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/DL-Overview_raw.png) : Compounded, raw data.
- [DL_Overview_processed.png](https://github.com/asher-k/MilkRA/blob/main/output/figures/general/DL_Overview_processed.png) : Compounded, processed data. Processed data averages over both sides of the droplet. Of interest is that < 30 points are visible on the plot, suggesting more consistent performance compared to the raw data.

## ``CAM`` plots
A Class Activation Map (CAM) is a visualization of the impact convolutional filters have on certain regions of an image. These filters are obtained by extracting the final convolved image before the Global Average Pooling (GAP) layer. By extracting input weights from a fully-connected layer that takes the GAP features as input we obtain "weights" for each convolution-class pair. In our plots we display the CAM for the predicted class from a high-performance model composed of 3 convolutional blocks, which achieved 100% training accuracy and 95% validation accuracy.
- [CAM/](https://github.com/asher-k/MilkRA/blob/main/output/figures/CAMs/)
