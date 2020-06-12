Some thoughts about this projects, some of them are no longer valid :)

- cluster cells
- check for convexity
- select genes
- assign to fish cells their phenotype

Selection of small number of signature genes:
LASSO forces coefficients to 0

once clusters are done, keep the 100 genes
then on fish data perform kNN
feature selection for more robustness: select genes with highest discriminative power = ratio between expression difference between clusters and expression variance per cluster.
bottom up approach: start with one gene, do kNN to assign with leave-one-out cells to clusters, look at F1 score with number of genes.


## questions

use original scRNAseq?

The hidden state space is assumed to consist of one of N possible values, modelled as a categorical distribution. (See the section below on extensions for other possibilities.)  

Alice knows the general weather trends in the area, and what Bob likes to do on average. In other words, the parameters of the HMM are known. nb of states?  

Statistical significance: What is the probability that a sequence drawn from some null distribution will have an HMM probability (in the case of the forward algorithm) or a maximum state sequence probability (in the case of the Viterbi algorithm) at least as large as that of a particular output sequence?  


 need for these features to be statistically independent of each other

 extend to 2nd, 3rd, etc neighbor
 possibly with different weight ~ kernel


 Neuropeptide genes and their receptors were frequently expressed
 in specific yet different cell types, suggesting specific cell-cell
 interactions (Supplementary Fig. 16).

 SVC on UMAP after HDBSCAN?

feature selection: L1 foro SVC, regularized trees, [Autoencoder inspired unsupervised feature selection](https://arxiv.org/pdf/1710.08310.pdf), [Local learning based feature selection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3445441/)
[recursive feature elimination](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2)

Optimal number of cluster: beware if assumption are that clusters are convex,like for the silhouette method (based on distance)
Use AIC, BIC or KIC

Effect of selecting high variability genes on detection of small phenotypes?
Use DE genes specific for cell types.

>A domain may
be formed by a cluster of cells from the same cell type, but it may also
consist of multiple cell types. In the latter scenario, the expression
patterns of cell-type-specific genes may not be spatially coherent, but
environment-associated genes would be expressed in spatial domains.

so they suppose some genes are related to regions, and their method detects them, but it's not really that, it detect patterns for all genes.
To detect only area related genes, we must subtract phenotype contributions.

> The
domain state of each cell was influenced by two sources (Fig. 2b): its
gene expression pattern and the domain states of neighboring cells

so they don't look at region-dependent genes, but at influence of supposed domains of neighboring cells.

> As an addi-
tional filter, we further removed 11 genes that were highly specific to
a single cell type, resulting in 69 genes (Supplementary Table 4) for
spatial domain identification. We found that this additional filtering
step improved the resolution while preserving the overall spatial pat-
tern

so we agree on that :)

> one domain was sporadically distributed
across in the inner layers of the cortex, and we labeled it as IS

suspicious...

> Differential gene expression analysis identified dis-
tinct signatures, which we labeled as the general domain signatures,
associated with each spatial domain

I have to do it too!

If we look at enough genes, aren't we sure to find one that validates our domain by being specific to it?
Try drawing random continuous domains, and test if I find "signatures".
--> imprtance of comparing to other datasets, like they do with the Allen Brain Atlas.

Lack of metric on clustering.

> we observed notable morphological variations
near the boundary between different domains at multiple regions (Fig. 3b),
including change of circularity and cell orientations, and these were
accompanied by metagene expression switches

interesting

> several domain-specific markers were also
markers of specific cell subtypes
