

- cluster cells
- check for convexity
- select genes
- assign to fish cells their phenotype

Selection of small number of signature genes:
LASSO forces coefficients to 0

once clusters are done, keep the 100 genes
then on fish data perform kNN
feature selection for morerobustness: select genes with highest discriminative power = ratio between expression difference between clusters and expression variance per cluster.
bottom up approach: start with one gene, do kNN to assign with leave-one-out cells to clusters, look at F1 score with number of genes.
