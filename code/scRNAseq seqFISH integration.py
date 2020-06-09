# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # scRNAsed seqFISH integration

# %% [markdown]
# This is a Jupytext file, i.e. a python script that can be used directly with a standard editor (Spyder, PyCharm, VS Code, ...) or as a JupyterLab notebook.  
# This format is much better for version control, see [here](https://nextjournal.com/schmudde/how-to-version-control-jupyter) for the several reasons.  
# If you want to use this file as a JupyterLab notebook (which I recommend) see the [installation instructions](https://jupytext.readthedocs.io/en/latest/install.html).

# %% [markdown]
# **Objectives:**
# - try to map non spatial scRNAseq data to spatial seqFISH data
# - find the minimum number of genes required for this mapping
# - investigate on whether there are some signatures in non spatial scRNAseq data about the spatial organisation of cells

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from path import Path

# %%
# %qtconsole

# %%
data_dir = Path("../data/raw/tasic_scRNAseq/")
scRNAseq_path = data_dir / "tasic_training_b2.txt"
seqFISH_path = data_dir / "seqfish_cortex_b2_testing.txt"
scRNAseq_labels_path = data_dir / "tasic_labels.tsv"
seqfish_labels_path = data_dir / "seqfish_labels.tsv"
seqFISH_coords_path = data_dir / "fcortex.coordinates.txt"

# %% [markdown]
# ### scRNAseq data

# %% [markdown]
# txt file of normalized scRNAseq data for `113 genes x 1723 cells`

# %%
scRNAseq = pd.read_csv(scRNAseq_path, sep='\t', header=None, index_col=0)
scRNAseq.index.name= 'genes'
scRNAseq = scRNAseq.transpose()
scRNAseq.index.name = 'cells'
scRNAseq.head()

# %% [markdown]
# ### seqFISH data

# %% [markdown]
# txt file of normalized seqFISH data for `113 genes x 1597 cells`

# %%
seqFISH = pd.read_csv(seqFISH_path, sep='\t', header=None, index_col=0)
seqFISH.index.name= 'genes'
seqFISH = seqFISH.transpose()
seqFISH.index.name = 'cells'
seqFISH.head()

# %% [markdown]
# ### scRNAseq labels

# %% [markdown]
# tsv file of cell type labels for scRNAseq

# %%
scRNAseq_labels = pd.read_csv(scRNAseq_labels_path, sep='\t', header=None)
scRNAseq_labels.head()

# %%
phenotypes = list(scRNAseq_labels.iloc[:,0].unique())
for phenotype in phenotypes:
    print(phenotype)

# %% [markdown]
# ### seqFISH coordinates

# %% [markdown]
# Spatial cell coordinates

# %%
seqFISH_coords = pd.read_csv(seqFISH_coords_path, sep=' ', header=None, usecols=[2,3], names=['x','y'])
seqFISH_coords.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Gene expression statistics

# %%
scRNAseq_stats = scRNAseq.describe().iloc[1:,:] # iloc to skip the `count` statistics
seqFISH_stats = seqFISH.describe().iloc[1:,:] # iloc to skip the `count` statistics
scRNAseq_stats.index.name = 'statistics'
seqFISH_stats.index.name = 'statistics'

# %%
scRNAseq_stats.T.hist(figsize=(17,8), sharex=True)
plt.suptitle('Summary statistics for scRNAseq data')

# %%
seqFISH_stats.T.hist(figsize=(17,8), sharex=True)
plt.suptitle('Summary statistics for seqFISH data')

# %% [markdown]
# So as stated in the documentation of the original BIRS repository, the scRNAseq data and seqFISH data are normalized, we shouldn't need to process them further.

# %% [markdown]
# ### Check data transformations with selected genes

# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from scipy.special import expit # the logistic function, the inverse of logit

# %%
# for 2 variables we compute different transformations that we store in a list
# choose preferentially variables that have different distribution statistics (outliers, ...)

distributions = []
selected_variables = ['capn13', 'cdc5l', 'cpne5']

for name in selected_variables:

    X = scRNAseq.loc[:,name].values.reshape(-1, 1) + 1 
    X_unitary = MinMaxScaler().fit_transform(X)
    
    distributions.append([
    ('Non transformed data '+name, X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',   # rescales the vector for each sample to have unit norm, independently of the distribution of the samples.
        Normalizer().fit_transform(X)),
    ('Data after square root transformation',
        np.sqrt((X))),
    ('Data after arcsin sqrt transformation',
        np.arcsin(np.sqrt((X_unitary)))),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X)),
    ('Data after log transformation (base e)',
        np.log(X)),
    ('Data after centered logistic transformation',
        expit(X-X.mean())),
])

# %%
nb_transfo = len(distributions[0])
fig, axes = plt.subplots(nrows=nb_transfo, ncols=len(selected_variables), figsize=plt.figaspect(2)*4)

for j in range(len(selected_variables)):
    one_var_all_transfo = distributions[j]
    for i in range(nb_transfo):
        title, data_transfo = one_var_all_transfo[i]
        bins = 50 # np.ceil(data_transfo.max() - data_transfo.min() + 1).astype(int)
        axes[i,j].hist(data_transfo, bins=bins)
        axes[i,j].set_title(title)
plt.tight_layout()

# %% [markdown]
# Here the data were already well transformed, so further transformations degrade the distributions for following tasks llike clustering.  
# Usually, on raw data, power-law transformations like Box-Cox or Yeo-Johnson lead to much improved distributions, they deal well with outliers.

# %% [markdown]
# ### Check data transformations with UMAP projections

# %%
import umap
# if not installed run: conda install -c conda-forge umap-learn

# %% [markdown]
# #### label re-encoding

# %%
# we transform Tasic's phenotypes from strings to integers
# for their visualization on projections

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(phenotypes)

y_true = le.transform(scRNAseq_labels.iloc[:,0])

# colors of data points according to their phenotypes
colors = [sns.color_palette()[x] for x in y_true]

# %% [markdown]
# #### scRNAseq transformation and projection

# %%
X = scRNAseq.values
scRNAseq_projections = [('No transformation ', 
                         umap.UMAP().fit_transform(X)),
                        ('Yeo-Johnson transformation',
                         umap.UMAP().fit_transform(PowerTransformer(method='yeo-johnson').fit_transform(X))),
                        ('Centered logistic transformation',
                         umap.UMAP().fit_transform(expit(X-X.mean())))]

# %%
size_points = 5.0
colormap = 'tab10'
marker='o'

nb_transfo = len(scRNAseq_projections)
fig, axes = plt.subplots(nrows=1, ncols=nb_transfo, figsize=(6*nb_transfo,6))

for i in range(nb_transfo):
    transfo_name, projection = scRNAseq_projections[i]
    axes[i].scatter(projection[:, 0], projection[:, 1], c=y_true, cmap=colormap, marker=marker, s=size_points)
    title = 'scRNAseq - ' + transfo_name
    axes[i].set_title(title, fontsize=15)
    axes[i].set_aspect('equal', 'datalim')
plt.tight_layout()

# %% [markdown]
# On the reduced space of the original data (left figure), the clusterization doesn't seem optimal, some groups are spread among different 'clouds' of points and some points are in the middle of another group.  
# The Yeo-Johnson doesn't help much as expected from the previous section.  
# The centered logistic transformation outputs 2 clear clusters, but there are certainly artifacts from the transformation. I actually selected this transformation because on the histograms of the selected genes we can see that the centered logistic transformation pushes data into 2 seperate groups.  
# But for exploratory analysis for this challenge we stick to the cluster definition of *Tasic et al.*

# %% [markdown]
# **Conclusion**  
# We will simply work on the data given by the workshop organizers as gene expression data are already well processed.

# %% [markdown]
# ## Map non spatial scRNAseq data to spatial seqFISH data

# %% [markdown]
# We train and test the models with the scRNAseq data, assuming Tasic's phenotypes definition is the gold standard.

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(scRNAseq)

# %%
from multiprocessing import cpu_count

nb_cores = cpu_count()
print(f"There are {nb_cores} available cores")

# %% [markdown]
# ### Test kNN

# %% [markdown]
# To test the minimum number of genes required for cell phenotype classification, we try quickly the k-nearest neighbors model.

# %%
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

clf = KNeighborsClassifier(n_neighbors=10)
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# The performance of kNN is not so good, we test the Support Vector Classifier

# %% [markdown]
# ### Test SVC

# %% [markdown]
# We use the same classifier as in *Zhu et al* for this exploratory anaysis, but ideally we should test a lot of different classifiers with hyperparameter search for each of them.  
# In Zhu et al they used `C = 1e−6`, `class_weight = 'balanced'`, `dual = False`, `max_iter = 10,000`, and `tol = 1e−4`  
# I just change max_iter to its default -1 (infinite)

# %%
from sklearn.svm import SVC

clf = SVC(class_weight = 'balanced', tol = 1e-4, C = 1e-6, max_iter = -1)
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# With Zhu's parameters the model is very bad!

# %%
from sklearn.svm import SVC

clf = SVC(class_weight = 'balanced')
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# With defaults parameters it's much better.

# %% [markdown]
# ### Find optimal hyperparameters for SVC

# %%
from time import time
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

clf = SVC(class_weight = 'balanced')

# specify parameters and distributions to sample from
param_dist = {'C': loguniform(1e-8, 1e4),
              'gamma': loguniform(1e-8, 1e4)}

# run randomized search
n_iter_search = 4000
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   n_jobs=nb_cores-1,
                                   scoring=['accuracy', 'balanced_accuracy'],
                                   refit='balanced_accuracy',
                                   random_state=0)

start = time()
random_search.fit(X, y_true)
end = time()
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((end - start), n_iter_search))


# Utility function to report best hyperparameters sets
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Utility function to report best hyperparameters sets with multiple scores
def report_multiple(results, n_top=3, scores=['balanced_accuracy','accuracy']):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_'+scores[0]] == i)
        for candidate in candidates:
            for score in scores:
                rank = results['rank_test_'+score][candidate]
                score_mean = results['mean_test_'+score][candidate]
                score_std = results['std_test_'+score][candidate]
                params = results['params'][candidate]
                print(f"Model with {score} rank: {rank}")
                print(f"Mean {score} validation score: {score_mean:1.3f} (std: {score_std:1.3f})")
            print(f"Parameters: {params}")
            print("")

report_multiple(random_search.cv_results_)

# %%
import json, time

# Utility function to save best hyperparameters sets
def summary_cv(results, n_top=20):
    summary = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            model = {'rank': i,
                     'mean validation score': results['mean_test_score'][candidate],
                     'std validation score': results['std_test_score'][candidate],
                     'parameters': results['params'][candidate]
                    }
            summary.append(model)
    return summary

# Utility function to save best hyperparameters sets with multiple scores
def cv_to_df(cv, scores, order_by=None, ascending=False):
    """
    Converts the result of a cross-validation randomized search from a dictionnary to a dataframe.
    
    Parameters
    ----------
    cv : dict
        The object output by scikit-learn RandomizedSearchCV
    scores: list(str)
        A list of metrics
    order_by: str, list(str), optional
        Single or list of scores (or ranks) used for ordering the dataframe 
    ascending: bool
        How to order the dataframe, default False
    
    Returns
    -------
    df : dataframe
        Results of the RandomizedSearchCV
    
    Examples
    --------
    >>> cv_to_df(random_search.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
    """
    
    scores=['balanced_accuracy','accuracy']
    df = pd.DataFrame(cv['params'])
    for score in scores:
        for stat in ['rank', 'mean', 'std']:
            col = stat + '_test_' + score
            new_col = stat + ' ' + score
            df = df.join(pd.DataFrame(cv[col], columns=[new_col]))
    
    if order_by is not None:
        df.sort_values(by=order_by, ascending=ascending, inplace=True)
        
    return df

t = time.localtime()
timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)

cv_search= cv_to_df(random_search.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
cv_search.to_csv(f'../data/processed/random_search_cv_results-{timestamp}.csv', index=False)


# %% [markdown]
# #### Map of hyperparameters scores

# %%
def hyperparam_map(results, param_x, param_y, score, n_top = 20, best_scores_coeff=500,
                   figsize = (10,10), best_cmap=sns.light_palette("green", as_cmap=True)):
    df = pd.DataFrame(results['params'])
    df['score'] = results[score]

    plt.figure(figsize=figsize)
    ax = sns.scatterplot(x=param_x, y=param_y, hue='score', size='score', data=df)
    ax.set_xscale('log')
    ax.set_yscale('log')

    best = df.sort_values(by='score', ascending=False).iloc[:n_top,:]
    best_scores = best['score'] - best['score'].min()
    best_scores = best_scores / best_scores.max()

    ax.scatter(x=best[param_x], y=best[param_y], c=best_scores, s=best_scores*best_scores_coeff, marker='+', cmap=best_cmap)
    ax.scatter(x=best[param_x].iloc[0], y=best[param_y].iloc[0], s=best_scores.iloc[0]*best_scores_coeff/2, marker='o', color='r', alpha=0.5)


# %%
hyperparam_map(random_search.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_balanced_accuracy', n_top = 20, figsize = (10,10))
title = 'CV balanced accuracy for SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_map(random_search.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_accuracy', n_top = 20, figsize = (10,10))
title = 'CV accuracy for SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# #### Zoomed hyperparameters search

# %%
clf = SVC(class_weight = 'balanced')

# specify parameters and distributions to sample from
param_dist = {'C': loguniform(1e-2, 1e4),
              'gamma': loguniform(1e-8, 1e-1)}

# run randomized search
n_iter_search = 2000
random_search_zoom = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   n_jobs=nb_cores-1,
                                   scoring=['accuracy', 'balanced_accuracy'],
                                   refit='balanced_accuracy',
                                   random_state=0)

start = time()
random_search_zoom.fit(X, y_true)
end = time()
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((end - start), n_iter_search))

report_multiple(random_search_zoom.cv_results_)

t = time.localtime()
timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
cv_search_zoom= cv_to_df(random_search_zoom.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
cv_search_zoom.to_csv(f'../data/processed/random_search_cv_zoom_results-{timestamp}.csv', index=False)

# %%
hyperparam_map(random_search_zoom.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_balanced_accuracy', n_top = 20, figsize = (10,10))
title = 'CV balanced accuracy for SVC zoomed hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_map(random_search_zoom.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_accuracy', n_top = 20, figsize = (10,10))
title = 'CV accuracy for SVC zoomed hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# ### Top-down elimination of variables

# %%
# If you want to run the LONG top-dpwn elimination of variables, uncomment and run this:

C = random_search.best_params_['C']
gamma = random_search.best_params_['gamma']
# clf = SVC(C=9.32, gamma=0.0157)
clf = SVC(C=C, gamma=gamma) 
Xelim = np.copy(X) # the X data that will be pruned

elimination_report = []

Nvar = X.shape[1]
for i in range(Nvar-1):
    print("Removing {} variables".format(i+1), end="    ")
    scores = []
    for var in range(Xelim.shape[1]):
        # we remove only one variabme at a time
        Xtest = np.delete(Xelim, var, axis=1)
        score = cross_val_score(clf, Xtest, y_true, cv=5, n_jobs=5).mean()
        #print("var {}/{}: {}".format(var+1, Xelim.shape[1], score))
        scores.append(score)
        
    # find the variable that was the less usefull for the model
    maxi_score = max(scores)
    worst_var = scores.index(maxi_score)
    print("eliminating var n°{}, the score was {:.3f}".format(worst_var, maxi_score))
    elimination_report.append([worst_var, maxi_score])
    # eliminate this variable for next round
    Xelim = np.delete(Xelim, worst_var, axis=1)

elimination_report = np.array(elimination_report)
np.savetxt("./elimination_report.csv", elimination_report, delimiter=',', header='var index, score', comments='', fmt=['%d', '%f'])

# %%
# If you want to load the data to display them directly, run this:
elimination_report = np.loadtxt("../data/processed/elimination_report.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables')
plt.ylabel('score')
plt.title('Score of scRNAseq classification during variables (genes) elimination')

# %% [markdown]
# First the funny thing is that the score is non monotonic wrt the number of remaining variables.  
# It looks like we could keep only 29 genes! (last maximum) 

# %% [markdown]
# From Zhu *et al*:  
#
# The performance was evaluated
# by cross-validation. By using only 40 genes, we were able to achieve
# an average level of 89% mapping accuracy. Increasing the number
# of genes led to better performance (92% for 60 genes and 96% for
# 80 genes)  
# Cross-validation analysis revealed that, using
# these 43 genes as input, the SVM model accurately mapped 90.1%
# of the cells in the scRNAseq data to the correct cell type. Thus, we
# used these 43 genes (Supplementary Table 2) to map cell types in
# the seqFISH data.  
# We found that 5.5%
# cells were excluded, that is, they could not be confidently mapped to
# a single cell type (with 0.5 or less probability). Among the mapped
# cells, 54% were glutamatergic neurons, 37% were GABAergic neurons,
# 4.8% were astrocytes, and other glial cell types and endothelial cells
# comprised the remaining 4.2% of cells

# %% [markdown]
# ### Infer cell types from restricted gene list

# %%
# Keep the inimum number of genes that lead to good predictions
genes_elim = elimination_report[:-28,0].astype(int)
# so probably it is actually not necessary to scale the data
# but for this notebook we will be consistent with the above analysis
Xtest = scaler.transform(scRNAseq)
Xpred = scaler.transform(seqFISH)  
for i in genes_elim:
    Xtest= np.delete(Xtest, i, axis=1)
    Xpred= np.delete(Xpred, i, axis=1)

# %%
clf = SVC(C=9.32, gamma=0.0157)
clf.fit(Xtest, y_true)
y_pred = clf.predict(Xpred)

# %%
plt.figure(figsize=[10,10])
plt.scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=y_pred, cmap=colormap, marker=marker, s=size_points)
plt.title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=18);

# %% [markdown]
# ## Spatial analysis

# %% [markdown]
# ### Network reconstruction

# %%
from scipy.spatial import Voronoi

vor = Voronoi(seqFISH_coords[['x','y']])

# arrays of x0, y0, x1, y1
voro_cells = np.zeros((vor.ridge_points.shape[0],4))
voro_cells[:,[0,1]] = seqFISH_coords.loc[vor.ridge_points[:,0], ['x','y']]
voro_cells[:,[2,3]] = seqFISH_coords.loc[vor.ridge_points[:,1], ['x','y']]
distances = np.sqrt((voro_cells[:,0]-voro_cells[:,2])**2+(voro_cells[:,1]-voro_cells[:,3])**2)

# %%
EDGE_DIST_THRESH = 300 # distance threshold to discard edges below it
selection = distances < EDGE_DIST_THRESH

plt.figure(figsize=[15,15])
for points in voro_cells[selection,:]:
    plt.plot(points[[0,2]],points[[1,3]], 'k-', alpha=0.5)
plt.scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=y_pred, cmap=colormap, marker=marker, s=10)
plt.title('Spatial network of seqFISH data', fontsize=18);

# %% [markdown]
# I have to fix the colour display of the nodes... :-/

# %% [markdown]
# The next step is, for each node, look at its neighboors, and aggregate in some way their gene expression data.  
# In the first place I think about mean and variance in order to capture the (non)homogeneity of cell types in the area.

# %% [markdown]
# ### Neighbors gene expression aggregation

# %%
nb_cells = Xtest.shape[0]
nb_genes = Xtest.shape[1]
genes_aggreg = np.zeros((nb_cells, nb_genes*2)) # *2 because mean and variance are stored
pair_points = vor.ridge_points[selection,:]

for i in range(nb_cells):
    left_neigh = pair_points[pair_points[:,1] == i, 0]
    right_neigh = pair_points[pair_points[:,0] == i, 1]
    # array of all neighboors of node i
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    if neigh.size != 0:
        genes_aggreg[i,:nb_genes] = Xtest[neigh,:].mean(axis=0)
        genes_aggreg[i,-nb_genes:] = Xtest[neigh,:].std(axis=0)
    else:
        genes_aggreg[i,:] = None

error_cells = np.isnan(genes_aggreg[:,0])
nb_errors = error_cells.sum()
print(f"There has been {nb_errors}/{nb_cells} cells set to NaN")

# %%
neigh_valid = genes_aggreg[~error_cells,:]

# %% [markdown]
# ### Neighbors aggregated genes visualization

# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(neigh_valid)
embedding.shape

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', marker=marker, s=size_points)
plt.title("Aggregated neighbors' genes data", fontsize=18);

# %% [markdown]
# It looks like we can define some clusters :)

# %% [markdown]
# ### Neighboors aggregated genes clustering

# %% [markdown]
# Now we can use our favorite clustering algorithm to find groups of similar points: HAC, OPTICS or HDBSCAN for instance.

# %% [markdown]
# #### HDBSCAN

# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=20)
clusterer.fit(neigh_valid)
print("HDBSCAN has detected {} clusters".format(clusterer.labels_.max()))

# %% [markdown]
# That isn't good at all!  
# We can try another algorithm

# %% [markdown]
# #### OPTICS

# %% [markdown]
# Defaults values are:  
# `OPTICS(min_samples=5, max_eps=inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None)`  
# a minkowsky distance with `p=2` is the euclidian distance, so that's fine

# %%
from sklearn.cluster import OPTICS, cluster_optics_dbscan

clust = OPTICS()
# Run the fit
clust.fit(neigh_valid)

# %%
clust.labels_.max()

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c=clust.labels_, cmap=colormap, marker=marker, s=size_points)
plt.title("Aggregated neighbors' genes data", fontsize=18);

# %% [markdown]
# That is not much better!  
#
# We should perform the clusterization on the reduced space, although it should be done with a lot of precautions (distances are not straighforwardly interpretable)

# %% [markdown]
# #### HDBSCAN on reduced space

# %%
embedding = umap.UMAP(n_neighbors=30,
                      min_dist=0.0,
                      n_components=2,
                      random_state=42,
                      ).fit_transform(neigh_valid)

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', marker=marker, s=size_points)
plt.title("Overview of aggregated neighbors' genes data", fontsize=18);

# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=20, min_samples=1)
clusterer.fit(embedding)
print("HDBSCAN has detected {} clusters".format(clusterer.labels_.max()))

# %% [markdown]
# we choose `min_samples=1` to avoid having points considered as noise

# %%
labels = clusterer.labels_
clustered = (labels >= 0)
plt.figure(figsize=[10,10])
plt.scatter(embedding[~clustered, 0],
            embedding[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=5,
            alpha=0.9)
plt.scatter(embedding[clustered, 0],
            embedding[clustered, 1],
            c=labels[clustered],
            s=5,
            cmap='Spectral');
plt.title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=18);

# %% [markdown]
# Of course one can tweak the parameters to obtain a clustering that fits him better.

# %% [markdown]
# #### OPTICS on reduced space

# %%
clust = OPTICS(min_cluster_size=50)
# Run the fit
clust.fit(embedding)

clust.labels_.max()

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c=clust.labels_, cmap=colormap, marker=marker, s=size_points)
plt.title("OPTICS clustering on aggregated neighbors' genes data", fontsize=18);

# %% [markdown]
# HDBSCAN provides a much better clustering regarding the data projection.

# %% [markdown]
# ### Visualisation of spatial seqFISH data and detected areas 

# %%
fig, ax = plt.subplots(1, 2, figsize=(15,7), tight_layout=True)
ax[0].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=y_pred, cmap=colormap, marker=marker, s=size_points)
ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=18);

ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels, cmap=colormap, marker=marker, s=size_points)
ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=18);

# %% [markdown]
# The detected areas look plausible as points affected to different area types are not randomly dispersed.  
# Moreover the detected areas span over areas of some phenotypes or form regions smaller than areas of some phenotypes.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# We have seen that it is possible to assign to seqFISH data points their corresponding phenotypes defined from the scRNAseq data, with only 29 genes.  
#
# Moreover for seqFISH data aggregating gene expression for each node and it's neighbors we have found different groups, which migh correspond to areas of cell of different proportions in phenotypes.  
# It would be interesting to check that in a further analysis.  
#
# An interesting lead could be, for each cell, retrieve the mean values of its corresponding phenotype (the 'signature' of the phenotype), and then run again an aggregated neighbors' gene expression analysis. That could emphasise the genes that are under or over expressed due to the localisation of the cells and eliminate the strong contributions of genes that are specific of cell type.

# %%
