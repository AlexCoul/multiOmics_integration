# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # scRNAseq seqFISH integration

# %% [markdown]
# This is a Jupytext file, i.e. a python script that can be used directly with a standard editor (Spyder, PyCharm, VS Code, ...) or as a JupyterLab notebook.  
# This format is much better for version control, see [here](https://nextjournal.com/schmudde/how-to-version-control-jupyter) for the several reasons.  
# If you want to use this file as a JupyterLab notebook (which I recommend) see the [installation instructions](https://jupytext.readthedocs.io/en/latest/install.html).

# %% [markdown]
# **Objectives:**
# - try to map non spatial scRNAseq data to spatial seqFISH data
# - find the minimum number of genes required for this mapping
# - investigate on whether there are some signatures in non spatial scRNAseq data about the spatial organisation of cells

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from path import Path
from time import time
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

import umap
# if not installed run: conda install -c conda-forge umap-learn
import hdbscan
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# %%
# %qtconsole

# %%
data_dir = Path("../data/raw/tasic_scRNAseq/")
scRNAseq_path = data_dir / "tasic_training_b2.txt"
seqFISH_path = data_dir / "seqfish_cortex_b2_testing.txt"
scRNAseq_labels_path = data_dir / "tasic_labels.tsv"
seqFISH_labels_path = data_dir / "seqfish_labels.tsv"
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
counts = scRNAseq_labels.groupby(0)[0].value_counts()
counts.index = counts.index.droplevel()
counts.index.name = 'phenotype'
counts

# %% [markdown]
# Tha classes are higly imbalanced.

# %% [markdown]
# #### label re-encoding

# %%
# we transform Tasic's phenotypes from strings to integers
# for their visualization on projections

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
phenotypes = list(scRNAseq_labels.iloc[:,0].unique())
le.fit(phenotypes)

y_true = le.transform(scRNAseq_labels.iloc[:,0])

# colors of data points according to their phenotypes
colors = [sns.color_palette()[x] for x in y_true]

# %% [markdown]
# ### seqFISH labels

# %% [markdown]
# tsv file of cell type labels for seqFISH

# %%
seqFISH_labels = pd.read_csv(seqFISH_labels_path, sep='\t', header=None)
seqFISH_labels.head()

# %%
seqFISH_counts = seqFISH_labels.groupby(2)[2].value_counts()
seqFISH_counts.index = seqFISH_counts.index.droplevel()
seqFISH_counts.index.name = 'phenotype'
seqFISH_counts

# %% [markdown]
# ### seqFISH coordinates

# %% [markdown]
# Spatial cell coordinates

# %%
seqFISH_coords = pd.read_csv(seqFISH_coords_path, sep=' ', header=None, usecols=[2,3], names=['x','y'])
seqFISH_coords.head()

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ## Exploratory Data Analysis

# %% [markdown]
# ### Gene expression statistics

# %% [markdown]
# #### Separate statistics

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
# #### Differences in distributions's statistics

# %%
diff_stats = seqFISH_stats - scRNAseq_stats
diff_stats.T.hist(figsize=(17,8), sharex=True)
plt.suptitle('Differences in summary statistics')

# %% [markdown]
# The distributions of gene expression data are very comparable between the 2 datasets.  
# If we use a standard scaler, does it make them more comparable?

# %%
scRNAseq_scaled = StandardScaler().fit_transform(scRNAseq)
seqFISH_scaled = StandardScaler().fit_transform(seqFISH)
# convert it to DataFrame to use convenient methods
gene_names = scRNAseq.columns
scRNAseq_scaled = pd.DataFrame(data=scRNAseq_scaled, columns=gene_names)
seqFISH_scaled = pd.DataFrame(data=seqFISH_scaled, columns=gene_names)
# compute the statistics
scRNAseq_stats_scaled = scRNAseq_scaled.describe().iloc[1:,:] # iloc to skip the `count` statistics
seqFISH_stats_scaled = seqFISH_scaled.describe().iloc[1:,:] # iloc to skip the `count` statistics
scRNAseq_stats_scaled.index.name = 'statistics'
seqFISH_stats_scaled.index.name = 'statistics'

# %%
diff_stats_scaled = seqFISH_stats_scaled - scRNAseq_stats_scaled
diff_stats_scaled.T.hist(figsize=(17,8), sharex=True)

plt.suptitle('Differences in summary statistics of scaled datasets');

# %% [markdown]
# With a dedicated scaler object for each dataset, their distribution statistics are more comparable.  
# We will use dedicated scaler objects to improve the applicability to seqFISH of a classifier trained on scRNAseq.


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
title = 'Check data transformations with UMAP projections'
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# On the reduced space of the original data (left figure), the clusterization doesn't seem optimal, some groups are spread among different 'clouds' of points and some points are in the middle of another group.  
# The Yeo-Johnson doesn't help much as expected from the previous section.  
# The centered logistic transformation outputs 2 clear clusters, but there are certainly artifacts from the transformation. I actually selected this transformation because on the histograms of the selected genes we can see that the centered logistic transformation pushes data into 2 seperate groups.  
# But for exploratory analysis for this challenge we stick to the cluster definition of *Tasic et al.*

# %% [markdown]
# **Conclusion**  
# We will simply work on the data given by the workshop organizers as gene expression data are already well processed.

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ## Infer scRNAseq-defined cell types from seqFISH data

# %% [markdown]
# We train and test the models with the scRNAseq data, assuming Tasic's phenotypes definition is the gold standard.

# %%
from multiprocessing import cpu_count

nb_cores = cpu_count()
print(f"There are {nb_cores} available cores")

# %%
scaler = StandardScaler()
X = scaler.fit_transform(scRNAseq)

# %% [markdown]
# ### Test kNN

# %% [markdown]
# To test the minimum number of genes required for cell phenotype classification, we try quickly the k-nearest neighbors model.

# %%
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
clf = SVC(class_weight = 'balanced', tol = 1e-4, C = 1e-6, max_iter = -1)
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# With Zhu's parameters the model is very bad, but they actually used `LinearSVC`!

# %%
clf = LinearSVC(class_weight="balanced", dual=False, C = 1e-6, max_iter=10000, tol=1e-4)
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# it's a bit better

# %%
clf = SVC(class_weight = 'balanced')
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")

# %% [markdown]
# Kernel SVC with defaults parameters it's much better.

# %%
clf = LinearSVC(class_weight="balanced", dual=False)
cv = cross_validate(clf, X, y_true,
                    cv=10,
                    n_jobs=nb_cores-1,
                    scoring=['accuracy', 'balanced_accuracy'])
print(f"Accuracy: {cv['test_accuracy'].mean():0.2f} (+/- {cv['test_accuracy'].std()*2:0.2f})")
print(f"Balanced accuracy: {cv['test_balanced_accuracy'].mean():0.2f} (+/- {cv['test_balanced_accuracy'].std()*2:0.2f})")


# %% [markdown]
# Linear SVC with default parameters is better than with Zhu's, but not than kernel SVC.

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ### Find optimal hyperparameters for kernel SVC

# %%
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

import json

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


# %%
# # LONG runtime

# clf = SVC(class_weight = 'balanced')

# # specify parameters and distributions to sample from
# param_dist = {'C': loguniform(1e-8, 1e4),
#               'gamma': loguniform(1e-8, 1e4)}

# # run randomized search
# n_iter_search = 4000
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search,
#                                    n_jobs=nb_cores-1,
#                                    scoring=['accuracy', 'balanced_accuracy'],
#                                    refit='balanced_accuracy',
#                                    random_state=0)

# start = time.time()
# random_search.fit(X, y_true)
# end = time.time()
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((end - start), n_iter_search))

# report_multiple(random_search.cv_results_)

# %%
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
# # LONG runtime
# clf = SVC(class_weight = 'balanced')

# # specify parameters and distributions to sample from
# param_dist = {'C': loguniform(1e-2, 1e4),
#               'gamma': loguniform(1e-8, 1e-1)}

# # run randomized search
# n_iter_search = 2000
# random_search_zoom = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search,
#                                    n_jobs=nb_cores-1,
#                                    scoring=['accuracy', 'balanced_accuracy'],
#                                    refit='balanced_accuracy',
#                                    random_state=0)

# start = time.time()
# random_search_zoom.fit(X, y_true)
# end = time.time()
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((end - start), n_iter_search))

# report_multiple(random_search_zoom.cv_results_)

# t = time.localtime()
# timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
# cv_search_zoom= cv_to_df(random_search_zoom.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
# cv_search_zoom.to_csv(f'../data/processed/random_search_cv_zoom_results-{timestamp}.csv', index=False)

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
# Best hyperparameters are higher for simple accuracy than for balanced accuracy

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ### Find optimal hyperparameters for Linear SVC

# %%
# # LONG runtime

# clf = LinearSVC(class_weight="balanced", dual=False, max_iter=10000)

# # specify parameters and distributions to sample from
# nb_candidates = 600
# param_grid = {'C': np.logspace(start=-8, stop=4, num=nb_candidates, endpoint=True, base=10.0)}

# # run grid search
# grid_search = GridSearchCV(clf,
#                            param_grid=param_grid,
#                            n_jobs=nb_cores-1,
#                            scoring=['accuracy', 'balanced_accuracy'],
#                            refit='balanced_accuracy')

# start = time.time()
# grid_search.fit(X, y_true)
# end = time.time()
# print("GridSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((end - start), nb_candidates))

# report_multiple(grid_search.cv_results_)

# %%
t = time.localtime()
timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)

cv_search = cv_to_df(grid_search.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
cv_search.to_csv(f'../data/processed/grid_search_cv_results-{timestamp}.csv', index=False)


# %% [markdown]
# #### Line hyperparameters scores

# %%
def hyperparam_line(df, param_x, scores, figsize=(10,5), log_scale=True, legend=True):
    plt.figure(figsize=figsize)
    for score in scores:
        ax = sns.lineplot(x=param_x, y=score, data=df, label=score)
    if log_scale:
        ax.set_xscale('log')
    if legend:
        plt.legend()


# %%
cv_search = pd.read_csv('../data/processed/grid_search_cv_results-2020-06-09_18h36.csv')

hyperparam_line(cv_search, param_x = 'C', scores = ['mean balanced_accuracy'], figsize = (10,5))
plt.ylim([0.45,0.95])
title = 'CV balanced accuracy for Linear SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_line(cv_search, param_x = 'C', scores = ['mean accuracy'], figsize = (10,5))
plt.ylim([0.45,0.95])
title = 'CV accuracy for Linear SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %%
cv_search = pd.read_csv('../data/processed/grid_search_cv_results-2020-06-09_18h36.csv')

hyperparam_line(cv_search, param_x='C', scores = ['mean balanced_accuracy', 'mean accuracy'])
title = 'CV accuracies for Linear SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# With LinearSVC as in Zhu's paper we can reach an accuracy of 0.9 too, but a balanced accuracy of 0.78 only.  
# We will stick to our kernel SVC as the balanced accuracy is a bit higher.

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## Top-down elimination of variables

# %%
from multiprocessing import cpu_count
from dask.distributed import Client
from dask import delayed

nb_cores = cpu_count()
print(f"There are {nb_cores} available cores")

client = Client(n_workers=nb_cores-1)

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ### For kernel SVC

# %%
# If you want to run the LONG top-dpwn elimination of variables, uncomment and run this:

# # cross_val_score because now we only look at balanced_accuracy
# from sklearn.model_selection import cross_val_score

# # if you have run the parameter search in the 2nd previous cell
# # C = random_search_zoom.best_params_['C']
# # gamma = random_search_zoom.best_params_['gamma']
# # or if you need to load search results previously saved:
# param_search = pd.read_csv('../data/processed/random_search_cv_zoom_results-2020-05-09_09h50.csv')
# scoring = 'balanced-accuracy'
# best = param_search.loc[param_search['rank '+scoring]==1]
# C = best['C'].values[0]
# gamma = best['gamma'].values[0]
# # clf = SVC(C=9.32, gamma=0.0157) # old value

# clf = SVC(C = C, gamma = gamma, class_weight = 'balanced') 
# Xelim = np.copy(X) # the X data that will be pruned

# elimination_report = []

# def score_leaveoneout(clf, y_true, Xelim, var, scoring):
#     Xtest = np.delete(Xelim, var, axis=1)
#     score = cross_val_score(clf, Xtest, y_true,
#                             cv=5,
#                             n_jobs=5,
#                             scoring=scoring).mean()
#     return score

# Nvar = X.shape[1]
# start = time.time()
# for i in range(Nvar-1):
#     print("Removing {} variables".format(i+1), end="    ")
#     scores = []
#     for var in range(Xelim.shape[1]):
#         score = delayed(score_leaveoneout)(clf, y_true, Xelim, var, scoring)
#         scores.append(score)

#     # find the variable that was the less usefull for the model
#     maxi_score = delayed(max)(scores)
#     worst_var = delayed(list.index)(scores, maxi_score)
#     maxi_score = maxi_score.compute()
#     worst_var = worst_var.compute()
#     print("eliminating var n°{}, the score was {:.6f}".format(worst_var, maxi_score))
#     elimination_report.append([worst_var, maxi_score])
#     # eliminate this variable for next round
#     Xelim = np.delete(Xelim, worst_var, axis=1)
    
# end = time.time()
# duration = end-start
# print(f"the computation lasted {duration:.1f}s")
# elimination_report = np.array(elimination_report)
# t = time.localtime()
# timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
# np.savetxt(f"../data/processed/elimination_report-{scoring}-{timestamp}.csv", 
#            elimination_report,
#            delimiter=',',
#            header='var index, score',
#            comments='',
#            fmt=['%d', '%f'])

# %% [markdown]
# #### elimination with balanced accuracy

# %%
# If you want to load the data to display them directly, run this:
elimination_report = np.loadtxt("../data/processed/elimination_report-balanced-accuracy-2020-06-09_13h11.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables + 1')
plt.ylabel('score')
title = 'Balanced accuracy of scRNAseq classification during variables (genes) elimination'
plt.title(title);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# We can eliminate ~3/4 of genes without a signicative change in the classification performance.  
# We even have better performances between 18 and 27 remaining genes.  
# It looks like we could keep only last 14 genes and still have a reasonable performance (0.81), the maximum performance being 0.838 when we have 26 genes left.

# %% [markdown]
# #### elimination with accuracy

# %%
elimination_report = np.loadtxt("../data/processed/elimination_report-accuracy-2020-06-09_15h02.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables + 1')
plt.ylabel('score')
title = 'Accuracy of scRNAseq classification during variables (genes) elimination'
plt.title(title);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# So there is a clear difference, and it is easy to overestimate the perfomance of the classifier with the wrong metrics

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ### For linear SVC

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

# %%
# If you want to run the LONG top-dpwn elimination of variables, uncomment and run this:

# # cross_val_score because now we only look at balanced_accuracy
# from sklearn.model_selection import cross_val_score

# # if you have run the parameter search in the 2nd previous cell
# # C = random_search_zoom.best_params_['C']
# # gamma = random_search_zoom.best_params_['gamma']
# # or if you need to load search results previously saved:
# param_search = pd.read_csv('../data/processed/grid_search_cv_results-2020-06-09_18h36.csv')
# scoring = 'accuracy'
# best = param_search.loc[param_search['rank '+scoring]==1]
# C = best['C'].values[0]

# clf = LinearSVC(C=C, class_weight='balanced', dual=False, max_iter=10000)
# Xelim = np.copy(X) # the X data that will be pruned

# elimination_report = []

# def score_leaveoneout(clf, y_true, Xelim, var, scoring):
#     Xtest = np.delete(Xelim, var, axis=1)
#     score = cross_val_score(clf, Xtest, y_true,
#                             cv=5,
#                             n_jobs=5,
#                             scoring=scoring).mean()
#     return score

# Nvar = X.shape[1]
# start = time.time()
# for i in range(Nvar-1):
#     print("Removing {} variables".format(i+1), end="    ")
#     scores = []
#     for var in range(Xelim.shape[1]):
#         score = delayed(score_leaveoneout)(clf, y_true, Xelim, var, scoring)
#         scores.append(score)

#     # find the variable that was the less usefull for the model
#     maxi_score = delayed(max)(scores)
#     worst_var = delayed(list.index)(scores, maxi_score)
#     maxi_score = maxi_score.compute()
#     worst_var = worst_var.compute()
#     print("eliminating var n°{}, the score was {:.6f}".format(worst_var, maxi_score))
#     elimination_report.append([worst_var, maxi_score])
#     # eliminate this variable for next round
#     Xelim = np.delete(Xelim, worst_var, axis=1)
    
# end = time.time()
# duration = end-start
# print(f"the computation lasted {duration:.1f}s")
# elimination_report = np.array(elimination_report)
# t = time.localtime()
# timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
# np.savetxt(f"../data/processed/elimination_report_linearSVC-{scoring}-{timestamp}.csv",
#            elimination_report,
#            delimiter=',',
#            header='var index, score',
#            comments='',
#            fmt=['%d', '%f'])

# %% [markdown]
# #### elimination with balanced accuracy

# %%
# If you want to load the data to display them directly, run this:
elimination_report = np.loadtxt("../data/processed/elimination_report_linearSVC-balanced_accuracy-2020-06-09_19h47.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables + 1')
plt.ylabel('score')
title = 'Balanced accuracy of scRNAseq classification with Linear SVC during variables (genes) elimination'
plt.title(title);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# #### elimination with accuracy

# %%
elimination_report = np.loadtxt("../data/processed/elimination_report_linearSVC-accuracy-2020-06-09_22h30.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables + 1')
plt.ylabel('score')
title = 'Accuracy of scRNAseq classification with Linear SVC during variables (genes) elimination'
plt.title(title);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# These plots confirm that Zhu *et al.* probably used the simple accuracy instead of the balanced accuracy.  
# The kernel SVC require fewer genes (\~19-29) than the linear SVC (\~35-39) to maintain a relatively high score.

# %%
linear_simple_acc = np.loadtxt("../data/processed/elimination_report_linearSVC-accuracy-2020-06-09_22h30.csv", skiprows=1, delimiter=',')
linear_balanced_acc = np.loadtxt("../data/processed/elimination_report_linearSVC-balanced_accuracy-2020-06-09_19h47.csv", skiprows=1, delimiter=',')
kernel_simple_acc = np.loadtxt("../data/processed/elimination_report-accuracy-2020-06-09_15h02.csv", skiprows=1, delimiter=',')
kernel_balanced_acc = np.loadtxt("../data/processed/elimination_report-balanced-accuracy-2020-06-09_13h11.csv", skiprows=1, delimiter=',')
# number of remaining variables
x = np.arange(linear_simple_acc.shape[0]) + 1

plt.figure(figsize=(12,7))
plt.plot(x, linear_simple_acc[::-1,1], label='Linear SVC, simple accuracy', c='dodgerblue', linestyle='dashed');
plt.plot(x, linear_balanced_acc[::-1,1], label='Linear SVC, balanced accuracy', c='dodgerblue', linestyle='solid');
plt.plot(x, kernel_simple_acc[::-1,1], label='Kernel SVC, simple accuracy', c='salmon', linestyle='dashed');
plt.plot(x, kernel_balanced_acc[::-1,1], label='Kernel SVC, balanced accuracy', c='salmon', linestyle='solid');
plt.xlabel('# remaining variables')
plt.ylabel('score')
plt.legend()
title = 'Perfomance evaluations of scRNAseq classification with Linear & Kernal SVC during variables (genes) elimination'
plt.title(title);
plt.savefig('../data/processed/'+title, bbox_inches='tight')


# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# ## Infer cell types from restricted gene list

# %%
def successive_elimination(list_elems, elems_id):
    """
    Compute the order of discarded elements during 
    successive deletions at indicated indices.
    
    Parameters
    ----------
    list_elems : list
        A list of elements that are eliminated
    elems_id: list
        A list of indices where elements are successively deleted
    
    Returns
    -------
    successive : list
        The ordered list of deleted elements
    
    Examples
    --------
    >>> gene_names = list(scRNAseq.columns)
    >>> order_elim_id = elimination_report[:,0].astype(int)
    >>> order_elim = successive_elimination(gene_names, genes_elim_id)
    """
    
    from copy import deepcopy
    elems = deepcopy(list_elems)
    
    successive = []
    for i in elems_id:
        successive.append(elems.pop(i))
    
    return successive


# %% [markdown]
# #### Drop genes

# %%
elimination_report = np.loadtxt("../data/processed/elimination_report-balanced-accuracy-2020-06-09_13h11.csv", skiprows=1, delimiter=',')

# Keep the minimum number of genes that lead to good predictions
gene_names = list(scRNAseq.columns)
# nb_elim = 86 # optimal number of genes to discard
nb_elim = 94 # eliminate more genes, with still good performance
order_elim_id = elimination_report[:,0].astype(int)
order_elim = successive_elimination(gene_names, order_elim_id)
genes_elim = order_elim[:nb_elim]

scRNAseq_drop = copy.deepcopy(scRNAseq)
seqFISH_drop = copy.deepcopy(seqFISH)

scRNAseq_drop.drop(columns=genes_elim, inplace=True)
seqFISH_drop.drop(columns=genes_elim, inplace=True)

scaler_sc = StandardScaler()  # for scRNAseq
scaler_seq = StandardScaler() # for seqFISH
Xtest = scaler_sc.fit_transform(scRNAseq_drop)
Xpred = scaler_seq.fit_transform(seqFISH_drop)  
print(f"There are {Xtest.shape[1]} remaining genes")


# %% [markdown]
# #### Re-run hyperparameters search

# %%
# # LONG runtime

# clf = SVC(class_weight = 'balanced')

# # specify parameters and distributions to sample from
# param_dist = {'C': loguniform(1e-8, 1e4),
#               'gamma': loguniform(1e-8, 1e4)}

# # run randomized search
# n_iter_search = 4000
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search,
#                                    n_jobs=nb_cores-1,
#                                    scoring=['accuracy', 'balanced_accuracy'],
#                                    refit='balanced_accuracy',
#                                    random_state=0)

# start = time.time()
# random_search.fit(Xtest, y_true)
# end = time.time()
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((end - start), n_iter_search))

# t = time.localtime()
# timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
# cv_search= cv_to_df(random_search.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
# cv_search.to_csv(f'../data/processed/random_search_cv_dropped_genes_results-{timestamp}.csv', index=False)

# cv_search.head(10)

# %% [markdown]
# #### Map of hyperparameters scores

# %%
hyperparam_map(random_search.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_balanced_accuracy', n_top = 20, figsize = (10,10))
title = 'CV balanced accuracy for SVC hyperparameters search - dropped genes'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_map(random_search.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_accuracy', n_top = 20, figsize = (10,10))
title = 'CV accuracy for SVC hyperparameters search - dropped genes'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# #### Zoomed hyperparameters search

# %%
# # LONG runtime

# clf = SVC(class_weight = 'balanced')

# # specify parameters and distributions to sample from
# param_dist = {'C': loguniform(1e-2, 1e4),
#               'gamma': loguniform(1e-8, 1e-1)}

# # run randomized search
# n_iter_search = 2000
# random_search_zoom = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search,
#                                    n_jobs=nb_cores-1,
#                                    scoring=['accuracy', 'balanced_accuracy'],
#                                    refit='balanced_accuracy',
#                                    random_state=0)

# start = time.time()
# random_search_zoom.fit(Xtest, y_true)
# end = time.time()
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((end - start), n_iter_search))

# t = time.localtime()
# timestamp = time.strftime('%Y-%m-%d_%Hh%M', t)
# cv_search_zoom= cv_to_df(random_search_zoom.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
# cv_search_zoom.to_csv(f'../data/processed/random_search_cv_dropped_genes_zoom_results-{timestamp}.csv', index=False)

# cv_search_zoom.head(10)

# %%
hyperparam_map(random_search_zoom.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_balanced_accuracy', n_top = 20, figsize = (10,10))
title = 'CV balanced accuracy for SVC zoomed hyperparameters search - dropped genes'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_map(random_search_zoom.cv_results_, param_x = 'C', param_y = 'gamma', score = 'mean_test_accuracy', n_top = 20, figsize = (10,10))
title = 'CV accuracy for SVC zoomed hyperparameters search - dropped genes'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# #### Predict phenotypes of seqFISH cells

# %%
param_search = pd.read_csv('../data/processed/random_search_cv_dropped_genes_zoom_results-2020-06-09_23h21.csv')
scoring = 'balanced_accuracy'
best = param_search.loc[param_search['rank '+scoring]==1]
C = best['C'].values[0]
gamma = best['gamma'].values[0]

clf = SVC(C=C, gamma=gamma, class_weight='balanced')
clf.fit(Xtest, y_true)
y_pred = clf.predict(Xpred)

# %%
pheno_id, counts = np.unique(y_pred, return_counts=True)
pheno_names = le.inverse_transform(pheno_id)
pd.DataFrame(data={'phenotype':pheno_names,
                   'counts':counts},
             index=pheno_id)

# %%
base_cmap = sns.color_palette('muted').as_hex() # plt.get_cmap("Set1")

# make a custom colormap with class integers, labels and hex colors like
# [[0, 'Astrocyte', '#023eff'],
#  [2, 'GABA-ergic Neuron', '#ff7c00'],
#  [3, 'Glutamatergic Neuron', '#1ac938']]
# 
color_labels = []
for i, i_pred in enumerate(np.unique(y_pred)):
    color_labels.append([i_pred, le.inverse_transform([i_pred])[0], base_cmap[i]])

# more custom colormap, switch to previous line if any issue like different class integers
# color_labels = [[0, le.inverse_transform([0])[0], base_cmap[0]],
#                 [2, le.inverse_transform([2])[0], base_cmap[1]],
#                 [3, le.inverse_transform([3])[0], base_cmap[2]]]

fig, ax = plt.subplots(figsize=[10,10])
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=10)
plt.legend()

title = 'Spatial map of prediction of phenotypes for seqFISH data'
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# ## Spatial analysis

# %% [markdown]
# The main idea is to reconstruct the spatial network of cells, where nodes are cells and edges link neighboring cells.  
# We will use this network to analyse how neighboring cells (linked nodes) can define a spacially coherent area.

# %% [markdown]
# ### Network reconstruction

# %% [markdown]
# We use Voronoi tessellation to define the edges that link neighboring cells.  
# Voroinoi tessellation defines virtual cell boundaries, and we link cells that share boundaries.

# %%
from scipy.spatial import Voronoi

vor = Voronoi(seqFISH_coords[['x','y']])

# arrays of x0, y0, x1, y1
voro_cells = np.zeros((vor.ridge_points.shape[0],4))
voro_cells[:,[0,1]] = seqFISH_coords.loc[vor.ridge_points[:,0], ['x','y']]
voro_cells[:,[2,3]] = seqFISH_coords.loc[vor.ridge_points[:,1], ['x','y']]
distances = np.sqrt((voro_cells[:,0]-voro_cells[:,2])**2+(voro_cells[:,1]-voro_cells[:,3])**2)

# %%
fig, ax = plt.subplots(figsize=[15, 15])
for points in voro_cells[:,:]:
    ax.plot(points[[0,2]],points[[1,3]], c='k',zorder=0, alpha=0.5)
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
plt.legend()

title = 'Spatial network of seqFISH data without distance threshold'
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# There are many edges that connect nodes on the border of the network, it's only due to Voronoi tessellation and limit conditions.  
# We need to cut edges that link nodes that are more distant than a given threshold.

# %%
# distance threshold to discard edges above it
#  mainly artifacts at the borders of the whole dataset
EDGE_DIST_THRESH = 300 
selection = distances < EDGE_DIST_THRESH
pairs = vor.ridge_points[selection,:]

fig, ax = plt.subplots(figsize=[15, 15])
for points in voro_cells[selection,:]:
    ax.plot(points[[0,2]],points[[1,3]], c='k',zorder=0, alpha=0.5)
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
plt.legend()

title = 'Spatial network of seqFISH data'
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')


# %% [markdown]
# Instead of cutting edges, we could give them weight thats decrease with distance. But that's an improvement that requires some time to implement.  
# The next step is, for each node, look at its neighboors, and aggregate in some way their gene expression data.  
# In the first place I think about mean and variance in order to capture the (non)homogeneity of cell types in the area.

# %% [markdown]
# ### Neighbors gene expression aggregation

# %% [markdown]
# We want to aggregate gene expression data up the 2nd, 3rd ot k-th neighbors.

# %% [markdown]
# #### Functions definition

# %%
def neighbors(pairs, n):
    """
    Return the list of neighbors of a node in a network defined 
    by edges between pairs of nodes. 
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
        
    Returns
    -------
    neigh : array_like
        The indices of neighboring nodes.
    """
    
    left_neigh = pairs[pairs[:,1] == n, 0]
    right_neigh = pairs[pairs[:,0] == n, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    return neigh

def neighbors_k_order(pairs, n, order):
    """
    Return the list of up the kth neighbors of a node 
    in a network defined by edges between pairs of nodes
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
    order : int
        Max order of neighbors.
        
    Returns
    -------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order
    
    
    Examples
    --------
    >>> pairs = np.array([[0, 10],
                        [0, 20],
                        [0, 30],
                        [10, 110],
                        [10, 210],
                        [10, 310],
                        [20, 120],
                        [20, 220],
                        [20, 320],
                        [30, 130],
                        [30, 230],
                        [30, 330],
                        [10, 20],
                        [20, 30],
                        [30, 10],
                        [310, 120],
                        [320, 130],
                        [330, 110]])
    >>> neighbors_k_order(pairs, 0, 2)
    [[array([0]), 0],
     [array([10, 20, 30]), 1],
     [array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    """
    
    # all_neigh stores all the unique neighbors and their oder
    all_neigh = [[np.array([n]), 0]]
    unique_neigh = np.array([n])
    
    for k in range(order):
        # detected neighbor nodes at the previous order
        last_neigh = all_neigh[k][0]
        k_neigh = []
        for node in last_neigh:
            # aggregate arrays of neighbors for each previous order neighbor
            neigh = np.unique(neighbors(pairs, node))
            k_neigh.append(neigh)
        # aggregate all unique kth order neighbors
        if len(k_neigh) > 0:
            k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
            # select the kth order neighbors that have never been detected in previous orders
            keep_neigh = np.in1d(k_unique_neigh, unique_neigh, invert=True)
            k_unique_neigh = k_unique_neigh[keep_neigh]
            # register the kth order unique neighbors along with their order
            all_neigh.append([k_unique_neigh, k+1])
            # update array of unique detected neighbors
            unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        else:
            break
        
    return all_neigh

def flatten_neighbors(all_neigh):
    """
    Convert the list of neighbors 1D arrays with their order into
    a single 1D array of neighbors.

    Parameters
    ----------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order.

    Returns
    -------
    flat_neigh : array_like
        The indices of neighboring nodes.
        
    Examples
    --------
    >>> all_neigh = [[np.array([0]), 0],
                     [np.array([10, 20, 30]), 1],
                     [np.array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    >>> flatten_neighbors(all_neigh)
    array([  0,  10,  20,  30, 110, 120, 130, 210, 220, 230, 310, 320, 330])
        
    Notes
    -----
    For future features it should return a 2D array of
    nodes and their respective order.
    """
    
    list_neigh = []
#     list_order = []
    for neigh, order in all_neigh:
        list_neigh.append(neigh)
#         list_order.append(np.ones(neigh.size) * order)
    flat_neigh = np.concatenate(list_neigh, axis=0)
#     flat_order = np.concatenate(list_order, axis=0)
#     flat_neigh = np.vstack([flat_neigh, flat_order]).T

    return flat_neigh

def aggregate_k_neighbors(X, pairs, order=1, var_names=None):
    """
    Compute the statistics on aggregated variables across
    the k order neighbors of each node in a network.

    Parameters
    ----------
    X : array_like
        The data on which to compute statistics (mean, std, ...).
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    order : int
        Max order of neighbors.
    var_names : list
        Names of variables of X.

    Returns
    -------
    aggregg : dataframe
        Statistics of X.
        
    Examples
    --------
    >>> pairs = vor.ridge_points[selection,:]
    >>> genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=2)
    """
    
    nb_obs = X.shape[0]
    nb_var = X.shape[1]
    aggreg = np.zeros((nb_obs, nb_var*2)) # *2 because mean and variance are stored

    for i in range(nb_obs):
        all_neigh = neighbors_k_order(pairs, n=i, order=order)
        neigh = flatten_neighbors(all_neigh)
        aggreg[i,:nb_var] = X[neigh,:].mean(axis=0)
        aggreg[i,-nb_var:] = X[neigh,:].std(axis=0)
    
    if var_names is None:
        var_names = [str(i) for i in range(nb_var)]
    columns = [var + ' mean' for var in var_names] +\
              [var + ' std' for var in var_names]
    aggreg = pd.DataFrame(data=aggreg, columns=columns)
    
    return aggreg

def make_cluster_cmap(labels, grey_pos='start'):
    """
    Creates an appropriate colormap for a vector of cluster labels.
    
    Parameters
    ----------
    labels : array_like
        The labels of multiple clustered points
    grey_pos: str
        Where to put the grey color for the noise
    
    Returns
    -------
    cmap : matplotlib colormap object
        A correct colormap
    
    Examples
    --------
    >>> my_cmap = make_cluster_cmap(labels=np.array([-1,3,5,2,4,1,3,-1,4,2,5]))
    """
    
    from matplotlib.colors import ListedColormap
    
    if labels.max() < 9:
        cmap = list(plt.get_cmap('tab10').colors)
        if grey_pos == 'end':
            cmap.append(cmap.pop(-3))
        elif grey_pos == 'start':
            cmap = [cmap.pop(-3)] + cmap
        elif grey_pos == 'del':
            del cmap[-3]
    else:
        cmap = list(plt.get_cmap('tab20').colors)
        if grey_pos == 'end':
            cmap.append(cmap.pop(-6))
            cmap.append(cmap.pop(-6))
        elif grey_pos == 'start':
            cmap = [cmap.pop(-5)] + cmap
            cmap = [cmap.pop(-5)] + cmap
        elif grey_pos == 'del':
            del cmap[-5]
            del cmap[-5]
    cmap = ListedColormap(cmap)
    
    return cmap


# %% [markdown]
# #### Visualization of exemplary network

# %%
pairs = np.array([[0, 10],
                  [0, 20],
                  [0, 30],
                  [10, 110],
                  [10, 210],
                  [10, 310],
                  [20, 120],
                  [20, 220],
                  [20, 320],
                  [30, 130],
                  [30, 230],
                  [30, 330],
                  [10, 20],
                  [20, 30],
                  [30, 10],
                  [310, 120],
                  [320, 130],
                  [330, 110]])

from collections import OrderedDict 
pos = OrderedDict([
                   (0, [0, 0]),
                   (10, [-1, 0]),
                   (20, [0, 1]),
                   (30, [1, 0]),
                   (110, [-1, -0.5]),
                   (210, [-1.5, 0]),
                   (310, [-1, 0.5]),
                   (120, [-0.5, 1]),
                   (220, [0, 1.5]),
                   (320, [0.5, 1]),
                   (130, [1, 0.5]),
                   (230, [1.5, 0]),
                   (330, [1, -0.5])
                  ])

order = [0] + [1] * 3 + [2] * 9
order_cmap = cmap = list(plt.get_cmap('tab10').colors)
order_color = [order_cmap[x] for x in order]

fig, ax = plt.subplots(figsize=[10, 10])
for a, b in pairs:
    xa, ya = pos[a]
    xb, yb = pos[b]
    ax.plot([xa, xb], [ya, yb], c='k',zorder=0, alpha=0.5)
for i, (x, y) in enumerate(pos.values()):
    ax.scatter(x, y, c=order_color[i], marker='o', s=40, zorder=10)
    
title = "exemplary network with neighbors orders"
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %%
all_neigh = neighbors_k_order(pairs, 0, 6)
all_neigh

# %%
flatten_neighbors(all_neigh)

# %% [markdown]
# #### First order neighbors

# %%
pairs = vor.ridge_points[selection,:]
genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=1, var_names=seqFISH_drop.columns)

# %%
# For the visualization
marker = 'o'
size_points = 10

# %%
reducer = umap.UMAP(random_state=0)
embedding = reducer.fit_transform(genes_aggreg)
embedding.shape

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='royalblue', marker=marker, s=size_points)
title = "Aggregated neighbors' genes data"
plt.title(title, fontsize=18);
# plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# It looks like we can define some clusters :)

# %% [markdown]
# ### Neighboors aggregated genes clustering

# %% [markdown]
# Now we can use our favorite clustering algorithm to find groups of similar points: HAC, OPTICS or HDBSCAN for instance.

# %% [markdown]
# #### HDBSCAN

# %%
clusterer = hdbscan.HDBSCAN(metric='euclidean')
clusterer.fit(genes_aggreg)
labels_hdbs = clusterer.labels_
nb_clust_hdbs = labels_hdbs.max() + 1
print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

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
clust = OPTICS()
# Run the fit
clust.fit(genes_aggreg)
labels_opt = clust.labels_
nb_clust_opt = labels_opt.max() + 1
print(f"OPTICS has detected {nb_clust_opt} clusters")

# %%
class_id, class_count = np.unique(labels_opt, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# %% [markdown]
# Most of points are classified as `-1`, which mean noise, which is bad!

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=labels_opt,
            cmap=make_cluster_cmap(labels_opt),
            marker=marker,
            s=size_points)
title = "Aggregated neighbors' genes data - OPTICS"
plt.title(title, fontsize=18);
# plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# That is not much better!  
#
# We should perform the clusterization on the reduced space, although it should be done with a lot of precautions (distances are not straighforwardly interpretable)

# %% [markdown]
# #### HDBSCAN on reduced space

# %% [markdown]
# UMAP projection with parameters adapted for clustering

# %%
embedding = umap.UMAP(n_neighbors=30,
                      min_dist=0.0,
                      n_components=2,
                      random_state=0,
                      ).fit_transform(genes_aggreg)

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='royalblue', marker=marker, s=size_points)
title = "Aggregated neighbors' genes data"
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %%
clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=30, min_samples=1)
clusterer.fit(embedding)
labels_hdbs = clusterer.labels_
nb_clust_hdbs = labels_hdbs.max() + 1
print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

# %% [markdown]
# we choose `min_samples=1` to avoid having points considered as noise

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=labels_hdbs,
            s=5,
            cmap=make_cluster_cmap(labels_hdbs));

title = "HDBSCAN clustering on aggregated neighbors' genes data"
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %%
class_id, class_count = np.unique(labels_hdbs, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# %% [markdown]
# This is pretty good :)  
# Of course one can tweak the parameters to obtain a clustering that fits him better.

# %% [markdown]
# #### OPTICS on reduced space

# %%
clust = OPTICS(min_cluster_size=50)
# Run the fit
clust.fit(embedding)

labels_opt = clust.labels_
nb_clust_opt = labels_opt.max() + 1
print(f"OPTICS has detected {nb_clust_opt} clusters")

# %%
class_id, class_count = np.unique(clust.labels_, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=labels_opt,
            cmap=make_cluster_cmap(labels_opt),
            marker=marker,
            s=size_points)

title = "OPTICS clustering on aggregated neighbors' genes data"
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# HDBSCAN provides a much better clustering regarding the data projection.

# %% [markdown]
# #### Visualisation of spatial seqFISH data and detected areas 

# %%
fig, ax = plt.subplots(1, 3, figsize=(20,7), tight_layout=True)

for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=14);
ax[0].legend()

ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[2].scatter(embedding[:, 0], embedding[:, 1], c=labels_hdbs, s=5, cmap=make_cluster_cmap(labels_hdbs));
ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

title = "spatial seqFISH data and detected areas"
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# The detected areas look plausible as points affected to different area types are not randomly dispersed.  
# Moreover will see with future configurations that the detected areas span over areas of some phenotypes or form regions smaller than areas of some phenotypes.

# %% [markdown]
# #### Screening of order and clustering

# %%
clf_name = 'Kernel SVC'
nb_genes = scRNAseq_drop.shape[1]
save_dir = Path(f'../data/processed/nb_genes {nb_genes} - {clf_name}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

orders = [1, 2, 3, 4]

for order in orders:
    # compute statistics on aggregated data across neighbors
    genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=order, var_names=seqFISH_drop.columns)
    
    # Dimension reduction for visualization
    embed_viz = umap.UMAP(n_components=2, random_state=0).fit_transform(genes_aggreg)

    for dim_clust in [2,3,4,5,6,7,8,9]:
        # Dimension reduction for clustering
        embed_clust = umap.UMAP(n_neighbors=30,
                                min_dist=0.0,
                                n_components=dim_clust,
                                random_state=0,
                               ).fit_transform(genes_aggreg)

        for min_cluster_size in [10,20,30,40,50,60]:
            clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=1)
            clusterer.fit(embed_clust)
            labels_hdbs = clusterer.labels_
            nb_clust_hdbs = labels_hdbs.max() + 1
            print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

            # Histogram of classes
            fig = plt.figure()
            class_id, class_count = np.unique(labels_hdbs, return_counts=True)
            plt.bar(class_id, class_count, width=0.8);
            plt.title('Clusters histogram');
            title = f"Clusters histogram - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
            plt.savefig(save_dir / title, bbox_inches='tight')
            plt.show()

            # Big summary plot
            fig, ax = plt.subplots(1, 3, figsize=(20,7), tight_layout=False)

            for class_pred, label, color in color_labels:
                select = class_pred == y_pred
                ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
            ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=14);
            ax[0].legend()

            ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
            ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

            ax[2].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_hdbs, s=5, cmap=make_cluster_cmap(labels_hdbs));
            ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

            suptitle = f"Spatial seqFISH data and detected areas - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
            fig.suptitle(suptitle, fontsize=14)
            fig.savefig(save_dir / suptitle, bbox_inches='tight')
            plt.show()

# %% [markdown]
# Some config that are interesting, for analyzes or for the presentation:
#
# | order | clust_dim | min_size |            remark            |
# |:-----:|:---------:|:--------:|:----------------------------:|
# |   1   |     5     |    30    |                              |
# |   1   |     6     |    40    |           or 50, 60          |
# |   1   |     9     |    30    |                              |
# |   2   |     2     |    40    |                              |
# |   2   |     4     |    30    |                              |
# |   2   |     5     |    60    |                              |
# |   2   |     8     |    40    | to explain link between maps |
# |   3   |     2     |    40    |                              |
# |   3   |     4     |    40    |                              |
# |   3   |     66    |    40    |                              |
# |   4   |     3     |    60    |        to show oder 4        |
# |   4   |     7     |    50    |             idem             |

# %% [markdown]
# #### Summary image for the white paper

# %%
pairs = vor.ridge_points[selection,:]
marker = 'o'
size_points = 10

clf_name = 'Kernel SVC'
nb_genes = scRNAseq_drop.shape[1]

order = 2
dim_clust = 2
min_cluster_size = 40

# compute statistics on aggregated data across neighbors
genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=order, var_names=seqFISH_drop.columns)

# Dimension reduction for visualization
embed_viz = umap.UMAP(n_components=2, random_state=0).fit_transform(genes_aggreg)


# Dimension reduction for clustering
embed_clust = umap.UMAP(n_neighbors=30,
                        min_dist=0.0,
                        n_components=dim_clust,
                        random_state=0,
                       ).fit_transform(genes_aggreg)


clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=1)
clusterer.fit(embed_clust)
labels_hdbs = clusterer.labels_
nb_clust_hdbs = labels_hdbs.max() + 1
print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

# Big summary plot
fig, ax = plt.subplots(1, 3, figsize=(25,9), tight_layout=True)

# Network
for points in voro_cells[selection,:]:
    ax[0].plot(points[[0,2]],points[[1,3]],zorder=0, c='k', alpha=0.7, linewidth=0.5)
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker=marker, s=size_points, zorder=10)
ax[0].set_title('Spatial network of predicted phenotypes', fontsize=14);
ax[0].axis('off')
ax[0].legend()

# Detected areas
ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
ax[1].set_title('Spatial map of detected areas', fontsize=14);
ax[1].axis('off')

# HDBSCAN clustering on UMAP projection
ax[2].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_hdbs, marker=marker, s=size_points, cmap=make_cluster_cmap(labels_hdbs));
ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);
ax[2].axis('off')

suptitle = f"Areas detection from seqFISH spatial network - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
# fig.suptitle(suptitle, fontsize=14)
fig.savefig("../data/processed/"+suptitle, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Differential Expression analysis

# %% [markdown]
# Accross the diverse configurations, there are regions that appear consistently.  
# One of them is curious, on the spatial map it's a spot in the middle of a huge class of cells.  
# To analyse why these cells stand out in several configuration we will use the config: `order=1, dim_clust=5, min_clust_size=30`, but other config are possible too.

# %%
# For the visualization
marker = 'o'
size_points = 10

order = 1
nb_genes = seqFISH_drop.shape[0]
clf_name = 'Kernel SVC'

# compute statistics on aggregated data across neighbors
genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=order, var_names=seqFISH_drop.columns)

# Dimension reduction for visualization
embed_viz = umap.UMAP(n_components=2, random_state=0).fit_transform(genes_aggreg)

dim_clust = 5
# Dimension reduction for clustering
embed_clust = umap.UMAP(n_neighbors=30,
                        min_dist=0.0,
                        n_components=dim_clust,
                        random_state=0,
                       ).fit_transform(genes_aggreg)

min_cluster_size = 30
clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=1)
clusterer.fit(embed_clust)
labels_hdbs = clusterer.labels_
nb_clust_hdbs = labels_hdbs.max() + 1
print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

# %%
# Histogram of classes
fig = plt.figure()
class_id, class_count = np.unique(labels_hdbs, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# Big summary plot
fig, ax = plt.subplots(1, 3, figsize=(20,7), tight_layout=False)

for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=14);
ax[0].legend()

ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[2].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_hdbs, s=5, cmap=make_cluster_cmap(labels_hdbs));
ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

suptitle = f"Spatial seqFISH data and detected areas - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}";
fig.suptitle(suptitle, fontsize=14)

# %%
import scanpy as sc

# %%
adata = sc.AnnData(genes_aggreg)
# adata.obs['cell_clusters'] = labels_hdbs
adata.obs['cell_clusters'] = pd.Series(labels_hdbs, dtype="category")

# %%
sc.pl.highest_expr_genes(adata, n_top=20)

# %%
sc.tl.rank_genes_groups(adata, groupby='cell_clusters', method='t-test')

# %%
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# %% [markdown]
# I have issue with Scanpy and Anndata

# %% [markdown]
# ### Home made DE analysis

# %% [markdown]
# Are the values normaly distributed?

# %%
genes_aggreg.hist(bins=50, figsize=(18,15));

# %% [markdown]
# It would be nice to display the histogram of each cluster per gene, but no time!

# %%
clust_id = 2
clust_targ = labels_hdbs == clust_id  # cluster of interest (target)
clust_comp = labels_hdbs != clust_id  # cluster(s) we compare with

fig, ax = plt.subplots(1, 2, figsize=(14,7), tight_layout=False)

ax[0].scatter(seqFISH_coords.loc[clust_targ,'x'], seqFISH_coords.loc[clust_targ,'y'], c='tomato', marker=marker, s=size_points)
ax[0].scatter(seqFISH_coords.loc[clust_comp,'x'], seqFISH_coords.loc[clust_comp,'y'], c='lightgrey', marker=marker, s=size_points)
ax[0].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[1].scatter(embed_viz[clust_targ, 0], embed_viz[clust_targ, 1], s=5, c='tomato');
ax[1].scatter(embed_viz[clust_comp, 0], embed_viz[clust_comp, 1], s=5, c='lightgrey');
ax[1].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

# %%
clust_id = 3
clust_targ = labels_hdbs == clust_id  # cluster of interest (target)
clust_comp = labels_hdbs != clust_id  # cluster(s) we compare with

fig, ax = plt.subplots(1, 2, figsize=(14,7), tight_layout=False)

ax[0].scatter(seqFISH_coords.loc[clust_targ,'x'], seqFISH_coords.loc[clust_targ,'y'], c='tomato', marker=marker, s=size_points)
ax[0].scatter(seqFISH_coords.loc[clust_comp,'x'], seqFISH_coords.loc[clust_comp,'y'], c='lightgrey', marker=marker, s=size_points)
ax[0].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[1].scatter(embed_viz[clust_targ, 0], embed_viz[clust_targ, 1], s=5, c='tomato');
ax[1].scatter(embed_viz[clust_comp, 0], embed_viz[clust_comp, 1], s=5, c='lightgrey');
ax[1].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

# %% [markdown]
# So we will compare cluster 2 vs the rest, and cluster 2 vs cluster 3.

# %%
from scipy.stats import ttest_ind    # Welch's t-test
from scipy.stats import mannwhitneyu # Mann-Whitney rank test
from scipy.stats import ks_2samp     # Kolmogorov-Smirnov statistic

var_names = genes_aggreg.columns
distrib_pval = {'Welch': [],
                'Mann-Whitney': [],
                'Kolmogorov-Smirnov': []}
select_1 = labels_hdbs == 2
select_2 = labels_hdbs == 3
for var_name in var_names:
    dist1 = genes_aggreg.loc[select_1, var_name]
    dist2 = genes_aggreg.loc[select_2, var_name]
    w_stat, w_pval = ttest_ind(dist1, dist2, equal_var=False)
    mwu_stat, mwu_pval = mannwhitneyu(dist1, dist2)
    ks_stat, ks_pval = ks_2samp(dist1, dist2)
    distrib_pval['Welch'].append(w_pval)
    distrib_pval['Mann-Whitney'].append(mwu_pval)
    distrib_pval['Kolmogorov-Smirnov'].append(ks_pval)
DE_pval = pd.DataFrame(distrib_pval, index=var_names)


# %%
def highlight_under(s, thresh=0.05, color='darkorange'):
    '''
    highlight values that are under a threshold
    '''
    is_under = s <= thresh
    attr = 'background-color: {}'.format(color)
    return [attr if v else '' for v in is_under]


# %%
DE_pval.T.style.apply(highlight_under)

# %%
diff_var = DE_pval.loc[DE_pval['Kolmogorov-Smirnov'] <= 0.05, 'Kolmogorov-Smirnov'].sort_values()
diff_var

# %%
diff_var_set = set([var.replace(' mean', '').replace(' std','') for var in diff_var.index])
for var in diff_var_set:
    print(var)

# %%
for var in seqFISH_drop.columns:
    print(var)

# %% [markdown]
# ## Whole analysis with all seqFISH genes

# %%
gene_names = list(scRNAseq.columns)

scaler_sc = StandardScaler()  # for scRNAseq
scaler_seq = StandardScaler() # for seqFISH
Xtest = scaler_sc.fit_transform(scRNAseq)
Xpred = scaler_seq.fit_transform(seqFISH)  
nb_genes = Xpred.shape[1]
print(f"There are {nb_genes} remaining genes")

# %% [markdown]
# #### Predict phenotypes of seqFISH cells - Linear SVC

# %%
model = 'Linear SVC'

param_search = pd.read_csv('../data/processed/grid_search_cv_results-2020-06-09_18h36.csv')
scoring = 'balanced_accuracy'
best = param_search.loc[param_search['rank '+scoring]==1]
C = best['C'].values[0]

clf = LinearSVC(C=C, class_weight='balanced')
clf.fit(Xtest, y_true)
y_pred = clf.predict(Xpred)

# %%
pheno_id, counts = np.unique(y_pred, return_counts=True)
pheno_names = le.inverse_transform(pheno_id)
pd.DataFrame(data={'phenotype':pheno_names,
                   'counts':counts},
             index=pheno_id)

# %%
base_cmap = sns.color_palette('muted').as_hex() # plt.get_cmap("Set1")

# make a custom colormap with class integers, labels and hex colors like
# [[0, 'Astrocyte', '#023eff'],
#  [2, 'GABA-ergic Neuron', '#ff7c00'],
#  [3, 'Glutamatergic Neuron', '#1ac938']]
# 
color_labels = []
for i, i_pred in enumerate(np.unique(y_pred)):
    color_labels.append([i_pred, le.inverse_transform([i_pred])[0], base_cmap[i]])

# more custom colormap, switch to previous line if any issue like different class integers
# color_labels = [[0, le.inverse_transform([0])[0], base_cmap[0]],
#                 [2, le.inverse_transform([2])[0], base_cmap[1]],
#                 [3, le.inverse_transform([3])[0], base_cmap[2]]]

fig, ax = plt.subplots(figsize=[10,10])
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=10)
plt.legend()

title = f"Map of predicted seqFISH cell types - {model} - {nb_genes} genes"
plt.title(title, fontsize=14);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# #### Predict phenotypes of seqFISH cells - Kernel SVC

# %%
model = 'Kernel SVC'

param_search = pd.read_csv('../data/processed/random_search_cv_zoom_results-2020-05-09_09h50.csv')
scoring = 'balanced_accuracy'
best = param_search.loc[param_search['rank '+scoring]==1]
C = best['C'].values[0]
gamma = best['gamma'].values[0]

clf = SVC(C=C, gamma=gamma, class_weight='balanced')
clf.fit(Xtest, y_true)
y_pred = clf.predict(Xpred)

# %%
pheno_id, counts = np.unique(y_pred, return_counts=True)
pheno_names = le.inverse_transform(pheno_id)
pd.DataFrame(data={'phenotype':pheno_names,
                   'counts':counts},
             index=pheno_id)

# %%
base_cmap = sns.color_palette('muted').as_hex() # plt.get_cmap("Set1")

# make a custom colormap with class integers, labels and hex colors like
# [[0, 'Astrocyte', '#023eff'],
#  [2, 'GABA-ergic Neuron', '#ff7c00'],
#  [3, 'Glutamatergic Neuron', '#1ac938']]
# 
color_labels = []
for i, i_pred in enumerate(np.unique(y_pred)):
    color_labels.append([i_pred, le.inverse_transform([i_pred])[0], base_cmap[i]])

# more custom colormap, switch to previous line if any issue like different class integers
# color_labels = [[0, le.inverse_transform([0])[0], base_cmap[0]],
#                 [2, le.inverse_transform([2])[0], base_cmap[1]],
#                 [3, le.inverse_transform([3])[0], base_cmap[2]]]

fig, ax = plt.subplots(figsize=[10,10])
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=10)
plt.legend()

title = f"Map of predicted seqFISH cell types - {model} - {nb_genes} genes"
plt.title(title, fontsize=14);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# ### Network construction

# %%
from scipy.spatial import Voronoi

vor = Voronoi(seqFISH_coords[['x','y']])

# arrays of x0, y0, x1, y1
voro_cells = np.zeros((vor.ridge_points.shape[0],4))
voro_cells[:,[0,1]] = seqFISH_coords.loc[vor.ridge_points[:,0], ['x','y']]
voro_cells[:,[2,3]] = seqFISH_coords.loc[vor.ridge_points[:,1], ['x','y']]
distances = np.sqrt((voro_cells[:,0]-voro_cells[:,2])**2+(voro_cells[:,1]-voro_cells[:,3])**2)

# %%
# distance threshold to discard edges above it
#  mainly artifacts at the borders of the whole dataset
EDGE_DIST_THRESH = 300 
selection = distances < EDGE_DIST_THRESH
pairs = vor.ridge_points[selection,:]

fig, ax = plt.subplots(figsize=[15, 15])
for points in voro_cells[selection,:]:
    ax.plot(points[[0,2]],points[[1,3]], c='k',zorder=0, alpha=0.5)
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
plt.legend()

title = f'Spatial network of seqFISH data - {model} - {nb_genes} genes'
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# ### Screening of order and clustering

# %%
clf_name = 'Kernel SVC'

save_dir = Path(f'../data/processed/nb_genes {nb_genes} - {clf_name}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

orders = [1, 2, 3, 4]

for order in orders:
    # compute statistics on aggregated data across neighbors
    genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=order, var_names=seqFISH.columns)
    
    # Dimension reduction for visualization
    embed_viz = umap.UMAP(n_components=2, random_state=0).fit_transform(genes_aggreg)

    for dim_clust in [2,3,4,5,6,7,8,9]:
        # Dimension reduction for clustering
        embed_clust = umap.UMAP(n_neighbors=30,
                                min_dist=0.0,
                                n_components=dim_clust,
                                random_state=0,
                               ).fit_transform(genes_aggreg)

        for min_cluster_size in [10,20,30,40,50,60]:
            clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=1)
            clusterer.fit(embed_clust)
            labels_hdbs = clusterer.labels_
            nb_clust_hdbs = labels_hdbs.max() + 1
            print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

            # Histogram of classes
            fig = plt.figure()
            class_id, class_count = np.unique(labels_hdbs, return_counts=True)
            plt.bar(class_id, class_count, width=0.8);
            plt.title('Clusters histogram');
            title = f"Clusters histogram - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
            plt.savefig(save_dir / title, bbox_inches='tight')
            plt.show()

            # Big summary plot
            fig, ax = plt.subplots(1, 3, figsize=(22,7), tight_layout=False)

            for class_pred, label, color in color_labels:
                select = class_pred == y_pred
                ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=20, zorder=10)
            ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=14);
            ax[0].legend()

            ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
            ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

            ax[2].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_hdbs, s=5, cmap=make_cluster_cmap(labels_hdbs));
            ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

            suptitle = f"Spatial seqFISH data and detected areas - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
            fig.suptitle(suptitle, fontsize=14)
            fig.savefig(save_dir / suptitle, bbox_inches='tight')
            plt.show()

# %% [markdown]
# ### DE analysis

# %% [markdown]
# We choose for "DE analysis" this configuration: `order=1, dim_clust=3, min_clust_size=30`

# %%
# For the visualization
marker = 'o'
size_points = 10

order = 1
nb_genes = seqFISH.shape[1]
clf_name = 'Kernel SVC'

# compute statistics on aggregated data across neighbors
genes_aggreg = aggregate_k_neighbors(X=Xpred, pairs=pairs, order=order, var_names=seqFISH.columns)

# Dimension reduction for visualization
embed_viz = umap.UMAP(n_components=2, random_state=0).fit_transform(genes_aggreg)

dim_clust = 3
# Dimension reduction for clustering
embed_clust = umap.UMAP(n_neighbors=30,
                        min_dist=0.0,
                        n_components=dim_clust,
                        random_state=0,
                       ).fit_transform(genes_aggreg)

min_cluster_size = 30
clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=1)
clusterer.fit(embed_clust)
labels_hdbs = clusterer.labels_
nb_clust_hdbs = labels_hdbs.max() + 1
print(f"HDBSCAN has detected {nb_clust_hdbs} clusters")

# %%
# Histogram of classes
fig = plt.figure()
class_id, class_count = np.unique(labels_hdbs, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# Big summary plot
fig, ax = plt.subplots(1, 3, figsize=(22,7), tight_layout=False)

for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax[0].scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=10, zorder=10)
ax[0].set_title('Spatial map of prediction of phenotypes for seqFISH data', fontsize=14);
ax[0].legend()

ax[1].scatter(seqFISH_coords.loc[:,'x'], seqFISH_coords.loc[:,'y'], c=labels_hdbs, cmap=make_cluster_cmap(labels_hdbs), marker=marker, s=size_points)
ax[1].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[2].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_hdbs, s=5, cmap=make_cluster_cmap(labels_hdbs));
ax[2].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

suptitle = f"Spatial seqFISH data and detected areas - nb_genes {nb_genes} - {clf_name} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}";
fig.suptitle(suptitle, fontsize=14)

# %% [markdown]
# #### Home made DE analysis

# %% [markdown]
# Are the values normaly distributed?

# %%
genes_aggreg.hist(bins=50, figsize=(18,15));

# %% [markdown]
# It would be nice to display the histogram of each cluster per gene, but no time!

# %%
for clust_id in range(19):
    clust_targ = labels_hdbs == clust_id  # cluster of interest (target)
    clust_comp = labels_hdbs != clust_id  # cluster(s) we compare with

    fig, ax = plt.subplots(1, 2, figsize=(14,7), tight_layout=False)

    ax[0].scatter(seqFISH_coords.loc[clust_targ,'x'], seqFISH_coords.loc[clust_targ,'y'], c='tomato', marker=marker, s=size_points)
    ax[0].scatter(seqFISH_coords.loc[clust_comp,'x'], seqFISH_coords.loc[clust_comp,'y'], c='lightgrey', marker=marker, s=size_points)
    ax[0].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

    ax[1].scatter(embed_viz[clust_targ, 0], embed_viz[clust_targ, 1], s=5, c='tomato');
    ax[1].scatter(embed_viz[clust_comp, 0], embed_viz[clust_comp, 1], s=5, c='lightgrey');
    ax[1].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);
    plt.suptitle(f"clust_id = {clust_id}")
    plt.show()

# %%
clust_id = 17
clust_targ = labels_hdbs == clust_id  # cluster of interest (target)
clust_comp = labels_hdbs != clust_id  # cluster(s) we compare with

fig, ax = plt.subplots(1, 2, figsize=(14,7), tight_layout=False)

ax[0].scatter(seqFISH_coords.loc[clust_targ,'x'], seqFISH_coords.loc[clust_targ,'y'], c='tomato', marker=marker, s=size_points)
ax[0].scatter(seqFISH_coords.loc[clust_comp,'x'], seqFISH_coords.loc[clust_comp,'y'], c='lightgrey', marker=marker, s=size_points)
ax[0].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[1].scatter(embed_viz[clust_targ, 0], embed_viz[clust_targ, 1], s=5, c='tomato');
ax[1].scatter(embed_viz[clust_comp, 0], embed_viz[clust_comp, 1], s=5, c='lightgrey');
ax[1].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

# %%
clust_ids = [8, 18]
clust_targ = pd.Series(labels_hdbs).isin(clust_ids)  # clusters of interest (target)
clust_comp = ~clust_targ  # cluster(s) we compare with

fig, ax = plt.subplots(1, 2, figsize=(14,7), tight_layout=False)

ax[0].scatter(seqFISH_coords.loc[clust_targ,'x'], seqFISH_coords.loc[clust_targ,'y'], c='tomato', marker=marker, s=size_points)
ax[0].scatter(seqFISH_coords.loc[clust_comp,'x'], seqFISH_coords.loc[clust_comp,'y'], c='lightgrey', marker=marker, s=size_points)
ax[0].set_title('Spatial map of detected areas for seqFISH data', fontsize=14);

ax[1].scatter(embed_viz[clust_targ, 0], embed_viz[clust_targ, 1], s=5, c='tomato');
ax[1].scatter(embed_viz[clust_comp, 0], embed_viz[clust_comp, 1], s=5, c='lightgrey');
ax[1].set_title("HDBSCAN clustering on aggregated neighbors' genes data", fontsize=14);

# %% [markdown]
# So we will compare cluster 17 vs the rest, and cluster 2 vs clusters 8 and 18.

# %%
from scipy.stats import ttest_ind    # Welch's t-test
from scipy.stats import mannwhitneyu # Mann-Whitney rank test
from scipy.stats import ks_2samp     # Kolmogorov-Smirnov statistic

var_names = genes_aggreg.columns
distrib_pval = {'Welch': [],
                'Mann-Whitney': [],
                'Kolmogorov-Smirnov': []}
select_1 = labels_hdbs == 17
clust_ids = [8, 18]
select_2 = pd.Series(labels_hdbs).isin(clust_ids) 
for var_name in var_names:
    dist1 = genes_aggreg.loc[select_1, var_name]
    dist2 = genes_aggreg.loc[select_2, var_name]
    w_stat, w_pval = ttest_ind(dist1, dist2, equal_var=False)
    mwu_stat, mwu_pval = mannwhitneyu(dist1, dist2)
    ks_stat, ks_pval = ks_2samp(dist1, dist2)
    distrib_pval['Welch'].append(w_pval)
    distrib_pval['Mann-Whitney'].append(mwu_pval)
    distrib_pval['Kolmogorov-Smirnov'].append(ks_pval)
DE_pval = pd.DataFrame(distrib_pval, index=var_names)
DE_pval.sort_values(by='Kolmogorov-Smirnov', inplace=True)


# %%
def highlight_under(s, thresh=0.05, color='darkorange'):
    '''
    highlight values that are under a threshold
    '''
    is_under = s <= thresh
    attr = 'background-color: {}'.format(color)
    return [attr if v else '' for v in is_under]


# %%
DE_pval.T.style.apply(highlight_under)

# %%
DE_pval.style.apply(highlight_under)

# %%
diff_var = DE_pval.loc[DE_pval['Kolmogorov-Smirnov'] <= 0.005, 'Kolmogorov-Smirnov']
diff_var_set = set([var.replace(' mean', '').replace(' std','') for var in diff_var.index])
print(f"Set of genes of size {len(diff_var_set)}")

# %%
# to copy-paste "DE" genes in a GO tool
for var in diff_var_set:
    print(var)

# %%
# to copy-paste genes in a GO tool
for var in seqFISH.columns:
    print(var)

# %% [markdown]
# ## Conclusion

# %% [markdown]
# We have seen that it is possible to assign to seqFISH data points their corresponding phenotypes defined from the scRNAseq data, with only 19 genes.  
#
# Moreover for seqFISH data aggregating gene expression for each node and it's neighbors we have found different groups, which migh correspond to areas of cell of different proportions in phenotypes.  
# It would be interesting to check that in a further analysis.  
#
# An interesting lead could be, for each cell, retrieve the mean values of its corresponding phenotype (the 'signature' of the phenotype), and then run again an aggregated neighbors' gene expression analysis. That could emphasise the genes that are under or over expressed due to the localisation of the cells and eliminate the strong contributions of genes that are specific of cell type.

# %%
