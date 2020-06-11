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

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
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
# ### seqFISH coordinates

# %% [markdown]
# Spatial cell coordinates

# %%
seqFISH_coords = pd.read_csv(seqFISH_coords_path, sep=' ', header=None, usecols=[2,3], names=['x','y'])
seqFISH_coords.head()

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
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

# %% [markdown]
# On the reduced space of the original data (left figure), the clusterization doesn't seem optimal, some groups are spread among different 'clouds' of points and some points are in the middle of another group.  
# The Yeo-Johnson doesn't help much as expected from the previous section.  
# The centered logistic transformation outputs 2 clear clusters, but there are certainly artifacts from the transformation. I actually selected this transformation because on the histograms of the selected genes we can see that the centered logistic transformation pushes data into 2 seperate groups.  
# But for exploratory analysis for this challenge we stick to the cluster definition of *Tasic et al.*

# %% [markdown]
# **Conclusion**  
# We will simply work on the data given by the workshop organizers as gene expression data are already well processed.

# %% [markdown] toc-hr-collapsed=true toc-nb-collapsed=true toc-hr-collapsed=true toc-nb-collapsed=true
# ## Map non spatial scRNAseq data to spatial seqFISH data

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

cv_search= cv_to_df(grid_search.cv_results_, scores=['balanced_accuracy','accuracy'], order_by='mean balanced_accuracy')
cv_search.to_csv(f'../data/processed/grid_search_cv_results-{timestamp}.csv', index=False)


# %% [markdown]
# #### Line hyperparameters scores

# %%
def hyperparam_line(results, param_x, score, n_top = 20, best_scores_coeff=500,
                   figsize = (10,10), best_cmap=sns.light_palette("green", as_cmap=True)):
    df = pd.DataFrame(results['params'])
    df['score'] = results[score]

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=param_x, y='score', data=df)
    ax.set_xscale('log')

    best = df.sort_values(by='score', ascending=False).iloc[:n_top,:]
    best_scores = best['score'] - best['score'].min()
    best_scores = best_scores / best_scores.max()

#     ax.scatter(x=best[param_x], y=best['score'], c=best_scores, s=best_scores*best_scores_coeff, marker='+', cmap=best_cmap)
#     ax.scatter(x=best[param_x].iloc[0], y=best[param_y].iloc[0], s=best_scores.iloc[0]*best_scores_coeff/2, marker='o', color='r', alpha=0.5)


# %%
hyperparam_line(grid_search.cv_results_, param_x = 'C', score = 'mean_test_balanced_accuracy', n_top = 20, figsize = (10,5))
title = 'CV balanced accuracy for Linear SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')
hyperparam_line(grid_search.cv_results_, param_x = 'C', score = 'mean_test_accuracy', n_top = 20, figsize = (10,5))
title = 'CV accuracy for Linear SVC hyperparameters search'
plt.title(title)
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %% [markdown]
# With LinearSVC as in Zhu's paper we can reach an accuracy of 0.9 too, but a balanced accuracy of 0.78 only.  
# We will stick to our kernel SVC as the balanced accuracy is a bit higher.

# %% [markdown]
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
plt.title('Balanced accuracy of scRNAseq classification during variables (genes) elimination');

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
plt.title('Accuracy of scRNAseq classification during variables (genes) elimination');

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
plt.title('Balanced accuracy of scRNAseq classification with Linear SVC during variables (genes) elimination');

# %% [markdown]
# #### elimination with accuracy

# %%
elimination_report = np.loadtxt("../data/processed/elimination_report_linearSVC-accuracy-2020-06-09_22h30.csv", skiprows=1, delimiter=',')

# %%
plt.figure(figsize=(14,8))
plt.plot(elimination_report[::-1,1]);
plt.xlabel('nb remaining variables + 1')
plt.ylabel('score')
plt.title('Accuracy of scRNAseq classification with Linear SVC during variables (genes) elimination');


# %% [markdown]
# These plots confirm that Zhu *et al.* probably used the simple accuracy instead of the balanced accuracy.  
# The kernel SVC require fewer genes (\~19-29) than the linear SVC (\~35-39) to maintain a relatively high score.

# %% [markdown]
# ## Infer cell types from restricted gene list

# %%
def successive_elimination(list_elems, elems_id, remaining=False):
    """
    Compute the list of elements that remain or that are eliminated 
    after successive deletions at indicated indices.
    
    Parameters
    ----------
    list_elems : list
        A list of elements that can be eliminated
    elems_id: list
        A list of indices where elements are successively deleted
    remaining: bool, default False
        If True the function returns a list of remaining elements
    
    Returns
    -------
    successive : list
        A list of deleted or remaining elements
    
    Examples
    --------
    >>> gene_names = list(scRNAseq.columns)
    >>> genes_elim_id = elimination_report[:93,0].astype(int)
    >>> genes_elim = successive_elimination(gene_names, genes_elim_id)
    """
    
    from copy import deepcopy
    elems = deepcopy(list_elems)
    
    if remaining:
        successive = elems
        for i in elems_id:
            del elems[i]
    else:
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
nb_elim = 93 # eliminate more genes, with still good performance
genes_elim_id = elimination_report[:nb_elim,0].astype(int)
genes_elim = successive_elimination(gene_names, genes_elim_id)

scRNAseq_drop = copy.deepcopy(scRNAseq)
seqFISH_drop = copy.deepcopy(seqFISH)

scRNAseq_drop.drop(columns=genes_elim, inplace=True)
seqFISH_drop.drop(columns=genes_elim, inplace=True)

scaler_sc = StandardScaler()  # for scRNAseq
scaler_seq = StandardScaler() # for seqFISH
Xtest = scaler_sc.fit_transform(scRNAseq_drop)
Xpred = scaler_seq.fit_transform(seqFISH_drop)  

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

clf = SVC(C=C, gamma=gamma)
clf.fit(Xtest, y_true)
y_pred = clf.predict(Xpred)

# %%
base_cmap = sns.color_palette('muted').as_hex() # plt.get_cmap("Set1")

# make a custom colormap with class integers, labels and hex colors like
# [[0, 'Astrocyte', '#023eff'],
#  [2, 'GABA-ergic Neuron', '#ff7c00'],
#  [3, 'Glutamatergic Neuron', '#1ac938']]
# 
# color_labels = []
# for i, i_pred in enumerate(np.unique(y_pred)):
#     color_labels.append([i_pred, le.inverse_transform([i_pred])[0], base_cmap[i]])

# more custom colormap, switch to previous line if any issue like different class integers
color_labels = [[0, le.inverse_transform([0])[0], base_cmap[0]],
                [2, le.inverse_transform([2])[0], base_cmap[1]],
                [3, le.inverse_transform([3])[0], base_cmap[2]]]

fig, ax = plt.subplots(figsize=[10,10])
for class_pred, label, color in color_labels:
    select = class_pred == y_pred
    ax.scatter(seqFISH_coords.loc[select,'x'], seqFISH_coords.loc[select,'y'], c=color, label=label, marker='o', s=10)
plt.legend()

title = 'Spatial map of prediction of phenotypes for seqFISH data'
plt.title(title, fontsize=18);
plt.savefig('../data/processed/'+title, bbox_inches='tight')

# %%
pheno_id, counts = np.unique(y_pred, return_counts=True)
pheno_names = le.inverse_transform(pheno_id)
pd.DataFrame(data={'phenotype':pheno_names,
                   'counts':counts},
             index=pheno_id)

# %% [markdown]
# There are only 3 cell types predicted, we are missing:
# - Endothelial Cell
# - Microglia
# - Oligodendrocyte.1
# - Oligodendrocyte.2
# - Oligodendrocyte.3

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
# #### Aggregation

# %%
# array to store aggregated data
nb_cells = Xpred.shape[0]
nb_genes = Xpred.shape[1]
genes_aggreg = np.zeros((nb_cells, nb_genes*2)) # *2 because mean and variance are stored
pair_points = vor.ridge_points[selection,:]

for i in range(nb_cells):
    # array of neighbors of a given node
    left_neigh = pair_points[pair_points[:,1] == i, 0]
    right_neigh = pair_points[pair_points[:,0] == i, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    if neigh.size != 0:
        genes_aggreg[i,:nb_genes] = Xtest[neigh,:].mean(axis=0)
        genes_aggreg[i,-nb_genes:] = Xtest[neigh,:].std(axis=0)
    else:
        genes_aggreg[i,:] = None

# the following lines were needed when I made the error of
# using Xtest instead of Xpred, the indices didn't match
error_cells = np.isnan(genes_aggreg[:,0])
nb_errors = error_cells.sum()
print(f"There has been {nb_errors}/{nb_cells} cells set to NaN")

# %%
neigh_valid = genes_aggreg[~error_cells,:]

# %% [markdown]
# #### Visualization

# %%
marker = 'o'
size_points = 10
colormap = 'tab10'

# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(neigh_valid)
embedding.shape

# %%
plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='royalblue', marker=marker, s=size_points)
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
clust = OPTICS()
# Run the fit
clust.fit(neigh_valid)

# %%
clust.labels_.max()

# %%
class_id, class_count = np.unique(clust.labels_, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# %% [markdown]
# Most of points are classified as `-1`, which mean noise, which is bad!

# %%
# Define cluster colors to highlight noise
from bokeh.palettes import Category10

nb_labels = clust.labels_.max()
palette = Category10[nb_labels]
palette_grey =  list(Category10[nb_labels])
palette_grey.append('#C0C0C0')
clust_colors = [palette_grey[x] for x in clust.labels_]

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c=clust_colors, marker=marker, s=size_points)
plt.title("Aggregated neighbors' genes data", fontsize=18);

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
                      random_state=42,
                      ).fit_transform(neigh_valid)

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c='royalblue', marker=marker, s=size_points)
plt.title("Overview of aggregated neighbors' genes data", fontsize=18);

# %%
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

# %%
class_id, class_count = np.unique(labels[clustered], return_counts=True)
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

clust.labels_.max()

# %%
class_id, class_count = np.unique(clust.labels_, return_counts=True)
plt.bar(class_id, class_count, width=0.8);
plt.title('Clusters histogram');

# %%
nb_labels = clust.labels_.max()
palette = Category10[nb_labels]
palette_grey =  list(Category10[nb_labels])
palette_grey.append('#C0C0C0')
clust_colors = [palette_grey[x] for x in clust.labels_]

plt.figure(figsize=[10,10])
plt.scatter(embedding[:, 0], embedding[:, 1], c=clust_colors, marker=marker, s=size_points)
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
# ### Multiple neighbors

# %% [markdown]
# We want to aggregate gene expression data up the 2nd, 3rd ot k-th neighbors.

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
    k_neigh : array_like
        The indices of neighboring nodes and their order.
    
    
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
        k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
        # select the kth order neighbors that have never been detected in previous orders
        keep_neigh = np.in1d(k_unique_neigh, unique_neigh, invert=True)
        k_unique_neigh = k_unique_neigh[keep_neigh]
        # register the kth order unique neighbors along with their order
        all_neigh.append([k_unique_neigh, k+1])
        # update array of unique detected neighbors
        unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        
    return all_neigh


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
order_color = [palette_grey[x] for x in order]

fig, ax = plt.subplots(figsize=[10, 10])
for a, b in pairs:
    xa, ya = pos[a]
    xb, yb = pos[b]
    ax.plot([xa, xb], [ya, yb], c='k',zorder=0, alpha=0.5)
for i, (x, y) in enumerate(pos.values()):
    ax.scatter(x, y, c=order_color[i], marker='o', s=40, zorder=10)

# %%
nb_cells = Xpred.shape[0]
nb_genes = Xpred.shape[1]
genes_aggreg = np.zeros((nb_cells, nb_genes*2)) # *2 because mean and variance are stored
pair_points = vor.ridge_points[selection,:]

for i in range(nb_cells):
    # array of neighbors of a given node
    left_neigh = pair_points[pair_points[:,1] == i, 0]
    right_neigh = pair_points[pair_points[:,0] == i, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    if neigh.size != 0:
        genes_aggreg[i,:nb_genes] = Xtest[neigh,:].mean(axis=0)
        genes_aggreg[i,-nb_genes:] = Xtest[neigh,:].std(axis=0)
    else:
        genes_aggreg[i,:] = None

# %% [markdown]
# ## Conclusion

# %% [markdown]
# We have seen that it is possible to assign to seqFISH data points their corresponding phenotypes defined from the scRNAseq data, with only 29 genes.  
#
# Moreover for seqFISH data aggregating gene expression for each node and it's neighbors we have found different groups, which migh correspond to areas of cell of different proportions in phenotypes.  
# It would be interesting to check that in a further analysis.  
#
# An interesting lead could be, for each cell, retrieve the mean values of its corresponding phenotype (the 'signature' of the phenotype), and then run again an aggregated neighbors' gene expression analysis. That could emphasise the genes that are under or over expressed due to the localisation of the cells and eliminate the strong contributions of genes that are specific of cell type.
