#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:54:30 2018

@author: timothy
"""
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA, TruncatedSVD

class ExploratoryAnalysis():
    def __init__(self, standardized_features, independent_variables,
                 dependent_variable, question_choices):
        self.standardized_features = standardized_features
        self.independent_variables = independent_variables
        self.dependent_variable = dependent_variable
        self.dependent_options = question_choices[dependent_variable]
        self.pca = self._fit_pca()
        self.lsa = self._fit_lsa()

    def correlation_matrix(self):
        return self.standardized_features[self.independent_variables + ['outcome']].corr()

    def _fit_pca(self):
        pca = PCA()
        try:
            pca.fit(self.standardized_features[self.independent_variables].dropna())
        except ValueError:
            pass
        return pca

    def pca_explained_variances(self):

        return self.pca.explained_variance_ratio_

    def correlation_matrix_plots(self):
        size = 2*len(self.independent_variables)
        matrix =\
            pd.plotting.scatter_matrix(self.standardized_features[self.independent_variables],
                              figsize=(size, size))
        return matrix

    def _fit_lsa(self):
        lsa = TruncatedSVD(n_components=2, n_iter=10)
        try:
           lsa.fit_transform((self.standardized_features[
                    self.independent_variables].dropna()))
        except ValueError:
            pass
        return lsa

    def plot_clusters(self):

        lsa_array =\
            self.lsa.transform((self.standardized_features[
                    self.independent_variables].dropna()))
        lsa_data = pd.DataFrame({'svd1':lsa_array[:,0], 'svd2':lsa_array[:,1]})
        lsa_data[self.dependent_variable] =\
            self.standardized_features[self.dependent_variable]

        ax = sns.scatterplot(x='svd1', y='svd2',
                             hue= self.dependent_variable, data=lsa_data,
                             palette="Paired")
        return ax

    def plot_outcome_clusters(self):
        lsa_array =\
            self.lsa.transform((self.standardized_features[
                    self.independent_variables].dropna()))
        lsa_data = pd.DataFrame({'svd1':lsa_array[:,0], 'svd2':lsa_array[:,1]})
        lsa_data['outcome'] =\
            self.standardized_features['outcome']

        ax = sns.scatterplot(x='svd1', y='svd2',
                             hue= 'outcome', data=lsa_data,
                             palette="Paired")
        return ax
