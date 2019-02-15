#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:21:18 2018

@author: timothy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy import stats
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import statsmodels.formula.api as sm
from tabulate import tabulate


def linear_func(x, m, b):
    return m*x + b

def sigmoid_func(x, k, x0, A):

    return A / (1 + np.exp(-k*(x-x0)))



class ClassificationModel():
    
    def __init__(self, training_df, dependent_variable, independent_variables):
        self.independent_variables = independent_variables
        self.training = training_df
        self.standardized = self.standardized_features()
        self.X_train = self.standardized[self.independent_variables]
        self.y_train = self.standardized['outcome']
        self.trained_model = self.train_model()
        self.y_pred = self.trained_model.predict_proba(self.X_train)[:,1]
        self.question_stats = self.independent_variable_stats()

    
    def covariance_matrix(self):

        X = self.X_train[self.independent_variables].values
        X_T = X.transpose()
        cf_array = self.trained_model.coef_
        w_diag = np.array([np.exp(np.dot(cf_array, x_i))[0]/
                           (1+np.exp(np.dot(cf_array, x_i))[0])**2
                            for x_i in X])
        W = np.diag(w_diag)
        cov = np.linalg.inv(np.dot(X_T,np.dot(W,X)))
        return np.matrix(cov)
    
    def independent_variable_stats(self):
        indep_stats = dict()
        for variable in self.independent_variables:
            std = self.training[variable].std()
            mu = self.training[variable].mean()
            indep_stats[variable] = {"mu": mu, "std": std}
        return indep_stats
    
    def lower_bound_90(self):
        beta = self.trained_model.coef_[0]
        return [beta_i - 1.64 * se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]
    
    def odds_increase(self):

        return [ 
            int(round(100 * (np.exp(coef) - 1), 0)) 
            for coef in self.trained_model.coef_[0]
        ]    
    
    def p_values(self):
        p_vals = [stats.norm.sf(np.abs(w_stat_i))*2 for w_stat_i in
                  self.wald_statistics()]
        return p_vals
    
    def prediction(self):

        return self.trained_model.predict_proba(self.X_train)[:,1]
    
    def print_results(self):

        coefs = [round(cf,3) for cf in self.trained_model.coef_[0]]
        standard_errors = [round(se,3) for se in self.standard_errors()]
        wald_stats = [round(ts,3) for ts in self.wald_statistics()]
        p_vals = [round(pv,3) for pv in self.p_values()]
        lower_bounds = [round(pv,3) for pv in self.lower_bound_90()]
        upper_bounds = [round(pv,3) for pv in self.upper_bound_90()]
        odds = self.odds_increase()
        col_names = ['Variable', 'Coefficienct', 'Standard Error', 'z','p-Value',
                     '[0.1', '0.9]', 'Odds Increase']
        printable_df = pd.DataFrame({col_names[0]: self.independent_variables,
                                     col_names[1]: coefs,
                                     col_names[2]: standard_errors,
                                     col_names[3]: wald_stats,
                                     col_names[4]: p_vals,
                                     col_names[5]: lower_bounds,
                                     col_names[6]: upper_bounds,
                                     col_names[7]: odds})
        print("Number of Observations:", len(self.X_train))
        print("MSE:", round(
            mean_squared_error(
                self.y_pred,
                self.y_train),
            3
        ))
        print("r^2:", round(
            r2_score(
                self.y_pred,
                self.y_train),
            3
        ))
        print(tabulate(printable_df[col_names], headers='keys'))
    
    def simulate_binary_outcomes(self, independent_variable):
        response_range = [0,1]
        outcome_labels = ['negative', 'positive']
        outcome_choices = [0, 1]
        idx = self.independent_variables.index(independent_variable)

        mu = self.question_stats[independent_variable]['mu']
        sig = self.question_stats[independent_variable]['std']
        predictions = []
        for j, val in enumerate(response_range):
            feature_array = np.zeros(len(self.independent_variables))
            feature_array[idx] = (val - mu)/sig
            predictions.append(self.trained_model.predict_proba([feature_array])[0][1])

        fig, ax = plt.subplots()
        ax.scatter(response_range, predictions)
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_choices))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_choices, outcome_labels)])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(['no','yes'], rotation=90)
        ax.set_title(independent_variable)
        return ax
    
    def simulate_continuous_outcomes(self, independent_variable, independent_variable_idx):
        response_range =\
            np.sort(self.training[independent_variable].unique())
        outcome_labels = ['negative', 'positive']
        outcome_choices = [0, 1]
        idx = self.independent_variables.index(independent_variable)

        mu = self.question_stats[independent_variable]['mu']
        sig = self.question_stats[independent_variable]['std']
        predictions = []
        
        for j, val in enumerate(response_range):
            feature_array = np.zeros(len(self.independent_variables))
            feature_array[idx] = (val - mu)/sig
            predictions.append(self.trained_model.predict_proba([feature_array])[0][1])

        fig, ax = plt.subplots()
        ax.scatter(response_range, predictions)
        """
        x0_start = np.mean(response_range)
        A_start = 1
        try:
            k_start = .5
            print([k_start, x0_start, A_start])
            popt, pcov =\
                 curve_fit(sigmoid_func, response_range,
                           predictions, p0 = [k_start, x0_start, A_start],
                           maxfev=1000)
            k = popt[0]
            x0 = popt[1]
            A = popt[2]
        except RuntimeError:
            try:
                k_start = -.5
                print([k_start, x0_start, A_start])
                popt, pcov =\
                     curve_fit(sigmoid_func, response_range,
                               predictions, p0 = [k_start, x0_start, A_start],
                               maxfev=2000)
                k = popt[0]
                x0 = popt[1]
                A = popt[2]
            except RuntimeError:
                try:
                    k_start = 0
                    A_start = .5
                    popt, pcov =\
                    curve_fit(sigmoid_func, response_range,
                              predictions, p0 = [k_start, x0_start, A_start],
                              maxfev=2000)
                    k = popt[0]
                    x0 = popt[1]
                    A = popt[2]
                except RuntimeError:
                    k = 0
                    x0 = 0
                    A = 1
        title = (independent_variable + "p(Positive|x) ="+str(round(A,2))+
            "/(1+exp(-"+str(round(k,2))+ "(x +"+ str(round(x0,2))+")))"
        )
        ax.plot(response_range, sigmoid_func(response_range, k, x0, A))
        """
        #ax.set_xlabel('Varied '+indep_var)
        x0 = round(self.trained_model.intercept_[0],2)
        coef = round(self.trained_model.coef_[0][independent_variable_idx],2)
        
        title = (independent_variable + "p(Positive|x) ="+
            "1/(1+exp(-"+str(coef)+ "(x +"+ str(x0)+")))"
        )
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_choices))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_choices, outcome_labels)])
        ax.set_xticks(np.sort(response_range))
        ax.set_title(title)
        return ax
    
    def standard_errors(self):
        cov_diag = np.array(self.covariance_matrix().diagonal())[0]
        se_beta = [np.sqrt(abs(beta_i)) for beta_i in cov_diag]
        return se_beta
    
    def standardized_features(self):
        feature_copy = self.training.copy()
        stats = self.independent_variable_stats()
        for indep in self.independent_variables:
            std = stats[indep]["std"]
            mu = stats[indep]["mu"]
            feature_copy[indep] = feature_copy.apply(
                lambda row: (row[indep] - mu) / std if std > 0
                else 0,
                axis=1
            )
            feature_copy[indep] = feature_copy[indep].fillna(-1)
        return feature_copy

    def train_model(self):
        log_reg = linear_model.LogisticRegressionCV(
            cv=5,
            max_iter=1000,
            Cs=[1.0,0.1,.001],
            random_state=1234,
            refit=True,
            class_weight='balanced'
        )
        log_reg.fit(self.X_train, self.y_train)

        return log_reg

    def upper_bound_90(self):
        beta = self.trained_model.coef_[0]
        return [beta_i + 1.64 * se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def visualize_fit(self):

        fig, ax = plt.subplots()
        ax.scatter(self.y_train, self.y_pred, edgecolors=(0, 0, 0), alpha = .2)

        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        return ax

    def visualize_goodness(self):
        fit_df = pd.DataFrame({'truth':self.y_train,
                               'prediction': [prob_a for prob_a in
                                              self.y_pred]})
        num_ticks = 2
        outcome_labels = ['negative', 'positive']
        outcome_choices = [0, 1]
        fig, ax = plt.subplots()
        for i, choice in enumerate(outcome_labels):
            encoded_choice = outcome_choices[i]
            temp = fit_df[fit_df['truth'] == encoded_choice]['prediction'].copy()
            if temp.empty:
                pass
            else:
                n, bins, patches =\
                    plt.hist(temp, np.linspace(0,num_ticks -1 ,5*num_ticks),
                             density=False, alpha=0.5, label = choice)
        plt.xticks(np.linspace(0,num_ticks -1, num_ticks),
                   outcome_labels,
                   rotation = 45)
        plt.legend(loc = 0)
        return ax

    def wald_statistics(self):

        beta = self.trained_model.coef_[0]
        return [beta_i/se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]








