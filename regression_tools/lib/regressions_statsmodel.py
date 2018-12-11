#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:21:18 2018

@author: timothy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import statsmodels.formula.api as sm
from tabulate import tabulate


def linear_func(x, m, b):
    return m*x + b

def sigmoid_func(x, k, x0, A):

    return A / (1 + np.exp(-k*(x-x0)))


class RegressionModel():
    def __init__(self, feature_object):
        self.features = feature_object.encoded_features
        self.dependent_variable = feature_object.dependent_variable
        self.independent_variables = feature_object.continuous_independent_variables
        self.X_train = feature_object.training_features()[self.independent_variables]
        self.y_train = feature_object.training_features()[self.dependent_variable]
        self.X_test = feature_object.standardized_features()[self.independent_variables]
        self.y_test = feature_object.standardized_features()[self.dependent_variable]
        self.question_choices = feature_object.question_choices()
        self.encoders = feature_object.encoders()
        self.simulating_features = feature_object.simulating_features()
        self.question_stats = feature_object.independent_variable_stats()
        self.trained_model = self.trained_pipeline()
        self.y_pred = self.prediction()

    def trained_pipeline(self):
        ols = sm.GLS(self.y_train, self.X_train)
        model = ols.fit()
        return model

    def prediction(self):

        print(self.trained_model.predict(self.X_train))
        return self.trained_model.predict(self.X_test)

    def visualize_fit(self):

        fig, ax = plt.subplots()
        predicted = self.y_pred
        ax.scatter(self.y_test, predicted, edgecolors=(0, 0, 0))
        ax.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        return ax

    def visualize_goodness(self):
        fit_df = pd.DataFrame({'truth':self.y_test,
                               'prediction': self.y_pred})
        num_ticks = len(self.question_choices[self.dependent_variable])
        fig, ax = plt.subplots()
        for i, choice in enumerate(self.question_choices[self.dependent_variable]):
            encoded_choice = self.encoders[self.dependent_variable].transform([choice])[0]
            temp = fit_df[fit_df['truth'] == encoded_choice]['prediction'].copy()
            if temp.empty:
                pass
            else:
                n, bins, patches =\
                    plt.hist(temp, np.linspace(0,num_ticks -1 ,5*num_ticks),
                             density=False, alpha=0.5, label = choice)
        plt.xticks(np.linspace(0,num_ticks -1, num_ticks),
                   self.question_choices[self.dependent_variable],
                   rotation = 45)
        plt.legend(loc = 0)
        return ax
    def simulate_outcomes(self, independent_variable):


        outcome_range =\
            np.sort(self.y_test.unique())
        idx = self.independent_variables.index(independent_variable)

        simulation_df = self.simulating_features
        standard_features = [
            feat+'_std' for feat in self.independent_variables
        ]
        simulation_df['prediction'] = (
            self.trained_model
            .predict(simulation_df[standard_features])
        )

        indep_vals = (
            simulation_df
            .groupby(
                [independent_variable+'_val',
                 independent_variable]
            )['prediction']
            .mean().reset_index()
        ).sort_values(by=independent_variable)
        response_range = (
            indep_vals[independent_variable].tolist()
        )

        predictions = (
            indep_vals['prediction'].tolist()
        )
        question_choices = (
            indep_vals[independent_variable+'_val'].tolist()
        )
        fig, ax = plt.subplots()
        ax.scatter(response_range, predictions)
        popt, pcov = curve_fit(linear_func, response_range, predictions)
        slope = popt[0]
        intercept = popt[1]
        if intercept > 0:
             title = independent_variable + ": y =" + str(round(slope,3)) +\
                     "x +" + str(round(intercept,2))
        elif intercept > 0:
             title = independent_variable + ": y =" + str(round(slope,3)) +\
                     "x -" + str(-1*round(intercept,2))
        else:
             title = independent_variable + ": y =" + str(round(slope,3)) + "x"
        #print(ticks[indep_var])

        #ax.scatter(response_range, linear_func(response_range, slope, intercept))
        ax.plot(
            response_range,
            [linear_func(resp, slope, intercept) for resp in response_range]
        )
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_range))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_range, self.question_choices[self.dependent_variable])])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(self.question_choices[independent_variable], rotation=90)
        ax.set_title(title)
        return ax

    def print_results(self):

        print(self.trained_model.summary())



class MixedClassificationModel():
    def __init__(self, feature_object):
        self.features = feature_object.encoded_features
        self.dependent_variable = feature_object.dependent_variable
        self.independent_variables = feature_object.independent_variables
        self.continuous_independent_variables  = feature_object.continuous_independent_variables
        self.dummies = feature_object.dummies
        self.standardized_features = feature_object.standardized_features()
        self.training_features = feature_object.training_features()
        self.balanced = self.balance_training()
        self.X_train = self.balanced[self.independent_variables]
        self.y_train = self.balanced['outcome']
        self.X_test = self.standardized_features[self.independent_variables]
        self.y_test = self.standardized_features['outcome']
        self.question_choices = feature_object.question_choices()
        self.encoders = feature_object.encoders()
        self.simulating_features = feature_object.simulating_features()
        self.question_stats = feature_object.independent_variable_stats()
        self.trained_model = self.trained_pipeline()
        self.y_pred = self.prediction()

    def balance_training(self):

        pos_samples = len(self.training_features\
         [self.training_features['outcome'] == 1])

        neg_samples = len(self.training_features\
         [self.training_features['outcome'] == 0])

        sample_size = min(pos_samples, neg_samples)
        balanced =\
            pd.concat([self.training_features
                        [self.training_features['outcome'] == 0]
                        .sample(n=sample_size),
                       self.training_features
                        [self.training_features['outcome'] == 1]
                        .sample(n=sample_size)
                      ])
        return balanced


    def trained_pipeline(self):
        log_reg = sm.Logit(self.y_train, self.X_train)
        model = log_reg.fit()
        return model

    def prediction(self):
        return self.trained_model.predict(self.X_test)

    def visualize_fit(self):

        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, edgecolors=(0, 0, 0), alpha = .2)

        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')

        return ax

    def visualize_goodness(self):
        fit_df = pd.DataFrame({'truth':self.y_test,
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

    def simulate_continuous_outcomes(self, independent_variable):
        outcome_labels = ['negative', 'positive']
        outcome_choices = [0, 1]
        simulation_df = (
            self.simulating_features
            .sort_values(by=independent_variable)
            .reset_index()
        )
        standard_features = [
            feat+'_std' for feat in self.independent_variables
        ]
        simulation_df['prediction'] = (
            self.trained_model
            .predict(simulation_df[standard_features])
        )

        indep_vals = (
            simulation_df
            .groupby(
                [independent_variable+'_val',
                 independent_variable]
            )['prediction']
            .mean().reset_index()
        ).sort_values(by=independent_variable)
        response_range = (
            indep_vals[independent_variable].tolist()
        )
        predictions = (
            indep_vals['prediction'].tolist()
        )
        question_choices = (
            indep_vals[independent_variable+'_val'].tolist()
        )
        fig, ax = plt.subplots()
        ax.scatter(response_range, predictions)
        x0_start = np.mean(response_range)
        A_start = 1
        try:
            k_start = .5
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
        ax.plot(
            response_range,
            [sigmoid_func(resp, k, x0, A) for resp in response_range]
        )
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_choices))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_choices, outcome_labels)])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(question_choices, rotation=90)
        ax.set_title(title)
        return ax

    def simulate_binary_outcomes(self, independent_variable):
        outcome_labels = ['negative', 'positive']
        outcome_choices = [0, 1]

        simulation_df = (
            self.simulating_features
            .sort_values(by=independent_variable)
            .reset_index()
        )
        standard_features = [
            feat+'_std' for feat in self.independent_variables
        ]
        simulation_df['prediction'] = (
            self.trained_model
            .predict(simulation_df[standard_features])
        )

        indep_vals = (
            simulation_df
            .groupby(
                [independent_variable+'_val',
                 independent_variable]
            )['prediction']
            .mean().reset_index()
        ).sort_values(by=independent_variable)
        #print(indep_vals)
        response_range = (
            indep_vals[independent_variable].tolist()
        )
        predictions = (
            indep_vals['prediction'].tolist()
        )

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



    def print_results(self):

        print(self.trained_model.summary())

    def odds_ratio(self):
        odds_increase = [
            int(round(100*(np.exp(param) - 1),0))
            for param in self.trained_model.params
        ]
        odds_ratio_table = pd.DataFrame(
            {'independent_variable':self.independent_variables,
             'odds_%_increase': odds_increase
            }
        )
        odds_ratio_table['mu'] = odds_ratio_table.apply(
            lambda row:
            round(self.question_stats[row['independent_variable']]['mu'], 2),
            axis=1
        )
        odds_ratio_table['std'] = odds_ratio_table.apply(
            lambda row:
            round(self.question_stats[row['independent_variable']]['std'],2),
            axis=1
        )
        print(odds_ratio_table.to_string(index=False))
        #print(np.exp(self.trained_model.params))
