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
from tabulate import tabulate


def linear_func(x, m, b):
    return m*x + b

def sigmoid_func(x, k, x0, A):

    return A / (1 + np.exp(-k*(x-x0)))

class ClassificationModel():
    def __init__(self, feature_object):
        self.features = feature_object.encoded_features
        self.dependent_variable = feature_object.dependent_variable
        self.independent_variables = feature_object.independent_variables
        self.training_features = feature_object.training_features()
        self.balanced = self.balance_training()
        self.X_train = self.balanced[self.independent_variables]
        self.y_train = self.balanced['outcome']
        self.X_test = feature_object.standardized_features()[self.independent_variables]
        self.y_test = feature_object.standardized_features()['outcome']
        self.question_choices = feature_object.question_choices()
        self.encoders = feature_object.encoders()
        self.question_stats = feature_object.independent_variable_stats()
        self.trained_model = self.trained_pipeline()
        self.y_pred = self.prediction()
    
    def balance_training(self):
        
        pos_samples = len(self.training_features()\
         [self.training_features()['outcome'] == 1])
        
        neg_samples = len(self.training_features()\
         [self.training_features()['outcome'] == 0])
        
        sample_size = min(pos_samples, neg_samples)
        balanced =\
            pd.concat([self.training_features()
                        [self.training_features()['outcome'] == 0]
                        .sample(n=sample_size),
                       self.training_features()
                        [self.training_features()['outcome'] == 1]
                        .sample(n=sample_size)
                      ])
        return balanced
        

    def trained_pipeline(self):
        pca = PCA()
        log_reg = linear_model.LogisticRegressionCV(class_weight='balanced', cv=5)
        pipe = Pipeline(steps=[('pca', pca), ('classifier', log_reg)])
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def prediction(self):

        return self.trained_model.predict_proba(self.X_test)[:,1]

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
    def simulate_outcomes(self, independent_variable):
        response_range =\
            np.sort(self.features[independent_variable].unique())
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

        title = "p(Positive|x) ="+str(round(A,2))+"/(1+exp(-"+str(round(k,2))+\
                                       "(x +"+ str(round(x0,2))+")))"
        ax.plot(response_range, sigmoid_func(response_range, k, x0, A))
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_choices))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_choices, outcome_labels)])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(self.question_choices[independent_variable], rotation=90)
        ax.set_title(title)
        return ax

    def covariance_matrix(self):

        X = self.X_test[self.independent_variables].values
        X_T = X.transpose()
        cf_array = self.trained_model.steps[1][1].coef_
        w_diag = np.array([np.exp(np.dot(cf_array, x_i))[0]/
                           (1+np.exp(np.dot(cf_array, x_i))[0])**2
                            for x_i in X])
        W = np.diag(w_diag)
        cov = np.linalg.inv(np.dot(X_T,np.dot(W,X)))
        return np.matrix(cov)

    def standard_errors(self):
        cov_diag = np.array(self.covariance_matrix().diagonal())[0]
        se_beta = [np.sqrt(beta_i) for beta_i in cov_diag]
        return se_beta

    def wald_statistics(self):

        beta = self.trained_model.steps[1][1].coef_[0]
        return [beta_i/se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def p_values(self):
        p_vals = [stats.norm.sf(np.abs(w_stat_i))*2 for w_stat_i in
                  self.wald_statistics()]
        return p_vals

    def print_results(self):

        coefs = [round(cf,3) for cf in self.trained_model.steps[1][1].coef_[0]]
        standard_errors = [round(se,3) for se in self.standard_errors()]
        wald_stats = [round(ts,3) for ts in self.wald_statistics()]
        p_vals = [round(pv,3) for pv in self.p_values()]
        col_names = ['Variable', 'Coefficienct', 'Standard Error', 'z','p-Value']
        printable_df = pd.DataFrame({col_names[0]: self.independent_variables,
                                     col_names[1]: coefs,
                                     col_names[2]: standard_errors,
                                     col_names[3]: wald_stats,
                                     col_names[4]: p_vals})
        print(tabulate(printable_df[col_names], headers='keys'))


class RegressionModel():
    def __init__(self, feature_object):
        self.features = feature_object.encoded_features
        self.dependent_variable = feature_object.dependent_variable
        self.independent_variables = feature_object.independent_variables
        self.X_train = feature_object.training_features()[self.independent_variables]
        self.y_train = feature_object.training_features()[self.dependent_variable]
        self.X_test = feature_object.standardized_features()[self.independent_variables]
        self.y_test = feature_object.standardized_features()[self.dependent_variable]
        self.question_choices = feature_object.question_choices()
        self.encoders = feature_object.encoders()
        self.question_stats = feature_object.independent_variable_stats()
        self.trained_model = self.trained_pipeline()
        self.y_pred = self.prediction()

    def trained_pipeline(self):
        pca = PCA()
        lin_reg = linear_model.HuberRegressor()
        pipe = Pipeline(steps=[('pca', pca), ('regressor', lin_reg)])
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def prediction(self):

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

        response_range =\
            np.sort(self.features[independent_variable].unique())
        outcome_range =\
            np.sort(self.y_test.unique())
        idx = self.independent_variables.index(independent_variable)

        mu = self.question_stats[independent_variable]['mu']
        sig = self.question_stats[independent_variable]['std']
        predictions = []
        for j, val in enumerate(response_range):
            feature_array = np.zeros(len(self.independent_variables))
            feature_array[idx] = (val - mu)/sig
            predictions.append(self.trained_model.predict([feature_array])[0])
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

        fig, ax = plt.subplots()
        #ax.scatter(response_range, linear_func(response_range, slope, intercept))
        ax.plot(response_range, linear_func(response_range, slope, intercept))
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_range))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_range, self.question_choices[self.dependent_variable])])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(self.question_choices[independent_variable], rotation=90)
        ax.set_title(title)
        return ax

    def get_mse(self):
        se = 0
        for truth, pred in zip(self.y_test, self.y_pred):
            se = se + (truth - pred)**2
        return se/(len(self.y_test) - len(self.independent_variables) - 1)

    def covariance_matrix(self):
        X_df = self.X_test
        X_df['ones'] = 1
        X = X_df[['ones'] + self.independent_variables].values()
        X_T = X.transpose()
        mse = mean_squared_error(self.y_test, self.y_pred)
        cov = mse *np.linalg.inv(np.dot(X_T,X))
        return np.matrix(cov)

    def standard_errors(self):
        cov_diag = np.array(self.covariance_matrix().diagonal())[0][1::]
        #cov_diag = np.array(self.covariance_matrix_skl().diagonal())[0]
        se_beta = [np.sqrt(beta_i) for beta_i in cov_diag]
        return se_beta

    def t_statistics(self):

        beta = self.trained_model.steps[1][1].coef_
        return [beta_i/se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def p_values(self):
        p_vals = [stats.t.sf(np.abs(t_stat_i), len(self.X_test)-1)*2 for t_stat_i in
                  self.t_statistics()]
        return p_vals

    def print_results(self):

        coefs = [round(cf,3) for cf in self.trained_model.steps[1][1].coef_]
        standard_errors = [round(se,3) for se in self.standard_errors()]
        t_stats = [round(ts,3) for ts in self.t_statistics()]
        p_vals = [round(pv,3) for pv in self.p_values()]
        col_names = ['Variable', 'Coefficienct', 'Standard Error', 't','p-Value']
        printable_df = pd.DataFrame({col_names[0]: self.independent_variables,
                                     col_names[1]: coefs,
                                     col_names[2]: standard_errors,
                                     col_names[3]: t_stats,
                                     col_names[4]: p_vals})
        print(tabulate(printable_df[col_names], headers='keys'))

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
        pca = PCA()
        log_reg = linear_model.LogisticRegressionCV(class_weight='balanced', cv=5)
        pipe = Pipeline(steps=[('classifier', log_reg)])
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def prediction(self):

        return self.trained_model.predict_proba(self.X_test)[:,1]

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
        response_range =\
            np.sort(self.features[independent_variable].unique())
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

        title = "p(Positive|x) ="+str(round(A,2))+"/(1+exp(-"+str(round(k,2))+\
                                       "(x +"+ str(round(x0,2))+")))"
        ax.plot(response_range, sigmoid_func(response_range, k, x0, A))
        #ax.set_xlabel('Varied '+indep_var)
        ax.set_ylabel('Prediction')
        ax.set_yticks(np.sort(outcome_choices))
        ax.set_yticklabels([str(outcome)+' ' + choice for outcome, choice in
                            zip(outcome_choices, outcome_labels)])
        ax.set_xticks(np.sort(response_range))
        ax.set_xticklabels(self.question_choices[independent_variable], rotation=90)
        ax.set_title(title)
        return ax

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

    def covariance_matrix(self):

        X = self.X_test[self.independent_variables].values
        X_T = X.transpose()
        cf_array = self.trained_model.steps[0][1].coef_
        w_diag = np.array([np.exp(np.dot(cf_array, x_i))[0]/
                           (1+np.exp(np.dot(cf_array, x_i))[0])**2
                            for x_i in X])
        W = np.diag(w_diag)
        cov = np.linalg.inv(np.dot(X_T,np.dot(W,X)))
        return np.matrix(cov)

    def standard_errors(self):
        cov_diag = np.array(self.covariance_matrix().diagonal())[0]
        se_beta = [np.sqrt(abs(beta_i)) for beta_i in cov_diag]
        return se_beta

    def wald_statistics(self):

        beta = self.trained_model.steps[0][1].coef_[0]
        return [beta_i/se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def p_values(self):
        p_vals = [stats.norm.sf(np.abs(w_stat_i))*2 for w_stat_i in
                  self.wald_statistics()]
        return p_vals

    def lower_bound_90(self):
        beta = self.trained_model.steps[0][1].coef_[0]
        return [beta_i - 1.64 * se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def upper_bound_90(self):
        beta = self.trained_model.steps[0][1].coef_[0]
        return [beta_i + 1.64 * se_beta_i for beta_i, se_beta_i
                in zip(beta,self.standard_errors())]

    def print_results(self):

        coefs = [round(cf,3) for cf in self.trained_model.steps[0][1].coef_[0]]
        standard_errors = [round(se,3) for se in self.standard_errors()]
        wald_stats = [round(ts,3) for ts in self.wald_statistics()]
        p_vals = [round(pv,3) for pv in self.p_values()]
        lower_bounds = [round(pv,3) for pv in self.lower_bound_90()]
        upper_bounds = [round(pv,3) for pv in self.upper_bound_90()]
        col_names = ['Variable', 'Coefficienct', 'Standard Error', 'z','p-Value',
                     'lower', 'upper']
        printable_df = pd.DataFrame({col_names[0]: self.independent_variables,
                                     col_names[1]: coefs,
                                     col_names[2]: standard_errors,
                                     col_names[3]: wald_stats,
                                     col_names[4]: p_vals,
                                     col_names[5]: lower_bounds,
                                     col_names[6]: upper_bounds})
        print(tabulate(printable_df[col_names], headers='keys'))
