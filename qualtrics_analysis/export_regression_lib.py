import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

def trained_model(train_features, train_outcomes):
    np.random.seed(44)
    log_reg = sm.Logit(train_outcomes, train_features)
    param_array = np.zeros(len(train_features) + 1)
    iterations = 5
    for i in range(0, iterations):
        model = log_reg.fit(
            maxiter=5000, avextol=0.0001, epsilon=0.1, full_output=1, disp=0
        )
        params = list(model.params)
        param_array = [a + b for a, b in zip(param_array, params)]
        start = [p / iterations for p in param_array]
        model = log_reg.fit(
            start_params=start,
            maxiter=5000,
            avextol=0.0001,
            epsilon=0.1,
            full_output=1,
            disp=1,
        )
    return model


def odds_ratio(trained_model, independent_variables, independent_variables_indicies):
    sig_params = [
        trained_model.params[i]
        for i, col in enumerate(trained_model.params)
        if i in independent_variables_indicies
    ]
    odds_increase = [
        int(round(100 * (np.exp(param) - 1), 0)) for param in sig_params
    ]
    odds_ratio_table = pd.DataFrame(
        {
            "independent_variable": independent_variables,
            "odds_%_increase": odds_increase,
        }
    )
    # odds_ratio_table.to_string(index=False))
    return odds_ratio_table


def independent_variable_stats(feature_data, independent_variables):
    indep_stats = dict()
    for variable in independent_variables:
        std = feature_data[variable].std()
        mu = feature_data[variable].mean()
        indep_stats[variable] = {"mu": mu, "std": std}
    return indep_stats


def standardized_features(feature_data, independent_variables, stats):
    feature_copy = feature_data.copy()

    for indep in independent_variables:
        std = stats[indep]["std"]
        mu = stats[indep]["mu"]
        feature_copy[indep] = feature_copy.apply(
            lambda row: (row[indep] - mu) / std, axis=1
        )
        feature_copy[indep] = feature_copy[indep].fillna(-1)
    return feature_copy

def simulating_features(standardized_features, independent_variables):
    simulation_df = standardized_features[independent_variables].drop_duplicates()
    return simulation_df


def simulate_continuous_outcomes(
    standardized_features, trained_model, independent_variables, independent_variable
):
    outcome_labels = ["negative", "positive"]
    outcome_choices = [0, 1]
    simulation_df = standardized_features.sort_values(
        by=independent_variable
    ).reset_index()
    simulation_df["prediction"] = trained_model.predict(
        simulation_df[independent_variables]
    )

    indep_vals = (
        simulation_df.groupby([independent_variable])[
            "prediction"
        ]
        .mean()
        .reset_index()
    ).sort_values(by=independent_variable)
    response_range = indep_vals[independent_variable].tolist()
    predictions = indep_vals["prediction"].tolist()
    question_choices = indep_vals[independent_variable].tolist()
    return {
        "response_range": response_range,
        "predictions": predictions,
        "question_choices": question_choices,
    }