import pandas as pd
from sklearn import preprocessing


def rename_question(part_num, question_num):
    return str(part_num + 1) + "-" + str(question_num + 1)


def convert_numerical_response(response):
    try:
        resp_clean = int(response[0:2])
    except ValueError:
        resp_clean = response
    return resp_clean


def get_outcome(val, positive_outcomes, negative_outcomes):
    if val in positive_outcomes or str(val) in positive_outcomes:
        return 1
    elif val in negative_outcomes or str(val) in negative_outcomes:
        return 0
    return -1


class FeatureData:
    def __init__(
        self,
        json,
        dependent_variable,
        independent_variables,
        positive_outcomes,
        negative_outcomes,
    ):
        self.raw = pd.read_json(json)
        self.dependent_variable = dependent_variable
        self.encoded_dependent_variable = dependent_variable + "_enc"
        self.independent_variables = independent_variables
        self.encoded_independent_variables = [
            indep_var + "_enc" for indep_var in independent_variables
        ]
        self.positive_outcomes = positive_outcomes
        self.negative_outcomes = negative_outcomes
        self.question_list = [dependent_variable] + independent_variables
        self.full_feature_df = self.full_features()
        self.encoded_features = self.encoded_features()

    def processed(self):
        processed_df = self.raw.copy()
        processed_df["first_choice"] = processed_df.apply(
            lambda row: row["structure"]["choices"][0], axis=1
        )
        processed_df["question_name"] = processed_df.apply(
            lambda row: rename_question(row["part_position"], row["question_position"]),
            axis=1,
        )
        processed_df["response"] = processed_df.apply(
            lambda row: row["answers"][0], axis=1
        )
        processed_df["response_clean"] = processed_df.apply(
            lambda row: convert_numerical_response(row["answers"][0]), axis=1
        )
        return processed_df

    def full_features(self):
        processed = self.processed()
        snippet_df_full = (
            processed.pivot(
                index="snippet_id", columns="question_name", values="response_clean"
            )
            .reset_index()
            .dropna()
        )
        snippet_df_full["outcome"] = snippet_df_full.apply(
            lambda row: get_outcome(
                row[self.dependent_variable],
                self.positive_outcomes,
                self.negative_outcomes,
            ),
            axis=1,
        )
        return snippet_df_full[self.question_list + ["outcome"]]

    def question_choices(self):
        processed = self.processed()
        choices = dict()
        for question in self.question_list:
            choices[question] = processed[processed["question_name"] == question][
                "structure"
            ].iloc[0]["choices"][::-1]
        return choices

    def encoders(self):
        full_features = self.full_feature_df
        encoder_dict = dict()
        for question in self.question_list:
            label_encoder = preprocessing.LabelEncoder()
            encoder_dict[question] = label_encoder.fit(full_features[question])
        return encoder_dict

    def encoded_features(self):
        encoders = self.encoders()
        feature_copy = self.full_feature_df.copy()
        for question in self.question_list:
            col_name = question
            feature_copy[col_name] = encoders[question].transform(
                feature_copy[question]
            )
        return feature_copy

    def independent_variable_stats(self):
        indep_stats = dict()
        for variable in self.independent_variables:
            std = self.encoded_features[variable].std()
            mu = self.encoded_features[variable].mean()
            indep_stats[variable] = {"mu": mu, "std": std}
        return indep_stats

    def standardized_features(self):
        feature_copy = self.encoded_features.copy()
        independent_variable_stats = self.independent_variable_stats()

        for indep in self.independent_variables:
            std = independent_variable_stats[indep]["std"]
            mu = independent_variable_stats[indep]["mu"]
            feature_copy[indep] = feature_copy.apply(
                lambda row: (row[indep] - mu) / std, axis=1
            )
        return feature_copy

    def training_features(self):
        training_df = self.standardized_features().copy()
        training_df["sum"] = 0.0
        for indep in self.independent_variables:
            training_df["sum"] = training_df.apply(
                lambda row: row["sum"] + abs(row[indep]), axis=1
            )
        training_df["avg"] = training_df.apply(
            lambda row: row["sum"] / len(self.independent_variables), axis=1
        )
        return training_df[training_df["avg"] < 2]
