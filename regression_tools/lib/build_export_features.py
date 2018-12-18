import datetime
from lib.encoders import continuous_label_encoder
import pandas as pd
import numpy as np
from sklearn import preprocessing


def rename_question(part_num, question_num):
    return str(part_num + 1) + "-" + str(question_num + 1)


def convert_na_response(response):
    if response.startswith('N/A'):
        return -1
    return response

def convert_numerical_response(response):
    try:
        resp_clean = int(response[0:2])
    except (ValueError, TypeError) as e:
        resp_clean = response
    return resp_clean


def convert_something_else_response(response, choices):
    if str(response).lower().startswith("something else") or (
        not any(
            [
                str(response).lower().startswith(str(choice).lower().split()[0])
                for choice in choices
            ]
        )
        and not any(
            [str(response).lower() == str(choice).lower() for choice in choices]
        )
    ):
        return "other"
    return response


def get_outcome(val, positive_outcomes, negative_outcomes):
    if val in positive_outcomes or str(val) in positive_outcomes:
        return 1
    elif val in negative_outcomes or str(val) in negative_outcomes:
        return 0
    return -1


def rename_categorical(question_name, response):

    q_name = str(question_name) + "_" + str(response)
    return q_name


def fill_non_null(cell_value):
    if cell_value != -1:
        return 1
    return 0

def get_age_today(birthday):
    if birthday is not None and birthday != pd.NaT:
        today =  datetime.datetime.today()
        try:
            return ((today - birthday).days/365.25)
        except TypeError:
            return ((today.date() - birthday).days/365.25)
    return -1


def drop_low_var_dummies(pivoted_df, grouping):
    included_columns = [grouping]
    for feature in list(pivoted_df):
        if feature != grouping:
            if pivoted_df[feature].sum() > 2 and pivoted_df[feature].sum() < (
                len(pivoted_df) - 2
            ):
                included_columns.append(feature)

    return pivoted_df[included_columns]


def fill_dummy_values(pivoted_df, grouping):
    for feature in list(pivoted_df):
        if feature != grouping:
            pivoted_df[feature] = pivoted_df.apply(
                lambda row: fill_non_null(row[feature]), axis=1
            )
    return drop_low_var_dummies(pivoted_df, grouping)


def make_dummies(categorical_df, grouping):
    categorical_df["question_name"] = categorical_df.apply(
        lambda row: rename_categorical(row["question_name"], row["response_clean"]),
        axis=1,
    )

    categorical_df = categorical_df.drop_duplicates([grouping, "question_name"])

    pivoted = (
        categorical_df.pivot(
            index=grouping, columns="question_name", values="response_clean"
        )
        .reset_index()
        .fillna(-1)
    )
    return fill_dummy_values(pivoted, grouping)


def convert_income(income):
    if income == 'very_high':
        return 6
    elif income == 'high':
        return 5
    elif income == 'lower_high':
        return 4
    elif income == 'upper_middle':
        return 3
    elif income == 'middle':
        return 2
    elif income == 'lower_middle':
        return 1
    elif income == 'low':
        return 0
    else:
        return -1

def get_first_choice(structure):
    try:
        return structure["choices"][0]
    except KeyError:
        return 0


def get_question_choices(structure):
    try:
        return structure["choices"][::-1]
    except KeyError:
        return [int(i) for i in np.linspace(0, 100, 101)]

def unlistify(df, column):
    matches = [i for i,n in enumerate(df.columns)
             if n==column]

    if len(matches)==0:
        raise Exception('Failed to find column named ' + column +'!')
    if len(matches)>1:
        raise Exception('More than one column named ' + column +'!')

    col_idx = matches[0]

  # Helper function to expand and repeat the column col_idx
    def fnc(d):
        row = list(d.values[0])
        bef = row[:col_idx]
        aft = row[col_idx+1:]
        col = row[col_idx]
        z = [bef + [c] + aft for c in col]
        return pd.DataFrame(z)

    col_idx += len(df.index.shape) # Since we will push reset the index
    index_names = list(df.index.names)
    column_names = list(index_names) + list(df.columns)
    return (df
          .reset_index()
          .groupby(level=0,as_index=0)
          .apply(fnc)
          .rename(columns = lambda i :column_names[i])
          .set_index(index_names)
          )


class MixedFeatureData:
    def __init__(
        self,
        raw,
        dependent_variable,
        continuous_independent_variables,
        binary_independent_variables,
        categorical_independent_variables,
        positive_outcomes,
        negative_outcomes,
        demo_independent_variables,
        tag_independent_variables,
        scout_group_independent_variables,
        grouping,
    ):
        self.raw = raw
        self.dependent_variable = dependent_variable
        self.encoded_dependent_variable = dependent_variable + "_enc"
        self.binary_independent_variables = binary_independent_variables
        self.continuous_independent_variables = (
            continuous_independent_variables + binary_independent_variables
        )
        self.demo_independent_variables = demo_independent_variables
        self.categorical_independent_variables = categorical_independent_variables
        self.tag_independent_variables = tag_independent_variables
        self.scout_group_independent_variables = scout_group_independent_variables
        self.positive_outcomes = positive_outcomes
        self.negative_outcomes = negative_outcomes
        self.grouping = grouping
        self.continous_question_list = [
            dependent_variable
        ] + self.continuous_independent_variables
        self.raw_question_list = (
            self.continous_question_list + categorical_independent_variables
        )
        self.question_list = (
            self.continous_question_list +
            self.binary_independent_variables
        )
        self.processed_df = self.processed()

    def processed(self):
        print("processing")
        processed_df = self.raw.copy()
        # processed_df = unlistify(processed_df, 'answers')
        processed_df["first_choice"] = processed_df.apply(
            lambda row: get_first_choice(row["structure"]), axis=1
        )
        processed_df["choice_list"] = processed_df.apply(
            lambda row: get_question_choices(row["structure"]), axis=1
        )
        processed_df["question_name"] = processed_df.apply(
            lambda row: rename_question(row["part_position"], row["question_position"]),
            axis=1,
        )

        processed_df["response"] = processed_df.apply(
            lambda row: row["answers"][0], axis=1
        )
        processed_df["response_clean"] = processed_df.apply(
            lambda row: convert_na_response(row["answers"][0]), axis=1
        )
        processed_df = processed_df[processed_df['response_clean'] != -1].copy()
        processed_df["response_clean"] = processed_df.apply(
            lambda row: convert_numerical_response(row["response_clean"]), axis=1
        )
        processed_df["response_clean"] = processed_df.apply(
            lambda row: convert_something_else_response(
                row["response_clean"], row["choice_list"]
            ),
            axis=1,
        )
        return processed_df

    def full_features(self):
        processed = self.processed()
        processed_of_interest = (processed[
            processed["question_name"].isin(
                self.continuous_independent_variables + [self.dependent_variable]
            )
            ].drop_duplicates([self.grouping,'question_name'], keep='last')
        )

        snippet_df_continuous = (
            processed_of_interest
            .drop_duplicates([self.grouping,'question_name'], keep='last')
            .pivot(index=self.grouping, columns="question_name", values="response_clean")
            .reset_index()
            .dropna()
        )
        other_variables = []

        if 'age' in self.demo_independent_variables:
            today = datetime.datetime.today()
            age_df = processed[[self.grouping, 'birthday']].drop_duplicates()
            age_df['birthday'] = age_df['birthday'].astype('datetime64[ns]')
            age_df['age'] = (
                age_df.apply(
                    lambda row: get_age_today(row['birthday']),
                    axis=1
                )
            )
            age_df['age'] = age_df[~age_df['age'].isnull()].apply(
                lambda row: int(round(row['age'])),
                axis=1
            )
            snippet_df_continuous = (
                pd.merge(snippet_df_continuous,
                         age_df[[self.grouping,'age']],
                         on=self.grouping)
            )
            other_variables.append('age')
        if 'gender' in self.demo_independent_variables:
            gender_df = processed[[self.grouping, 'gender']].drop_duplicates()
            gender_df['gender'] = gender_df.apply(lambda row: 0 if row['gender'] == 'male' else 1,
                                                  axis=1)
            snippet_df_continuous = (
                pd.merge(snippet_df_continuous,
                         gender_df[[self.grouping,'gender']],
                         on=self.grouping)
            )
            other_variables.append('gender')
        if 'household_income' in self.demo_independent_variables:
            income_df = processed[[self.grouping, 'household_income']].drop_duplicates()
            income_df['income'] = income_df.apply(lambda row: convert_income(row['household_income']),
                                                  axis=1)
            snippet_df_continuous = (
                pd.merge(snippet_df_continuous,
                         income_df[[self.grouping,'income']],
                         on=self.grouping)
            )
            other_variables.append('income')

        if self.categorical_independent_variables:
            snippets_categorical = processed[
                processed["question_name"].isin(self.categorical_independent_variables)
            ].copy()
            snippet_df_categorical = make_dummies(snippets_categorical, self.grouping)
            other_variables = (
                        other_variables + [var  for var in list(snippet_df_categorical)
                                           if var != self.grouping]
                    )
            for demo in ['ethnicity', 'education','tag']:
                if demo in self.demo_independent_variables:
                    demo_df = processed[[self.grouping, demo]].drop_duplicates()
                    demo_df_dummies = pd.get_dummies(user_snippets[[self.grouping, demo]],
                                                     self.grouping, demo)
                    snippet_df_categorical = (
                        pd.merge(snippet_df_categorical,
                        demo_df_dummies,
                        on=self.grouping
                      )
                    )
                    other_variables = (
                        other_variables + [var  for var in list(demo_df_dummies)
                                           if var != self.grouping]
                    )

            if self.tag_independent_variables:
                tag_df = (
                    processed[processed['tag'].isin(self.tag_independent_variables)]
                    [[self.grouping, 'tag']].drop_duplicates()
                )
                tag_dummy_df = pd.get_dummies(tag_df, 'tag')
                snippet_df_categorical = (
                    pd.merge(snippet_df_categorical,
                             tag_dummy_df,
                             on=self.grouping,
                             how='left').fillna(0)
                )
                other_variables = other_variables + [var  for var in list(tag_dummy_df)
                                           if var != self.grouping]
            if self.scout_group_independent_variables:
                scout_group_df = (
                    processed[processed['scout_group'].isin(self.scout_group_independent_variables)]
                    [[self.grouping, 'scout_group']].drop_duplicates()
                )
                scout_group_dummy_df = pd.get_dummies(scout_group_df, 'scout_group')
                snippet_df_categorical = (
                    pd.merge(snippet_df_categorical,
                             scout_group_dummy_df,
                             on=self.grouping,
                             how='left').fillna(0)
                )
                #print(scout_group_dummy_df)
                other_variables = other_variables + [var  for var in list(scout_group_dummy_df)
                                           if var != self.grouping]
            #print(list(snippet_df_categorical))
            #print(list(snippet_df_continuous))
            snippet_df_full = (pd.merge(
                snippet_df_continuous,
                snippet_df_categorical,
                on=self.grouping,
                how="outer",)
            )

        else:
            snippet_df_full = snippet_df_continuous.copy()
            for demo in ['ethnicity', 'education']:
                if demo in self.demo_independent_variables:
                    demo_df = processed[[self.grouping, demo]].drop_duplicates()
                    demo_df_dummies = pd.get_dummies(user_snippets[[self.grouping, demo]],
                                                     self.grouping, demo)
                    snippet_df_full = (
                        pd.merge(snippet_df_full,
                        demo_df_dummies[[self.grouping, demo]],
                        on=self.grouping,
                        how="left")
                    )
                    other_variables = (
                        other_variables + [var  for var in list(demo_df_dummies)
                                           if var != self.grouping]
                    )
            if self.tag_independent_variables:
                tag_df = (
                    processed[processed['tag'].isin(self.tag_independent_variables)]
                    [[self.grouping, 'tag']].drop_duplicates()
                )
                tag_dummy_df = pd.get_dummies(tag_df, 'tag')
                snippet_df_full = (
                    pd.merge(snippet_df_full,
                             tag_dummy_df,
                             on=self.grouping,
                             how='left').fillna(0)
                )
                other_variables = other_variables + [var  for var in list(tag_dummy_df)
                                           if var != self.grouping]
            if self.scout_group_independent_variables:
                scout_group_df = (
                    processed[processed['scout_group'].isin(self.scout_group_independent_variables)]
                    [[self.grouping, 'scout_group']].drop_duplicates()
                )
                scout_group_dummy_df = pd.get_dummies(scout_group_df, 'scout_group')
                snippet_df_full = (
                    pd.merge(snippet_df_full,
                             scout_group_dummy_df,
                             on=self.grouping,
                             how='left').fillna(0)
                )
                other_variables = other_variables + [var  for var in list(scout_group_dummy_df)
                                           if var != self.grouping]
        snippet_df_full["outcome"] = snippet_df_full.apply(
            lambda row: get_outcome(
                row[self.dependent_variable],
                self.positive_outcomes,
                self.negative_outcomes,
            ),
            axis=1,
        )

        return snippet_df_full[
            self.continous_question_list + other_variables + ["outcome"]
        ].dropna()
