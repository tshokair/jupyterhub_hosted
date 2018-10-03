import pandas as pd
import numpy as np
from sklearn import preprocessing

def rename_question(part_num, question_num):
    return str(part_num + 1)+'-'+str(question_num + 1)

def convert_numerical_response(response):
    try:
        resp_clean  = int(response[0:2])
    except ValueError:
        resp_clean = response
    return resp_clean


def convert_something_else_response(response, choices):

    if str(response).lower().startswith('something else') or\
        (not any([str(response).lower().startswith(str(choice).lower().split()[0])
                  for choice in choices]) and
         not any([str(response).lower() == str(choice).lower() for choice in choices])):
        return 'other'
    return response

def get_outcome(val, positive_outcomes, negative_outcomes):
    if val in positive_outcomes or str(val) in positive_outcomes:
        return 1
    elif val in negative_outcomes or str(val) in negative_outcomes:
        return 0
    return -1

def rename_categorical(question_name, response):

    q_name = str(question_name)+'_'+str(response)
    return q_name

def fill_non_null(cell_value):
    if cell_value != -1:
        return 1
    return 0

def drop_low_var_dummies(pivoted_df):
    included_columns = ['snippet_id']
    for feature in list(pivoted_df):
        if feature != 'snippet_id':
            if pivoted_df[feature].sum() > 2 and\
                pivoted_df[feature].sum() < (len(pivoted_df) - 2):
                included_columns.append(feature)
            else:
                print(feature, "column sum", pivoted_df[feature].sum(), len(pivoted_df))

    return pivoted_df[included_columns]

def fill_dummy_values(pivoted_df):
    for feature in list(pivoted_df):
        if feature != 'snippet_id':
            pivoted_df[feature] =\
                pivoted_df.apply(lambda row: fill_non_null(row[feature]),
                                 axis = 1)
    print(pivoted_df)
    return drop_low_var_dummies(pivoted_df)

def make_dummies(categorical_df):
    categorical_df['question_name'] =\
        categorical_df.apply(lambda row: rename_categorical(row['question_name'],
                                                              row['response_clean']
                                                             ),
                               axis = 1
                               )

    categorical_df = categorical_df.drop_duplicates(['snippet_id','question_name'])

    pivoted = categorical_df.pivot(index='snippet_id',
                        columns='question_name',
                        values='response_clean').reset_index().fillna(-1)
    return fill_dummy_values(pivoted)


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

def get_first_choice(structure):
    try:
        return structure['choices'][0]
    except KeyError:
        return 0

def get_question_choices(structure):
    try:
        return structure['choices'][::-1]
    except KeyError:
        return [int(i) for i in np.linspace(0,100, 101)]

class MixedFeatureData():

    def __init__(self, json, dependent_variable,
                 continuous_independent_variables,
                 binary_independent_variables,
                 categorical_independent_variables,
                 positive_outcomes,
                 negative_outcomes):
        self.raw = pd.read_json(json)
        self.dependent_variable = dependent_variable
        self.encoded_dependent_variable = dependent_variable + '_enc'
        self.binary_independent_variables = binary_independent_variables
        self.continuous_independent_variables = continuous_independent_variables\
            + binary_independent_variables
        self.categorical_independent_variables = categorical_independent_variables
        self.positive_outcomes = positive_outcomes
        self.negative_outcomes = negative_outcomes
        self.continous_question_list = [dependent_variable] + self.continuous_independent_variables
        self.full_feature_df = self.full_features()
        self.dummies = [col for col in list(self.full_feature_df)
           if any(q_num in col for q_num in self.categorical_independent_variables)]
        self.raw_question_list = self.continous_question_list + categorical_independent_variables
        self.question_list = self.continous_question_list+ self.dummies
        self.encoded_independent_variables =\
            [indep_var+'_enc' for indep_var in self.continuous_independent_variables] +\
            self.dummies
        self.independent_variables =\
            self.continuous_independent_variables +\
            self.dummies
        self.encoded_features = self.encoded_features()

    def processed(self):
        processed_df = self.raw.copy()
        #processed_df = unlistify(processed_df, 'answers')
        processed_df['first_choice'] =\
            processed_df.apply(lambda row: get_first_choice(row['structure']), axis = 1)
        processed_df['choice_list'] =\
            processed_df.apply(lambda row: get_question_choices(row['structure']), axis = 1)
        processed_df['question_name'] =\
            processed_df.apply(lambda row: rename_question(row['part_position'],
                                                    row['question_position']),
                        axis = 1)
        processed_df['response'] =\
            processed_df.apply(lambda row: row['answers'][0],
                        axis = 1)
        processed_df['response_clean'] =\
            processed_df.apply(lambda row: convert_numerical_response(row['answers'][0]),
                        axis = 1)
        processed_df['response_clean'] =\
            processed_df.apply(lambda row:
                               convert_something_else_response(row['response_clean'],
                                                               row['choice_list']),
                               axis = 1)
        return processed_df

    def full_features(self):
        processed = self.processed()
        snippet_df_continuous =\
            processed[processed['question_name'].isin(self.continuous_independent_variables
                                                      + [self.dependent_variable])]\
                .pivot(index='snippet_id',
                       columns='question_name',
                       values='response_clean').reset_index().dropna()
        if self.categorical_independent_variables:
            snippets_categorical =\
                processed[processed['question_name'].isin(self.categorical_independent_variables)].copy()
            snippet_df_categorical = make_dummies(snippets_categorical)

            snippet_df_full = pd.merge(snippet_df_continuous, snippet_df_categorical,
                                       on='snippet_id', how='outer')
            dummies = [col for col in list(snippet_df_full)
                       if any(q_num in col for q_num in self.categorical_independent_variables)]
        else:
            snippet_df_full = snippet_df_continuous.copy()
            dummies = []
        snippet_df_full['outcome'] =\
            snippet_df_full.apply(lambda row: get_outcome(row[self.dependent_variable],
                                                          self.positive_outcomes,
                                                          self.negative_outcomes),
                                  axis = 1)
        return snippet_df_full[self.continous_question_list + dummies + ['outcome']]#.fillna('No Answer')

    def question_choices(self):
        processed = self.processed()
        choices = dict()
        for question in self.raw_question_list:
            choices[question] = get_question_choices(processed[processed['question_name'] == question]
                           ['structure'].iloc[0])
        return choices


    def encoders(self):
        full_features = self.full_feature_df
        encoder_dict = dict()
        for question in self.question_list:
            #print(question, len(full_features[~pd.notnull(full_features[question])]))
            #print(full_features[pd.notnull(full_features[question])][question].unique())
            label_encoder = preprocessing.LabelEncoder()
            encoder_dict[question] = label_encoder.fit(full_features[question].astype(str))
        return encoder_dict

    def encoded_features(self):
        encoders = self.encoders()
        feature_copy = self.full_feature_df.copy()
        for question in self.question_list:
            col_name = question
            feature_copy[col_name] =\
                encoders[question].transform(feature_copy[question].astype(str))
        return feature_copy

    def independent_variable_stats(self):
        indep_stats = dict()
        for variable in self.independent_variables:
            std = self.encoded_features[variable].std()
            mu = self.encoded_features[variable].mean()
            indep_stats[variable] = {'mu':mu, 'std':std}
        return indep_stats


    def standardized_features(self):
        feature_copy = self.encoded_features.copy()
        independent_variable_stats = self.independent_variable_stats()

        for indep in self.independent_variables:
            std = independent_variable_stats[indep]['std']
            mu = independent_variable_stats[indep]['mu']
            feature_copy[indep] =\
                feature_copy.apply(lambda row: (row[indep]- mu)/std, axis = 1)
            feature_copy[indep] = feature_copy[indep].fillna(-1)
        return feature_copy

    def training_features(self):
        training_df = self.standardized_features().copy()
        training_df['sum'] = 0.0
        for indep in self.independent_variables:
            training_df['sum'] = training_df.apply(lambda row: row['sum'] +
                                               abs(row[indep]),
                                               axis = 1)
        training_df['avg'] =\
            training_df.apply(lambda row:row['sum']/len(self.independent_variables),
                             axis = 1)
        return training_df[training_df['avg'] < 2]
