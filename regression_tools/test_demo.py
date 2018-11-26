import pandas as pd
from lib.build_regressions_features_demo import MixedFeatureData
from lib.explore_data import ExploratoryAnalysis
from lib.regressions import MixedClassificationModel


# CHANGE THESE VALUES
mission_id = "25614"
# ONLY USE SINGLE QUESTIONS FOR DEPENDENT AND INDEPENDENT VARIABLES IN FORMAT 'PART_NUM-QUESTION_NUM'
#continuous_independent_variables = ["4-5","4-6", "4-10",
#                                    "4-18","4-21"]
continuous_independent_variables = ["4-4", "4-5","4-6", "4-10",
                                    "4-11","4-12","4-13","4-14","4-15",
                                    "4-17","4-18","4-19","4-21"]
# NOTE MULTIPLE QUESTIONS NEED TO BE CATEGORICAL
categorical_independent_variables = []
binary_independent_variables = []
dependent_variable = "4-2"
negative_outcomes = [
"Moderately satisfied",
"Neither satisfied nor dissatisfied",
"Slightly dissatisfied",
"Moderately dissatisfied",
"Extremely dissatisfied"
]
positive_outcomes = ["Extremely satisfied"]
demo_independent_variables = ['age']
tag_independent_variables = []
#mutually exclusive scout groups only at this time
scout_group_independent_variables = []
response_data = pd.read_pickle("mission_" + mission_id + "_data.pkl")
question_response_filtering = {}
grouping = 'user_id'
ethnicity_filters = []
education_filters = []
tag_filters = []
scout_group_filters = ["Apple","Samsung","Bose","Pixel"]
print(len(response_data.groupby('user_id').first()))
print(len(response_data[response_data['scout_group'].isin(scout_group_filters)].groupby('user_id').first()))
print(response_data['scout_group'].unique())
if ethnicity_filters:
    ethnicities = ethnicity_filters
else:
    response_data['ethnicity'] = (
        response_data['ethnicity'].fillna('missing')
    )
    ethnicities = response_data['ethnicity'].unique()

if education_filters:
    educations = education_filters
else:
    response_data['education'] = (
        response_data['education'].fillna('missing')
    )
    educations = response_data['education'].unique()
if tag_filters:
    tags = tag_filters
else:
    response_data['tag'] = (
        response_data['tag'].fillna('None')
    )
    tags = response_data['tag'].unique()
if scout_group_filters:
    scout_groups = scout_group_filters
else:
    response_data['scout_group'] = (
        response_data['scout_group'].fillna('None')
    )
    scout_groups = response_data['scout_group'].unique()

print(ethnicities)
print(educations)
print(tags)
print(scout_groups)
filtered = response_data[(response_data['ethnicity'].isin(ethnicities)) & 
                        (response_data['education'].isin(educations)) &
                        (response_data['tag'].isin(tags)) &
                        (response_data['scout_group'].isin(scout_groups))].copy()
print(len(filtered))
filtered_id_list = []
for question in question_response_filtering:
    part, num = int(question.split('-')[0])-1, int(question.split('-')[1])-1
    response = question_response_filtering[question]
    ids = filtered[(filtered['part_position'] == part) &
                        (filtered['question_position'] == num) &
                        (any([response in x for x in filtered['answers']]))][grouping].unique()
    filtered_id_list = filtered_id_list + list(ids) 
    
if filtered_id_list:
    filtered_all = filtered[filtered["snippet_id"].isin(filtered_id_list)].copy()
else:
    filtered_all = filtered.copy()
print(len(filtered_all))
#print(list(response_data))
fd = MixedFeatureData(
        filtered_all,
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
    )
print(fd.encoded_features['age'].mean())
print(fd.independent_variables)
print(fd.dependent_variable)
eda = ExploratoryAnalysis(fd.encoded_features, fd.independent_variables,
                          fd.dependent_variable, fd.question_choices())

logistic_regression = MixedClassificationModel(fd)
#print(len(fd.raw.groupby(grouping).count()))
#print(len(fd.processed().groupby(grouping).count()))
#print(len(logistic_regression.balanced), len(fd.encoded_features), len(fd.full_feature_df) )
logistic_regression.print_results()