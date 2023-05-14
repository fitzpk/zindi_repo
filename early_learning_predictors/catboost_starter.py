# ********************************************
# Zindi Early Learning Predictors Competition
# --------------------------------------------
# Starter example script on how to approach the
# problem using a CatBoostRegressor
# ********************************************


# -------------------------------------
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import os
from datetime import datetime
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import geopy.distance

from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor, Pool, cv
import optuna
import patsy
import re
from category_encoders import CountEncoder, TargetEncoder, GLMMEncoder, JamesSteinEncoder, MEstimateEncoder, LeaveOneOutEncoder, CatBoostEncoder, WOEEncoder, HelmertEncoder, SumEncoder

import shap

# -------------------------------------
# IMPORT DATA & PRE-PROCESSING

dir_path = os.path.dirname(os.path.realpath(__file__))

train = pd.read_csv(dir_path+'/Train.csv')
test = pd.read_csv(dir_path+'/Test.csv')

metadata = pd.read_csv(dir_path+'/VariableDescription.csv')
sample_sub = pd.read_csv(dir_path+'/SampleSubmission.csv')


train_labels = train['target']
train.drop(['target'],axis=1,inplace=True)



# for each variable in metadata list
for index,row in metadata.iterrows():

  # tranform to categorical
  if row['Transformations'] == 'Convert to Categorical':
    train[row['Variable Name']] = train[row['Variable Name']].astype('object')
    test[row['Variable Name']] = test[row['Variable Name']].astype('object')
  
  # tranform to datetime
  elif row['Transformations'] == 'Convert to Date':
    train[row['Variable Name']] = train[row['Variable Name']].astype('datetime64')
    test[row['Variable Name']] = test[row['Variable Name']].astype('datetime64')

  # drop columns
  elif row['Transformations'] == 'Drop':
  
    if row['Ideas'] == 'Clean up before using':

      train[row['Variable Name']+'_flag'] = np.where(train[row['Variable Name']].notna() == True, 1, 0)
      test[row['Variable Name']+'_flag'] = np.where(test[row['Variable Name']].notna() == True, 1, 0)


    train.drop([row['Variable Name']],axis=1,inplace=True)
    test.drop([row['Variable Name']],axis=1,inplace=True)

    
  # clean up some text in a few ordinal columns
  elif row['Transformations'] == 'Convert to Ordinal (Clean First)':
    train[row['Variable Name']] = train[row['Variable Name']].str.split('<b>').str[1].str.strip().str.split('</b>').str[0]
    test[row['Variable Name']] = test[row['Variable Name']].str.split('<b>').str[1].str.strip().str.split('</b>').str[0]

    for char in ['.',')','(',':',',']:
      train[row['Variable Name']] = train[row['Variable Name']].str.replace(char,'')
      test[row['Variable Name']] = test[row['Variable Name']].str.replace(char,'')



# Fill empty child_observant_total by using the sum of preceeding columns
# for some reason the columns that create this number are full but the total column has nulls
value_map = {'Almost never':0, 'Sometimes':1, 'Often':2, 'Almost always':3}
train_totals = pd.DataFrame()
test_totals = pd.DataFrame()

for col in ['child_observe_attentive','child_observe_concentrated','child_observe_diligent','child_observe_interested']:
  train_col_total = train[[col]][col].map(value_map)
  test_col_total = test[[col]][col].map(value_map)

  train_totals = pd.concat([train_totals,train_col_total],axis=1)
  test_totals = pd.concat([test_totals,test_col_total],axis=1)

train_fill = train_totals.sum(axis=1)
test_fill = test_totals.sum(axis=1)

train['child_observe_total'] = train_fill
test['child_observe_total'] = test_fill



# -------------------------------------
# PRELIMINARY FEATURE ENGINEERING


# number of PRA features that contain nulls 
train['pra_nulls'] = train.filter(regex='pra_').isna().sum(axis=1)
test['pra_nulls'] = test.filter(regex='pra_').isna().sum(axis=1)


# difference from major city/area longitude
train['capetown_lon_diff'] = train['longitude'] - 18.423300
test['capetown_lon_diff'] = test['longitude'] - 18.423300

train['lesotho_lon_diff'] = train['longitude'] - 28.332347
test['lesotho_lon_diff'] = test['longitude'] - 28.332347


# dual language indicator
train['dual_lang'] = np.where(train['child_languages'].fillna('').str.contains('\+') == True, 1, 0)
test['dual_lang'] = np.where(test['child_languages'].fillna('').str.contains('\+') == True, 1, 0)


# teacher_social_total X pqa_score_teaching
train['social_PQAteach'] = train['teacher_social_total']*train['pqa_score_teaching']
test['social_PQAteach'] = test['teacher_social_total']*test['pqa_score_teaching']


# cap teacher duration
train['teacher_duration'] = np.clip(train['teacher_duration'], a_min=0, a_max=2682)
test['teacher_duration'] = np.clip(test['teacher_duration'], a_min=0, a_max=2682)


# space per child
train['space_per_child'] = train['pra_class_space']/train['pra_class_size']
test['space_per_child'] = test['pra_class_space']/test['pra_class_size']




# -------------------------------------
# IMPUTATION USING KNN FOR NUMERICS AND MISSING FLAG FOR CATEGORICALS

def knn_noneTag_imputation(train_set,test_set):

    numeric_cols = list(train_set.select_dtypes('number').columns.values)

    # KNN works best with normalized data so we will normalize, fit to KNN imputer, and then reverse the normalization
    scaler = StandardScaler()
    scaled_train_nums = pd.DataFrame(scaler.fit_transform(train_set[numeric_cols]), columns = train_set[numeric_cols].columns, index=train_set.index)
    scaled_test_nums = pd.DataFrame(scaler.transform(test_set[numeric_cols]), columns = test_set[numeric_cols].columns, index=test_set.index)

    # Fill missing numeric with nearest neighbour values
    knn_imputer = KNNImputer(n_neighbors=5).fit(scaled_train_nums)
    train_num = pd.DataFrame(knn_imputer.transform(scaled_train_nums), columns = scaled_train_nums.columns, index=train_set.index)
    test_num = pd.DataFrame(knn_imputer.transform(scaled_test_nums), columns = scaled_test_nums.columns, index=test_set.index)

    # Now reverse scaling using KNN output
    train_num = pd.DataFrame(scaler.inverse_transform(train_num), columns = train_num.columns, index=train_set.index)
    test_num = pd.DataFrame(scaler.inverse_transform(test_num), columns = test_num.columns, index=test_set.index)

    # Now simply fill with 'None' for the categorical features
    train_cats = train_set.select_dtypes('object')
    test_cats = test_set.select_dtypes('object')

    train_cats = train_cats.fillna(value='missing')
    test_cats = test_cats.fillna(value='missing')

    train_df = pd.concat([train_num,train_cats],axis=1)
    test_df = pd.concat([test_num,test_cats],axis=1)

    return train_df,test_df




# -------------------------------------
# DATE FEATURE EXTRACTION

def date_extraction(train_set,test_set):
  train_set['child_month_elom'] = train_set['child_date'].dt.month
  test_set['child_month_elom'] = test_set['child_date'].dt.month
  

  # Get day of the week in which elom took place
  train_set['child_day_elom'] = train_set['child_date'].dt.day_name()
  test_set['child_day_elom'] = test_set['child_date'].dt.day_name()

  # month child was born in (using number ended up working better)
  train_set['child_month_dob'] = train_set['child_dob'].dt.month
  test_set['child_month_dob'] = test_set['child_dob'].dt.month


  train_set['child_date_unix'] = (train_set['child_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')
  test_set['child_date_unix'] = (test_set['child_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')

  train_set['obs_date_unix'] = (train_set['obs_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')
  test_set['obs_date_unix'] = (test_set['obs_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')

  train_set['pri_date_unix'] = (train_set['pri_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')
  test_set['pri_date_unix'] = (test_set['pri_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')

  train_set['pra_date_unix'] = (train_set['pra_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')
  test_set['pra_date_unix'] = (test_set['pra_date'] - pd.Timestamp('1970-01-01')).astype('timedelta64[s]')



  return train_set, test_set



# -------------------------------------
# ORDINAL ENCODING


def ordinal_encoder(x_train,x_test):

    # Create flag before concat
    x_train['trainTest'] = 'train'
    x_test['trainTest'] = 'test'

    # Concat into one
    input_data = pd.concat([x_train,x_test],axis=0)


    # Ordinal column mappings
    ordinal_cols = [
        'child_years_in_programme',
        'child_observe_attentive',
        'child_observe_concentrated',
        'child_observe_diligent',
        'child_observe_interested',
        'pra_engaged',
        'pra_salary',
        'pra_motivate_support',
        'pra_motivate_recognition',
        'pra_motivate_mentoring',
        'pri_separate',
        'pri_bank',
        'pri_registered_partial',
        'pri_registered_programme',
        'pri_registered_dsd',
        'pri_attendance',
        'pri_fees_paid_proportion',
        'pri_education',
        'pri_same_language',
        'pri_location',
        'pri_support_dsd',
        'pri_support_dbe',
        'pri_support_municipality',
        'pri_support_ngo',
        'pri_staff_employed',
        'pri_covid_staff_salaries',
        'pri_food_parents_breakfast',
        'pri_food_parents_morning',
        'pri_food_parents_lunch',
        'pri_food_parents_afternoon',
        'pri_parents_frequency',
        'pri_parents_contact',
        'pri_parents_activities',
        'pri_support_frequency',
        'teacher_social_peers',
        'teacher_social_nonaggressive',
        'teacher_social_cooperate',
        'teacher_social_assistance',
        'teacher_social_ideas',
        'teacher_social_initiative',
        'teacher_emotional_understand',
        'teacher_emotional_appropriate',
        'teacher_emotional_independent',
        'teacher_emotional_adjust',
        'teacher_emotional_confidence',
        'teacher_emotional_selfstarter',
        'teacher_social_met',
        'teacher_emotional_met',
        'teacher_selfcare_met',


        'obs_water',
        'obs_building',
        'obs_condition_equipment',
        'obs_material_display',
        
        'grade_r',

        'certificate_registration_partial',
        'certificate_registration_program',
        'certificate_registration_npo',
        'certificate_register',


        'urban',
        'plan',
        'ses_cat',
        
        'pqa_relationships_peers',
        'pqa_relationships_staff',
        'pqa_relationships_acknowledge',
        'pqa_relationships_discipline',

        'pri_transport',
        'pra_previous',

        
        ]
    
    ordinal_dicts = [
        {"missing":-1, 'Do Not Know':0, '1st year in the programme':1, '2nd year in programme':2, '3rd year in programme':3},
        {'Almost never':1, 'Sometimes':2, 'Often':3, 'Almost always':4},
        {'Almost never':1, 'Sometimes':2, 'Often':3, 'Almost always':4},
        {'Almost never':1, 'Sometimes':2, 'Often':3, 'Almost always':4},
        {'Almost never':1, 'Sometimes':2, 'Often':3, 'Almost always':4},
        {'missing':0, 'Seldom':1, 'Sometime':2, 'Often':3},
        {'missing':-1, 'R0':0, 'Less than R500 per month':1, 'R500 – R749':2, 'R750 – R999':3, 'R1000 – R1249':4, 'R1250 – R1499':5, 'R1500 – R1999':6, 'R2000 – R2499':7, 'R2500 – R2999':8, 'R3000 – R3499':9, 'R3500 – R3999':10, 'R4000 – R4449':11, 'R 5000 – R5999':12, 'More than R6000':13},
        {'missing':0, 'Disagree strongly':1, 'Disagree a little':2, 'Agree a little':3, 'Agree strongly':4},
        {'missing':0, 'Disagree strongly':1, 'Disagree a little':2, 'Agree a little':3, 'Agree strongly':4},
        {'missing':0, 'Disagree strongly':1, 'Disagree a little':2, 'Agree a little':3, 'Agree strongly':4},
        {'missing':0, 'No, children of all ages are learning and playing together':1, 'Yes, children are grouped by age but are using the same space':2, 'Yes, children are grouped by age and divided into different rooms':3},
        {'missing':0, "No, we don't use a bank account":1, 'No ,the principal uses his/her ount account':2, 'Yes':3},
        {'missing':0, 'Not registered':1, 'Lapsed registration':2, 'In process':3, 'Conditionally registered':4, 'Fully registered':5},
        {'missing':0, 'Not registered':1, 'Lapsed registration':2, 'In process':3, 'Conditionally registered':4, 'Fully registered':5},
        {'missing':0, 'Not registered':1, 'Lapsed registration':2, 'In process':3, 'Conditionally registered':4, 'Fully registered':5},
        {'missing':0, 'Once a week':1, 'Two times a week':2, 'Three times a week':3, 'Four times a week':4, 'Five times a week':5},
        {'missing':-1, 'None of the parents':0, 'Only a few parents':1, 'About half of the parents':2, 'Most parrents, but not all':3, 'All parents':4},
        {'missing':0, 'Below Grade 12/matric':1, 'Matric/National Senior Certificate':2, 'Certificate':3, 'Diploma':4, 'Undergraduate Degree':5, 'Postgraduate degree':6},
        {'missing':-1, 'None of the children speak this language at home':0, 'Less than half of the children speak this language at home':1, 'More than half of the children speak this language at home':2, 'All children speak this language at home':3},
        {'missing':-1, 'Other':0, 'It is in the garage of someone’s house':1, 'It is in someone’s house':2, 'It is at a church / mosque / place of worship':3, 'It is a municipal building':4, 'It is at a community hall / centre':5, 'It is an ECD Centre':6},
        {'missing':-1, 'Never':0, 'Once':1, 'Twice':2, 'Three times':3, 'More than three times':4},
        {'missing':-1, 'Never':0, 'Once':1, 'Twice':2, 'Three times':3, 'More than three times':4},
        {'missing':-1, 'Never':0, 'Once':1, 'Twice':2, 'Three times':3, 'More than three times':4},
        {'missing':-1, 'Never':0, 'Once':1, 'Twice':2, 'Three times':3, 'More than three times':4},
        {'missing':0, 'I have less than half of the number of staff employed compared to previous years':1, 'I have about the same number of staff employed compared to previous years':2, 'I have more staff employed compared to previous years':3, 'I have more than half of the number of staff employed compared to previous years':4},
        {'missing':-1, 'No, I have not been able to pay them at all':0, 'No, I had to reduce their salaries':1, 'Yes, their salaries have remained the same':2},
        {'missing':-1, 'None':0, 'Some':1, 'All':2},
        {'missing':-1, 'None':0, 'Some':1, 'All':2},
        {'missing':-1, 'None':0, 'Some':1, 'All':2},
        {'missing':-1, 'None':0, 'Some':1, 'All':2},
        {'missing':-1, 'Never':0, 'Other':1, 'Annually':2, 'Quarterly':3, 'Monthly':4, 'Weekly':5},
        {'missing':-1, 'No, no-one has asked me this year.':0, 'Other':1, 'Yes, but only one or two have asked me this year.':2, 'Yes, some parents have asked me this year.':3, 'Yes, most parents have asked me this year.':4},
        {'missing':-1, 'No, no-one has asked me this year.':0, 'Other':1, 'Yes, but only one or two have asked me this year.':2, 'Yes, some parents have asked me this year.':3, 'Yes, most parents have asked me this year.':4},
        {'missing':-1, 'Other':1, 'Once a year':2, 'Once a term':3, 'Once a month':4, 'Once a week':5},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':-1, 'None of the time':0, 'A little of the time':1, 'Most of the time':2, 'All of the time':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Not True':1, 'Sometimes True':2, 'Often True':3},
        {'missing':0, 'Does not meet standard':1, 'Meets the standard':2},
        {'missing':0, 'Does not meet standard':1, 'Meets the standard':2},
        {'missing':0, 'Does not meet standard':1, 'Meets the standard':2},

        # facility features
        {'missing':-1, 'None':0, 'Other':1, 'Public or communal tap off-site':2, 'Rain water tank on-site':3, 'Bore-hole water on-site':4, 'Tap water on-site/ outside the building':5, 'Tap water in the building':6},
        {'missing':0, 'Shipping container':1, 'Informal housing (shack)':2, 'Traditional, mortar or mud walls with zinc or thatch roof':3, 'Conventional, brick or block with tile or zinc roof':4, 'Prefab building':5},
        {'missing':0, 'Bad (Mostly broken and unused)':1, 'Okay (some in working condition)':2, 'Fine (mostly in working condition)':3, 'Very good':4},
        {'missing':0, 'No, there is nothing against the walls':1, 'Yes, but of poor quality and not relevant to the age group':2, 'Yes, it is of average quality and relevant to the age group':3, 'Yes, it is stimulating and appropriate':4},

        # grade_r
        {'missing':-1, 'No':0, 'Yes, but I was not able to see them':1, 'Yes, and I was able to see them':2},


        # certificates
        {'missing':-1, 'Does not exist':0, 'No, but it exists':1, 'Yes':2},
        {'missing':-1, 'Does not exist':0, 'No, but it exists':1, 'Yes':2},
        {'missing':-1, 'Does not exist':0, 'No, but it exists':1, 'Yes':2},
        {'missing':-1, 'Does not exist':0, 'No, but it exists':1, 'Yes':2},


        {'missing':-1, 'Rural':0, 'Urban':1},
        {'missing':-1, 'No':0, 'Yes, details provided for the main activities in the week':1, 'Yes, details provided for the activities each day':2},
        {'missing':0, 'R0-110':1, 'R111-290':2, 'R291-750':3, 'R751-1750':4, 'R1751+':5},

        # pqa relationships
        {'missing':-1, 'Inadequate':0, 'Basic':1, 'Good':2},
        {'missing':-1, 'Inadequate':0, 'Basic':1, 'Good':2},
        {'missing':-1, 'Inadequate':0, 'Basic':1, 'Good':2},
        {'missing':-1, 'Inadequate':0, 'Basic':1, 'Good':2},

        # pri transport
        {'missing':-1, 'No':0, 'Yes':1},

        # pra previous
        {'missing':-1, 'Other':0, 'I was not employed, but I was looking for work':1, 'I was in school/ studying':2, 'I was self-employed':3,'I was working in another industry/ sector':4,'I was working in another school/ ECD programme':5},

        
        ]

    # for each column map overwrite the original column with the ordinal mapping
    for col,dictionary in zip(ordinal_cols,ordinal_dicts):
        input_data[col] = input_data[col].map(dictionary)
                



    # Split data up and drop flag column created earlier
    x_trainOutput = input_data[input_data['trainTest']=='train']
    x_trainOutput.drop(['trainTest'],axis=1,inplace=True)
    
    x_testOutput = input_data[input_data['trainTest']=='test']
    x_testOutput.drop(['trainTest'],axis=1,inplace=True)

    return x_trainOutput,x_testOutput




# -------------------------------------
# DUMMY ENCODER FOR CATEGORICAL VARIABLES

def dummy_encoder(x_trainData, x_testData):

    # Create flag before concat
    x_trainData['trainTest'] = 'train'
    x_testData['trainTest'] = 'test'

    # Concat into one
    input_data = pd.concat([x_trainData,x_testData],axis=0)
    
    # Loop through object columns and transform to dummy variable
    collector = pd.DataFrame()
    for col in input_data.select_dtypes('object'):
        if col != 'trainTest':
            col_dummies = pd.get_dummies(input_data[col], drop_first=True, prefix=col, prefix_sep='_')
            collector = pd.concat([collector, col_dummies], axis=1)

    # Combine encoded object data with numeric data
    output_data = pd.concat([input_data.select_dtypes(['number']),collector,input_data['trainTest']],axis=1)

    # Split data up and drop flag column created earlier
    x_trainOutput = output_data[output_data['trainTest']=='train']
    x_trainOutput.drop(['trainTest'],axis=1,inplace=True)
    
    x_testOutput = output_data[output_data['trainTest']=='test']
    x_testOutput.drop(['trainTest'],axis=1,inplace=True)

    return x_trainOutput, x_testOutput




# -------------------------------------
# SCALING

def stand_scaling(x_trainCV,x_valCV):

    # At this point all data will be numeric so no need to separate categorical features    
    for col in x_trainCV.columns:
        if col != 'child_id':
            
            stand_scaler = StandardScaler()
            x_trainCV[[col]] = stand_scaler.fit_transform(x_trainCV[[col]])
            x_valCV[[col]] = stand_scaler.transform(x_valCV[[col]])

    return x_trainCV, x_valCV


def robust_scaling(x_trainCV,x_valCV):

    # At this point all data will be numeric so no need to separate categorical features    
    for col in x_trainCV.columns:
        if col != 'child_id':
            
            rob_scaler = RobustScaler()
            x_trainCV[[col]] = rob_scaler.fit_transform(x_trainCV[[col]])
            x_valCV[[col]] = rob_scaler.transform(x_valCV[[col]])

    return x_trainCV, x_valCV



# -------------------------------------
# FINAL CLEANING

def cleanup(x_training, x_testing):

    for col in ['child_id','child_date','child_dob','child_enrolment_date',
                'pqa_date','pra_date','pri_date','obs_date']:
        if col in x_training.columns:
            x_training = x_training.drop([col],axis=1)
        if col in x_testing.columns:
            x_testing = x_testing.drop([col],axis=1)
    
    # some algos don't like certain characters in column names so remove them
    for char in ['<', '>', '[' ,']' ,'+', '.', ':', ',']:
        x_training.columns = x_training.columns.str.replace(char, '')
        
    for char in ['<', '>', '[' ,']' ,'+', '.', ':', ',']:
        x_testing.columns = x_testing.columns.str.replace(char, '')

    # Convert object types to categorical
    # note: this is done for when using lightgbm default encoding, also helps with memory
    cat_cols = list(x_training.select_dtypes('object'))
    for c in cat_cols:
      x_training[c] = x_training[c].astype('category')
      x_testing[c] = x_testing[c].astype('category')
        
    return x_training, x_testing



# **********************************************************
# MODELING

# transform data
X_train, X_test = date_extraction(train,test)
X_train, X_test = knn_noneTag_imputation(X_train, X_test)
X_train, X_test = ordinal_encoder(X_train, X_test)
X_train, X_test = dummy_encoder(X_train, X_test)


# fit and predict
catboost_model = CatBoostRegressor(verbose=False)
catboost_model.fit(X_train, train_labels)
y_pred = catboost_model.predict(X_test)

# explain model usinf SHAP
explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(X_test)
feat_importance = pd.DataFrame(shap_values, columns=X_test.columns)

# obtain top 15 most important features for each prediction
nlargest = 15
order = np.argsort(-feat_importance.values, axis=1)[:, :nlargest]
prediction_importances = pd.DataFrame(feat_importance.columns[order], 
                                      columns=['feature_{}'.format(i) for i in range(1, nlargest+1)],
                                      index=feat_importance.index)

# create submission file
submission = pd.concat([sample_sub['child_id'],sample_sub['target'],prediction_importances],axis=1)
submission['target'] = y_pred

submission.to_csv(dir_path+'/submission_catboost_starter.csv',index=False)

