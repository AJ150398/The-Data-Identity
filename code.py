# importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# importing data set
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# DATA PRE-PROCESSING

# check null values
train_null = train_set.isnull().sum()
test_null = test_set.isnull().sum()

# onehotencode
train_set_ohe = pd.DataFrame()
test_set_ohe = pd.DataFrame()

def my_ohe(X_train, X_test, column):
    global train_set_ohe
    global test_set_ohe
    
    train_set_ohe = pd.concat([train_set_ohe, pd.get_dummies(X_train[column], drop_first = True, prefix="_" + column)], axis = 1)
    test_set_ohe = pd.concat([test_set_ohe, pd.get_dummies(X_test[column], drop_first = True, prefix="_" + column)], axis = 1)

def my_combine(X_train, X_test, column):
    global train_set_ohe
    global test_set_ohe
    
    train_set_ohe = pd.concat([train_set_ohe, X_train[column]], axis = 1)
    test_set_ohe = pd.concat([test_set_ohe, X_test[column]], axis = 1)


# a) 'id'

# b) 'program_id'

# c) 'program_type' 
my_combine(train_set, test_set, 'program_type')

# d) 'program_duration'
my_combine(train_set, test_set, 'program_duration')

# e) 'test_id'
my_combine(train_set, test_set, 'test_id')

# f) 'test_type'
my_ohe(train_set, test_set, 'test_type')

# g) 'difficulty_level'
diff_map = {
'easy' : 4,
'intermediate' : 3,
'hard' : 2,
'vary hard' : 1
}

train_set['new_diff'] = train_set['difficulty_level'].map(diff_map)
test_set['new_diff'] = test_set['difficulty_level'].map(diff_map)

my_combine(train_set, test_set, 'new_diff')

# h) 'trainee_id'
my_combine(train_set, test_set, 'trainee_id')

# i) 'gender'
my_ohe(train_set, test_set, 'gender')

# j) 'education'
edu_map = {
'No Qualification' : 1,
'Masters' : 5,
'Bachelors' : 4,
'High School Diploma' : 3,
'Matriculation' : 2
}

train_set['new_edu'] = train_set['education'].map(edu_map)
test_set['new_edu'] = test_set['education'].map(edu_map)

my_combine(train_set, test_set, 'new_edu')

# k) 'city_tier'

city_map = {
1 : 4,
2 : 3,
3 : 2,
4 : 1,
}

train_set['new_city'] = train_set['city_tier'].map(city_map)
test_set['new_city'] = test_set['city_tier'].map(city_map)

my_combine(train_set, test_set, 'new_city')

# l) 'age'

train_set['age'].fillna(train_set['age'].median(), inplace = True)
test_set['age'].fillna(train_set['age'].median(), inplace = True)

my_combine(train_set, test_set, 'age')

# m) 'total_programs_enrolled' 
my_combine(train_set, test_set, 'total_programs_enrolled')

# n) 'is_handicapped'
my_ohe(train_set, test_set, 'is_handicapped')

# o) 'trainee_engagement_rating'

train_set['trainee_engagement_rating'].fillna(train_set['trainee_engagement_rating'].mode()[0], inplace = True)
test_set['trainee_engagement_rating'].fillna(train_set['trainee_engagement_rating'].mode()[0], inplace = True)

engage_dict = dict(train_set["is_pass"].groupby(train_set["trainee_engagement_rating"]).mean())
train_set['new_trainee_engagement_rating'] = train_set['trainee_engagement_rating'].map(engage_dict)
test_set['new_trainee_engagement_rating'] = test_set['trainee_engagement_rating'].map(engage_dict)

my_combine(train_set, test_set, 'new_trainee_engagement_rating')


# Convert to matrices of independent vars
X_train = train_set_ohe.iloc[:, :].values
X_test = test_set_ohe.iloc[:, :].values

y_train = train_set['is_pass'] # vector of dependent var

# fitting model 0
import catboost as cb
cat_index = [0, 1, 2, 5]

clf_cb = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=2,
                            depth=8, iterations= 128, l2_leaf_reg= 16, learning_rate= 0.15)

clf_cb.fit(X_train, y_train, cat_features= cat_index)

# predicting results
y_pred_cb = clf_cb.predict_proba(X_test)
y_pred_cb = y_pred_cb[:, 1]

# write to file
submission = pd.read_csv("sample_submission.csv")
submission['is_pass'] = y_pred_cb
submission.to_csv('sample_submission.csv', index=False)
