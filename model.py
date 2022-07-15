import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

from prepare import split_data

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)

random_state = 42
positive = 1

# for this first iteration we will use only features which were observed to be potentially useful
# during exploration

def encode_data(df):
    '''
    This function takes in our dataset and encodes a given set of 
    categorical features using pandas one-hot encoder. It drops
    the original un-encoded columns and returns the df. 
    '''
    # categorical variables (that aren't already binary True/False)
    cols_to_encode = [
                      'propertytype', 
                      'propertysubtype', 
                     ]
    
    # create encoded column for each feature
    for col in cols_to_encode:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        # add encoded column to df
        df = pd.concat([df, dummy_df], axis=1)
        # drop original column
        df = df.drop(columns=col)
        
    return df

def convert_bools(df):
    '''
    This function takes in our dataset and converts all boolean columns to 1 or 0
    numeric datatypes, then returns the df.
    '''
    # identify boolean columns
    bools = [col for col in df.columns if df[col].dtype == 'bool']
    # convert to 1 or 0
    for col in bools:
        df[col] = df[col].map({True: 1, False: 0})
    return df

def scale_data(train, test, scaler_type=MinMaxScaler()):
    '''
    This takes in the train and test dataframes. 

    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the trainand test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 

    train and test dataframes are returned, in that order. 
    '''
    # identify quantitative features to scale (that aren't already scaled)
    cols_to_scale = [
                     'lotsizearea', 
                     'bedroomstotal', 
                     'bathroomstotalinteger',
                     'bathroomsfull',
                     'bathroomshalf', 
                     'livingarea',
                     'stories', 
                     'yearbuilt',
                     'years_since_build', 
                     'garage_size', 
                     'central_cooling_units', 
                     'windowwall_cooling_units',
                     'listing_month',
                     'listing_dayofmonth', 
                     'listing_dayofweek'
                    ]
    
    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    test_scaled = pd.DataFrame(index=test.index)
    
    # screate and fit the scaler
    scaler = scaler_type.fit(train[cols_to_scale])
    
    # adding scaled features to scaled dataframes
    train_scaled[cols_to_scale] = scaler.transform(train[cols_to_scale])
    test_scaled[cols_to_scale] = scaler.transform(test[cols_to_scale])
    
    # add 'scaled' prefix to columns
    for feature in cols_to_scale:
        train_scaled = train_scaled.rename(columns={feature: f'scaled_{feature}'})
        test_scaled = test_scaled.rename(columns={feature: f'scaled_{feature}'})
        
    # concat scaled feature columns to original train and test df's
    train = pd.concat([train, train_scaled], axis=1)
    test = pd.concat([test, test_scaled], axis=1)
    
    # drop the original columns
    train = train.drop(columns=cols_to_scale)
    test = test.drop(columns=cols_to_scale)

    return train, test

def prep_for_modeling(df):

    cols_to_drop = [
                'address_id',               # unique identifier not useful
                'listingcontractdate',      # we'll use engineered date features instead
                'originallistprice',         # we'll use the scaled prices instead
                'originallistprice_persqft', # we'll use the scaled prices instead
                'originallistprice_scaled', # we'll use the persqft prices instead
                ]

    df = df.drop(columns=cols_to_drop)
    df = encode_data(df)
    df = convert_bools(df)
    train, test = split_data(df)
    train, test = scale_data(train, test)
    return train, test



def run_baseline_1(train,
                   target,
                   positive,
                   model_info,
                   model_results):
    '''
    This function takes in the train sample, the target variable label, the positive condition label,
    an initialized model_number variable, as well as model_info and model_results dataframes dataframes that will be used for 
    storing information about the models. It then performs the operations necessary for making baseline predictions
    on our dataset, and stores information about our baseline model in the model_info and model_results dataframes. 
    (i.e. predicts the most common class)
    The model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # separate each sample into x (features) and y (target)
    x_train = train.drop(columns=target)
    y_train = train[target]


    # store baseline metrics

    # identify model number
    model_number = 'baseline_1'
    #identify model type
    model_type = 'baseline_1'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)

    # establish baseline predictions for train sample
    y_pred = pd.Series([train[target].mode()[0]]).repeat(len(train))

    # get metrics

    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'metric_type': 'accuracy',
           'score': sk.metrics.accuracy_score(y_train, y_pred)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'precision',
           'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'recall',
           'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'f1_score',
           'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    # set the model number to from 'baseline' to 0 
    model_number = 0
    
    return model_number, model_info, model_results

def run_baseline_2(train,
                   target,
                   positive,
                   model_info,
                   model_results):
    '''
    This function takes in the train sample, the target variable label, the positive condition label,
    an initialized model_number variable, as well as model_info and model_results dataframes dataframes that will be used for 
    storing information about the models. It then performs the operations necessary for making baseline predictions
    on our dataset, and stores information about our baseline model in the model_info and model_results dataframes. 
    The model_number, model_info, and model_results variables are returned (in that order). 
    
    For this alternative baseline, we will maximize recall by always predicting 1.
    '''

    # separate each sample into x (features) and y (target)
    x_train = train.drop(columns=target)
    y_train = train[target]

    # store baseline metrics

    # identify model number
    model_number = 'baseline_2'
    #identify model type
    model_type = 'baseline_2'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)

    # establish baseline predictions for train sample
    y_pred = pd.Series(1).repeat(len(train))

    # get metrics

    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'metric_type': 'accuracy',
           'score': sk.metrics.accuracy_score(y_train, y_pred)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'precision',
           'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'recall',
           'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'f1_score',
           'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    # set the model number to from 'baseline' to 0 
    model_number = 0
    
    return model_number, model_info, model_results

def run_model_1(train,
                target,
                positive,
                model_info,
                model_results):
    '''
    This function predicts whether a property is new construction based only on whether the build year is within
    two calendar years of the listing. This will create a more effective and useful baseline for which to compare
    future, more complex models. 
    
    This function takes in the train sample, the target variable label, the positive condition label,
    as well as the model_number variable and model_info and model_results dataframes. It then updates and returns
    the model_number, model_info, and model_results variables after creating and storing info about the model 
    described above.
    '''

    # separate each sample into x (features) and y (target)
    x_train = train.drop(columns=target)
    y_train = train[target]

    # identify model number
    model_number = 1
    #identify model type
    model_type = 'simple build year'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)

    # establish predictions for train sample
    y_pred = train.built_last_two_years

    # get metrics

    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'metric_type': 'accuracy',
           'score': sk.metrics.accuracy_score(y_train, y_pred)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'precision',
           'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'recall',
           'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'metric_type': 'f1_score',
           'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)
    
    return model_number, model_info, model_results


def display_model_results(model_results):
    '''
    This function takes in the model_results dataframe. This is a dataframe in tidy data format 
    containing the following information for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index as metric_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)

def baseline_models(train, target):
    
    model_info = pd.DataFrame()
    model_results = pd.DataFrame()

    model_number, model_info, model_results = run_baseline_1(train,
                                                            target,
                                                            positive,
                                                            model_info,
                                                            model_results)
    model_number, model_info, model_results = run_baseline_2(train,
                                                            target,
                                                            positive,
                                                            model_info,
                                                            model_results)
    model_number, model_info, model_results = run_model_1(train,
                                                            target,
                                                            positive,
                                                            model_info,
                                                            model_results)

    return display_model_results(model_results)


def decision_tree(train, target, features):
    # split dataset into x (features) and y (target)
    x_train = train[features]
    y_train = train[target]

    # identify model_type
    model_type = 'decision tree'

    # set hyperparameter ranges
    parameter_space = {'max_depth': [2,3,4,5,6,7]}

    # create the classifier
    clf = DecisionTreeClassifier()

    # define scoring methods
    scoring = {'recall': make_scorer(sk.metrics.accuracy_score),
               'precision': make_scorer(sk.metrics.precision_score),
               'accuracy': make_scorer(sk.metrics.accuracy_score),
               'f1_score': make_scorer(sk.metrics.f1_score)}

    # create and fit the GridSearchCV object
    grid = GridSearchCV(clf, parameter_space, cv=5, 
                        scoring=scoring,
                        refit='recall')
    grid.fit(x_train, y_train)


    # get results and store as dataframe

    results = grid.cv_results_

    params = results['params']
    accuracy = results['mean_test_accuracy'] 
    recall = results['mean_test_recall']
    precision = results['mean_test_precision']
    F1_score = results['mean_test_f1_score']

    for par, acc, rec, prec, f1 in zip(params, accuracy, recall, precision, F1_score):
        par['model_type'] = model_type
        par['features'] = features
        par['accuracy'] = acc
        par['recall'] = rec
        par['precision'] = prec
        par['F1_score'] = f1

    decision_tree_results = pd.DataFrame(params)
    
    return decision_tree_results

def random_forest(train, target, features):
    # split dataset into x (features) and y (target)
    x_train = train[features]
    y_train = train[target]

    # identify model_type
    model_type = 'random forest'

    # set hyperparameter ranges
    parameter_space = {'max_depth': [2,3,4,5,6,7],
                       'min_samples_leaf': [2,3,4]}

    # create the classifier
    clf = RandomForestClassifier()

    # define scoring methods
    scoring = {'recall': make_scorer(sk.metrics.accuracy_score),
               'precision': make_scorer(sk.metrics.precision_score),
               'accuracy': make_scorer(sk.metrics.accuracy_score),
               'f1_score': make_scorer(sk.metrics.f1_score)}

    # create and fit the GridSearchCV object
    grid = GridSearchCV(clf, parameter_space, cv=5, 
                        scoring=scoring,
                        refit='recall')
    grid.fit(x_train, y_train)


    # get results and store as dataframe

    results = grid.cv_results_

    params = results['params']
    accuracy = results['mean_test_accuracy'] 
    recall = results['mean_test_recall']
    precision = results['mean_test_precision']
    F1_score = results['mean_test_f1_score']

    for par, acc, rec, prec, f1 in zip(params, accuracy, recall, precision, F1_score):
        par['model_type'] = model_type
        par['features'] = features
        par['accuracy'] = acc
        par['recall'] = rec
        par['precision'] = prec
        par['F1_score'] = f1

    random_forest_results = pd.DataFrame(params)
    
    return random_forest_results

def log_regression(train, target, features):
    # split dataset into x (features) and y (target)
    x_train = train[features]
    y_train = train[target]

    # identify model_type
    model_type = 'logistic regression'

    # set hyperparameter ranges
    parameter_space = {'C': [.001, .01, .1, 1, 10, 100, 1000]}

    # create the classifier
    clf = LogisticRegression()

    # define scoring methods
    scoring = {'recall': make_scorer(sk.metrics.accuracy_score),
               'precision': make_scorer(sk.metrics.precision_score),
               'accuracy': make_scorer(sk.metrics.accuracy_score),
               'f1_score': make_scorer(sk.metrics.f1_score)}

    # create and fit the GridSearchCV object
    grid = GridSearchCV(clf, parameter_space, cv=5, 
                        scoring=scoring,
                        refit='recall')
    grid.fit(x_train, y_train)


    # get results and store as dataframe

    results = grid.cv_results_

    params = results['params']
    accuracy = results['mean_test_accuracy'] 
    recall = results['mean_test_recall']
    precision = results['mean_test_precision']
    f1_score = results['mean_test_f1_score']

    for par, acc, rec, prec, f1 in zip(params, accuracy, recall, precision, f1_score):
        par['model_type'] = model_type
        par['features'] = features
        par['accuracy'] = acc
        par['recall'] = rec
        par['precision'] = prec
        par['F1_score'] = f1

    log_regression_results = pd.DataFrame(params)
    
    return log_regression_results


