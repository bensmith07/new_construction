import pandas as pd
import numpy as np

def wrangle_data():

    df = pd.read_csv('data.csv')
    
    # define target column
    target = 'newconstructionyn'

    # remove columns where there is only one unique value, 
    # since such a feature cannot possibly help distinguish between target classes
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])

    # dropping additional columns which are not useful 
    # (because model would not be generalizable to other locations)
    cols = ['city', 'postal_code', 'census_tract']
    df = df.drop(columns=cols)

    # dropping columns which are not relevant/available at the time of a new listing
    cols = ['standardstatus', 
            'mlsstatus', 
            'contractstatuschangedate', 
            'purchasecontractdate', 
            'closedate', 
            'daysonmarket', 
            'closeprice', 
            'listprice']
    df = df.drop(columns=cols)

    ########################
    # handling null values #
    ######################## 
    # (for exploration and rationale, see prep_notebook.ipynb)

    # drop observations where the target value is null
    df = df[df[target].notnull()]

    # imputing 1 in garageyn where parkingfeatures is 'Attached' or 'Oversized'
    df['garageyn'] = np.where(df.parkingfeatures.isin(['Attached', 'Oversized']) & df.garageyn.isna(), 1, df.garageyn)
    # imputing 0 in garageyn where parkingfeatures is 'None/Not Applicable'
    df['garageyn'] = np.where((df.parkingfeatures == 'None/Not Applicable') & df.garageyn.isna(), 0, df.garageyn) 

    # dropping columns
    cols_to_drop = ['stories', 'lotfeatures'] 
    df = df.drop(columns=cols_to_drop)

    # imputing median
    cols_to_impute_median = ['lotsizearea']
    for col in cols_to_impute_median:
        df[col] = np.where(df[col].isna(), df[col].median, df[col])

    # dropping rows
    cols_to_drop_rows = [col for col in df.columns if (df[col].isna().sum() > 0) 
                                                    & (df[col].isna().sum() <=10)]
    cols_to_drop_rows.append('originallistprice')
    for col in cols_to_drop_rows:
        df = df[df[col].notnull()]

    # separating rentals vs. for-sale listings
    rent_df = df[df.propertytype == 'Residential Rental']
    sale_df = df[df.propertytype == 'Residential']
    # drop irrelevant column
    sale_df = sale_df.drop(columns='totalactualrent')

    return sale_df, rent_df