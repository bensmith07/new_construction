import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def wrangle_data(filename='data.csv'):
    '''
    This function pulls in San Antonio MLS listings data from a local csv 
    with a given filename and applies all necessary
    transformations to the data for future exploration and modeling. 

    Returns 3 dataframes in the following order: 
    1. 'df' (the whole dataset)
    2. 'sale_df' (for-sale listings only)
    3. 'rent_df' (for-rent listings only)
    
    These transformations include:

    REMOVING IRRELEVANT DATA

    - Removing the following columns due to irrelevancy to the problem at hand:
        - any column where there is only one unique value
        - city, postal_code, and census_tract
        - standardstatus, mlsstatus, contractstatuschangedate, purchasecontractdate, closedate,
          daysonmarket, closeprice, listprice

    - Removing the following columns due to an abundance of null values:
        - lotfeatures, totalactualrent
    
    HANDLING MISSING VALUES

    - Imputing missing values for the following columns:
        - garageyn (imputed based on information found in parkingfeatures column)
        - stories (based on information found in architecturalstyle column) 
        - lotsizearea (imputing with median)

    - Removing any other rows that remain with missing values 
      (at this point no column had more than 15 nulls)

    ADJUSTING DATA TYPES

    - address_id: integer >> string
    - newconstructionyn: float (0,1) >> boolean
    - garageyn: float (0,1) >> boolean
    - listingcontractdate: string >> pandas timestamp

    DATA CLEANING/PREP
    - propertysubtype: combining "Single Family Detached" and "Single Family Residence Detached"
                       to one single value
    - lotsizearea: correcting negative lotsize values (likely listed in error)
                   by taking the absolute value

    ENGINEERING NEW FEATURES

    including:
    (for definitions, see data dictionary in README.md)
    - listing_quarter
    - listing_month
    - listing_dayofmonth
    - listing_dayofweek
    - listed_on_weekend
    - years_since_build
    - built_last_two_years
    - parkingfeatures_attached
    - parkingfeatures_detached
    - parkingfeatures_oversized
    - parkingfeatures_converted
    - parkingfeatures_sideentry
    - parkingfeatures_rearentry
    - parkingfeatures_tandem
    - parkingfeatures_golfcart
    - heating_central
    - heating_naturalgas
    - heating_electric
    - heating_heatpump
    - heating_2units
    - heating_1unit
    - heating_zoned
    - heating_other
    - heating_floorfurnace
    - heating_solar
    - heating_propaneowned
    - heating_none
    - heating_windowunit
    - cooling_central
    - cooling_winddowwall
    - cooling_heatpump
    - cooling_zoned
    - central_cooling_units
    - winddowwall_cooling_units
    - archstyle_traditional
    - archstyle_splitlevel
    - archstyle_ranch
    - archstyle_texashillcountry
    - archstyle_craftsman
    - archstyle_other
    - archstyle_colonial
    - archstyle_spanish
    - archstyle_manufacturedhome-singlewide
    - archstyle_a-frame
    - lotsizearea_listed_negative
    - lotsizearea_small
    - originallistprice_persqft
    - originallistprice_scaled
    - originallistprice_scaled_persqft

    SEPARATING FOR-RENT AND FOR-SALE

    - Due to the incomparable ranges of the monthly rent prices
      and the property purchase prices found in the same column: 
      we created separate dataframe 'sale_df' and 'rent_df' with
      for-sale properties and for-rent properties respectively. 
      These separate dataframes can be used for exploration and 
      potentially modeling, while we also maintain the original 
      dataset in it's entirety as the 'df' variable.
    '''

    df = pd.read_csv(filename)

    # define target column for easier reference
    target = 'newconstructionyn'

    ############################
    # REMOVING IRRELEVANT DATA #
    ############################

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
    # HANDLING NULL VALUES #
    ######################## 
    # (for a detailed explanation of this process, see prep_notebook.ipynb)

    # drop observations where the target value is null
    df = df[df[target].notnull()]

    # imputing 1 in garageyn where parkingfeatures is 'Attached' or 'Oversized'
    df['garageyn'] = np.where(df.parkingfeatures.isin(['Attached', 'Oversized']) & df.garageyn.isna(), 
                                1, df.garageyn)
    # imputing 0 in garageyn where parkingfeatures is 'None/Not Applicable'
    df['garageyn'] = np.where((df.parkingfeatures == 'None/Not Applicable') & df.garageyn.isna(), 
                                0, df.garageyn) 

    # dropping columns
    cols_to_drop = ['lotfeatures', 'totalactualrent'] 
    df = df.drop(columns=cols_to_drop)

    # imputing median
    cols_to_impute_median = ['lotsizearea']
    for col in cols_to_impute_median:
        df[col] = np.where(df[col].isna(), df[col].median(), df[col])

    # dropping rows
    cols_to_drop_rows = [col for col in df.columns if (df[col].isna().sum() > 0) 
                                                    & (df[col].isna().sum() <=10)]
    cols_to_drop_rows.append('originallistprice')
    for col in cols_to_drop_rows:
        df = df[df[col].notnull()]

    # imputing "stories" based on info in "architecturalstyle"

    # create a new column which bases number of stories on "architecturalstyle" info
    def get_stories_to_fillna(archstyle):
        if 'One Story' in archstyle:
            return 1
        elif 'Two Story' in archstyle:
            return 2
        elif '3 or More' in archstyle:
            return 3
        else:
            return np.nan
    df['stories_to_fillna'] = df.architecturalstyle.apply(get_stories_to_fillna)
    # fill null values in stories column with the newly created stories_to_fillna column
    df['stories'] = df.stories.fillna(df.stories_to_fillna)
    # dropping rows where "stories" is still null
    df = df[df.stories.notnull()]
    # getting rid of the temporary "stories_to_fillna" column
    df = df.drop(columns=['stories_to_fillna'])

    ########################
    # ADJUSTING DATA TYPES #
    ########################

    # address_id from int to string
    df['address_id'] = df.address_id.astype(str)

    # garageyn and newconstructionyn to from float to boolean
    df['newconstructionyn'] = df.newconstructionyn.astype(bool)
    df['garageyn'] = df.garageyn.astype(bool)

    # listingcontractdate from string to pandas datetime
    df['listingcontractdate'] = pd.to_datetime(df.listingcontractdate)

    #######################
    # FEATURE ENGINEERING #
    #######################

    # PROPERTY SUBTYPE #
    ####################

    # combing "Single Family Detached" and "Single Family Residence Detached" into one value: "Single Family Detached"
    df['propertysubtype'] = np.where(df.propertysubtype == 'Single Family Residence Detached', 
                                    'Single Family Detached', df.propertysubtype)

    # LISTING DATE FEATURES #
    #########################

    # quarter of listing
    df['listing_quarter'] = df.listingcontractdate.dt.quarter

    # month of listing
    df['listing_month'] = df.listingcontractdate.dt.month

    # date of listing
    df['listing_dayofmonth'] = df.listingcontractdate.dt.day

    # day of week of listing
    df['listing_dayofweek'] = df.listingcontractdate.apply(lambda x: x.strftime('%a'))
    df['listing_dayofweek'] = pd.Categorical(df.listing_dayofweek,
                                            categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                            ordered=True)

    # listed on the weekend
    df['listed_on_weekend'] = np.where(df.listing_dayofweek.isin(['5-Sat', '6-Sun']), True, False)

    # BUILD YEAR FEATURES # 
    #######################

    # years since build year
    df['years_since_build'] = df.listingcontractdate.apply(lambda x: x.year) - df.yearbuilt

    # built within two calendar years of listing (i.e. this year or last)
    df['built_last_two_years'] = df.years_since_build <= 1


    # PARKING FEATURES #
    ####################

    # numerical column for size of garage

    def get_garage_size(parkingfeatures):
        if 'None' in parkingfeatures:
            return 0
        if 'One Car' in parkingfeatures:
            return 1
        elif 'Two Car' in parkingfeatures:
            return 2
        elif 'Three Car' in parkingfeatures:
            return 3
        elif 'Four or More' in parkingfeatures:
            return 4
        # 2-car garages are by far the most common in the dataset, so we will impute 2 when 
        # garage size is unknown
        else:
            return 2

    df['garage_size'] = df.parkingfeatures.apply(get_garage_size)

    # boolean columns for each listed parking feature
    df['parkingfeatures_attached'] = df.parkingfeatures.str.contains('Attached')
    df['parkingfeatures_detached'] = df.parkingfeatures.str.contains('Detached')
    df['parkingfeatures_oversized'] = df.parkingfeatures.str.contains('Oversized')
    df['parkingfeatures_converted'] = df.parkingfeatures.str.contains('Converted')
    df['parkingfeatures_sideentry'] = df.parkingfeatures.str.contains('Side Entry')
    df['parkingfeatures_rearentry'] = df.parkingfeatures.str.contains('Rear Entry')
    df['parkingfeatures_tandem'] = df.parkingfeatures.str.contains('Tandem')
    df['parkingfeatures_golfcart'] = df.parkingfeatures.str.contains('Golf Cart')

    # drop original parkingfeatures column
    df = df.drop(columns=['parkingfeatures'])


    # HEATING FEATURES # 
    ####################

    # create boolean columns for each listed heating feature

    # a one-time-use function
    def get_unique_heating_features(df):
        '''
        takes in the original "heating" feature that lists multiple attributes in one column,
        separates the individual attributes and returns a list of unique attributes
        '''
        heating_features = ''
        for feature in df.heating.unique():
            heating_features += (feature + ',')
        heating_features = heating_features.split(sep=',')
        heating_features = pd.Series(heating_features).unique()
        heating_features = heating_features[heating_features != '']
        return heating_features

    # for each unique attribute, create a boolean column 
    # based on whether the "heating" column contains that attribute
    for feature in get_unique_heating_features(df):
        df[f'heating_{feature.lower().replace(" ", "")}'] = df.heating.str.contains(feature)

    # drop original "heating" column
    df = df.drop(columns=['heating'])


    # COOLING FEATURES #
    ####################

    # create boolean columns for cooling features
    df['cooling_central'] = df.cooling.str.contains('Central')
    df['cooling_windowwall'] = df.cooling.str.contains('Window/Wall')
    df['cooling_heatpump'] = df.cooling.str.contains('Heat Pump')
    df['cooling_zoned'] = df.cooling.str.contains('Zoned')

    # create numerical columns for numbers of Central and Window/Wall cooling units

    # a one-time-use function
    def get_central_cooling_units(cooling_features):
        '''
        takes in the original "cooling" feature which lists several attributes in one column
        returns a number representing how many central cooling units are listed
        '''
        if 'Three+ Central' in cooling_features:
            return 3
        elif 'Two Central' in cooling_features:
            return 2
        elif 'One Central' in cooling_features:
            return 1
        elif 'Not Applicable' in cooling_features:
            return 0
        # 'One Central' is by far the most common value, 
        # so we will impute a 1 where data is unavailable
        else:
            return 1

    # apply the function above to the "cooling" column to create the new feature
    df['central_cooling_units'] = df.cooling.apply(get_central_cooling_units)

    # a one-time-use function
    def get_windowwall_units(cooling_features):
        '''
        takes in the original "cooling" feature which lists several attributes in one column
        returns a number representing how many window/wall cooling units are listed
        '''
        if '3+ Window/Wall' in cooling_features:
            return 3
        elif 'Two Window/Wall' in cooling_features:
            return 2
        elif 'One Window/Wall' in cooling_features:
            return 1
        # the vast majority of properties do not have window/wall units,
        # so we will impute 0 where data is unavailable
        else:
            return 0

    # apply the function above to create the new feature
    df['windowwall_cooling_units'] = df.cooling.apply(get_windowwall_units)

    # drop the original "cooling" column
    df = df.drop(columns=['cooling'])


    # ARCHITECTURAL STYLE FEATURES #
    ################################

    # creating a boolean column for each of the listed architectural style features

    # a one-time-use function
    def get_unique_archstyle_features(df):
        '''
        takes in the original "architecturalstyle" column that lists multiple attributes in one column, 
        separates the individual attributes and returns a list of unique attributes
        '''
        archstyle_features = ''
        for feature in df.architecturalstyle.unique():
            archstyle_features += (feature + ',')
        archstyle_features = archstyle_features.split(sep=',')
        archstyle_features = pd.Series(archstyle_features).unique()
        archstyle_features = archstyle_features[archstyle_features != '']
        return archstyle_features

    # for each unique attribute, create a boolean column indicating whether the 
    # # architectural style column contains that attribute
    for feature in get_unique_archstyle_features(df):
        df[f'archstyle_{feature.lower().replace(" ", "")}'] = df.architecturalstyle.str.contains(feature)

    # drop the original architecturalstyle column
    # as well as the new columns that contain info about number of stories (redundant from "stories" column)
    cols = ['architecturalstyle', 'archstyle_onestory', 'archstyle_twostory', 'archstyle_3ormore']
    df = df.drop(columns=cols)

    # LOT SIZE FEATURES #
    #####################

    # creating a boolean column for whether a property was listed with a negative value for lot size
    # this is likely an error, but could still be useful info, since only new construction was listed with negative values
    df['lotsizearea_listed_negative'] = (df.lotsizearea < 0)

    # correcting the error by taking the absolute value of the lot size
    df['lotsizearea'] = df.lotsizearea.abs()

    # create a boolean column for whether the lot size is 0.2 or less
    df['lotsizearea_small'] = df.lotsizearea <= 0.2


    # PRICE FEATURES #
    ##################

    # price per SqFt
    df['originallistprice_persqft'] = round(df.originallistprice / df.livingarea, 0)

    # CREATING SCALED PRICE COLUMNS #
    #################################
    # this accommodates for the difference in scale for "originallistprice" between
    # for-rent and for-sale listings.

    # separating rentals vs. for-sale listings
    rent_df = df[df.propertytype == 'Residential Rental']
    sale_df = df[df.propertytype == 'Residential']
    # drop irrelevant columns
    sale_df = sale_df.drop(columns=['propertytype'])
    rent_df = rent_df.drop(columns=['propertytype'])

    # scale each category to a value between 0 and 1 based on the category's respective min and max values

    scaler = MinMaxScaler()

    sale_df['originallistprice_scaled'] = scaler.fit_transform(sale_df[['originallistprice']])
    rent_df['originallistprice_scaled'] = scaler.fit_transform(rent_df[['originallistprice']])

    sale_df['originallistprice_persqft_scaled'] = scaler.fit_transform(sale_df[['originallistprice_persqft']])
    rent_df['originallistprice_persqft_scaled'] = scaler.fit_transform(rent_df[['originallistprice_persqft']])

    # combine the sale and rental scaled price columns into one df
    df2 = pd.concat([sale_df[['originallistprice_scaled', 'originallistprice_persqft_scaled']],
                    rent_df[['originallistprice_scaled', 'originallistprice_persqft_scaled']]
                    ]).sort_index()

    # join the df with scaled columns back to the original df
    df = pd.merge(df, df2, 
                how='left', 
                left_index=True, 
                right_index=True)


    return df, sale_df, rent_df



