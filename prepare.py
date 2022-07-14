import pandas as pd
import numpy as np

def wrangle_data():

    df = pd.read_csv('data.csv')

    # define target column
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
    cols_to_drop = ['lotfeatures'] 
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

    #############################################
    # SEPARATING FOR-SALE AND FOR-RENT LISTINGS #
    #############################################

    # separating rentals vs. for-sale listings
    rent_df = df[df.propertytype == 'Residential Rental']
    sale_df = df[df.propertytype == 'Residential']

    # drop irrelevant columns
    sale_df = sale_df.drop(columns=['totalactualrent', 'propertytype'])
    rent_df = rent_df.drop(columns=['propertytype'])

    ########################
    # ADJUSTING DATA TYPES #
    ########################

    # address_id from int to string
    sale_df['address_id'] = sale_df.address_id.astype(str)
    rent_df['address_id'] = rent_df.address_id.astype(str)

    # garageyn and newconstructionyn to from float to boolean
    sale_df['newconstructionyn'] = sale_df.newconstructionyn.astype(bool)
    rent_df['newconstructionyn'] = rent_df.newconstructionyn.astype(bool)

    sale_df['garageyn'] = sale_df.garageyn.astype(bool)
    rent_df['garageyn'] = rent_df.garageyn.astype(bool)

    # listingcontractdate from string to pandas datetime
    sale_df['listingcontractdate'] = pd.to_datetime(sale_df.listingcontractdate)
    rent_df['listingcontractdate'] = pd.to_datetime(rent_df.listingcontractdate)

    # yearbuilt from float to int
    sale_df['yearbuilt'] = sale_df.yearbuilt.astype(int)
    rent_df['yearbuilt'] = rent_df.yearbuilt.astype(int)


    #######################
    # FEATURE ENGINEERING #
    #######################

    # PROPERTY SUBTYPE #
    ####################

    # combing "Single Family Detached" and "Single Family Residence Detached" into one value: "Single Family Detached"
    sale_df['propertysubtype'] = np.where(sale_df.propertysubtype == 'Single Family Residence Detached', 
                                        'Single Family Detached', sale_df.propertysubtype)
    rent_df['propertysubtype'] = np.where(rent_df.propertysubtype == 'Single Family Residence Detached', 
                                      'Single Family Detached', rent_df.propertysubtype)

    # LISTING DATE FEATURES #
    #########################

    # quarter of listing
    sale_df['listing_quarter'] = sale_df.listingcontractdate.dt.quarter
    rent_df['listing_quarter'] = rent_df.listingcontractdate.dt.quarter

    # month of listing
    sale_df['listing_month'] = sale_df.listingcontractdate.dt.month
    rent_df['listing_month'] = rent_df.listingcontractdate.dt.month

    # date of listing
    sale_df['listing_dayofmonth'] = sale_df.listingcontractdate.dt.day
    rent_df['listing_dayofmonth'] = rent_df.listingcontractdate.dt.day

    # weekday of listing
    sale_df['listing_dayofweek'] = sale_df.listingcontractdate.apply(lambda x: x.strftime('%a'))
    rent_df['listing_dayofweek'] = rent_df.listingcontractdate.apply(lambda x: x.strftime('%a'))

    sale_df['listing_dayofweek'] = pd.Categorical(sale_df.listing_dayofweek,
                                                categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                                ordered=True)
    rent_df['listing_dayofweek'] = pd.Categorical(rent_df.listing_dayofweek,
                                                categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                                ordered=True)
    # listed on the weekend
    sale_df['listed_on_weekend'] = np.where(sale_df.listing_dayofweek.isin(['5-Sat', '6-Sun']), True, False)
    rent_df['listed_on_weekend'] = np.where(rent_df.listing_dayofweek.isin(['5-Sat', '6-Sun']), True, False)

    # BUILD YEAR FEATURES # 
    #######################

    # years since build year
    sale_df['years_since_build'] = sale_df.listingcontractdate.apply(lambda x: x.year) - sale_df.yearbuilt
    rent_df['years_since_build'] = rent_df.listingcontractdate.apply(lambda x: x.year) - rent_df.yearbuilt

    # built within two calendar years of listing (i.e. this year or last)
    sale_df['built_last_two_years'] = sale_df.years_since_build <= 1
    rent_df['built_last_two_years'] = rent_df.years_since_build <= 1

    # PRICE FEATURES #
    ##################

    # price-per-square-foot
    sale_df['originallistprice_persqft'] = round(sale_df.originallistprice / sale_df.livingarea, 0)
    rent_df['originallistprice_persqft'] = round(rent_df.originallistprice / rent_df.livingarea, 2)
    rent_df['totalactualrent_persqft'] = round(rent_df.totalactualrent / rent_df.livingarea, 2)


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

    sale_df['garage_size'] = sale_df.parkingfeatures.apply(get_garage_size)
    rent_df['garage_size'] = rent_df.parkingfeatures.apply(get_garage_size)

    # boolean columns for each listed parking feature

    sale_df['parkingfeatures_attached'] = sale_df.parkingfeatures.str.contains('Attached')
    rent_df['parkingfeatures_attached'] = rent_df.parkingfeatures.str.contains('Attached')

    sale_df['parkingfeatures_detached'] = sale_df.parkingfeatures.str.contains('Detached')
    rent_df['parkingfeatures_detached'] = rent_df.parkingfeatures.str.contains('Detached')

    sale_df['parkingfeatures_oversized'] = sale_df.parkingfeatures.str.contains('Oversized')
    rent_df['parkingfeatures_oversized'] = rent_df.parkingfeatures.str.contains('Oversized')

    sale_df['parkingfeatures_converted'] = sale_df.parkingfeatures.str.contains('Converted')
    rent_df['parkingfeatures_converted'] = rent_df.parkingfeatures.str.contains('Converted')

    sale_df['parkingfeatures_sideentry'] = sale_df.parkingfeatures.str.contains('Side Entry')
    rent_df['parkingfeatures_sideentry'] = rent_df.parkingfeatures.str.contains('Side Entry')

    sale_df['parkingfeatures_rearentry'] = sale_df.parkingfeatures.str.contains('Rear Entry')
    rent_df['parkingfeatures_rearentry'] = rent_df.parkingfeatures.str.contains('Rear Entry')

    sale_df['parkingfeatures_tandem'] = sale_df.parkingfeatures.str.contains('Tandem')
    rent_df['parkingfeatures_tandem'] = rent_df.parkingfeatures.str.contains('Tandem')

    sale_df['parkingfeatures_golfcart'] = sale_df.parkingfeatures.str.contains('Golf Cart')
    rent_df['parkingfeatures_golfcart'] = rent_df.parkingfeatures.str.contains('Golf Cart')

    # drop original parkingfeatures column
    sale_df = sale_df.drop(columns=['parkingfeatures'])
    rent_df = rent_df.drop(columns=['parkingfeatures'])


    # HEATING FEATURES # 
    ####################

    # create boolean columns for each listed heating feature

    def get_unique_heating_features(df):
        heating_features = ''
        for feature in df.heating.unique():
            heating_features += (feature + ',')
        heating_features = heating_features.split(sep=',')
        heating_features = pd.Series(heating_features).unique()
        heating_features = heating_features[heating_features != '']
        return heating_features

    for feature in get_unique_heating_features(sale_df):
        sale_df[f'heating_{feature.lower().replace(" ", "")}'] = sale_df.heating.str.contains(feature)
        
    for feature in get_unique_heating_features(rent_df):
        rent_df[f'heating_{feature.lower().replace(" ", "")}'] = rent_df.heating.str.contains(feature)

    # drop original "heating" column
    sale_df = sale_df.drop(columns=['heating'])
    rent_df = rent_df.drop(columns=['heating'])

    
    # COOLING FEATURES #
    ####################

    # create boolean columns for cooling features

    sale_df['cooling_central'] = sale_df.cooling.str.contains('Central')
    rent_df['cooling_central'] = rent_df.cooling.str.contains('Central')

    sale_df['cooling_windowwall'] = sale_df.cooling.str.contains('Window/Wall')
    rent_df['cooling_windowwall'] = rent_df.cooling.str.contains('Window/Wall')

    sale_df['cooling_heatpump'] = sale_df.cooling.str.contains('Heat Pump')
    rent_df['cooling_heatpump'] = rent_df.cooling.str.contains('Heat Pump')

    sale_df['cooling_zoned'] = sale_df.cooling.str.contains('Zoned')
    rent_df['cooling_zoned'] = rent_df.cooling.str.contains('Zoned')

    # create numerical columns for numbers of Central and Window/Wall cooling units

    def get_central_cooling_units(cooling_features):
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
        
    sale_df['central_cooling_units'] = sale_df.cooling.apply(get_central_cooling_units)
    rent_df['central_cooling_units'] = rent_df.cooling.apply(get_central_cooling_units)

    def get_windowwall_units(cooling_features):
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

    sale_df['windowwall_cooling_units'] = sale_df.cooling.apply(get_windowwall_units)
    rent_df['windowwall_cooling_units'] = rent_df.cooling.apply(get_windowwall_units)

    # drop the original "cooling" column
    sale_df = sale_df.drop(columns=['cooling'])
    rent_df = rent_df.drop(columns=['cooling'])


    # ARCHITECTURAL STYLE FEATURES #
    ################################

    # creating a boolean column for each of the listed architectural style features

    def get_unique_archstyle_features(df):
        archstyle_features = ''
        for feature in df.architecturalstyle.unique():
            archstyle_features += (feature + ',')
        archstyle_features = archstyle_features.split(sep=',')
        archstyle_features = pd.Series(archstyle_features).unique()
        archstyle_features = archstyle_features[archstyle_features != '']
        return archstyle_features

    for feature in get_unique_archstyle_features(df):
        sale_df[f'archstyle_{feature.lower().replace(" ", "")}'] = sale_df.architecturalstyle.str.contains(feature)
        
    for feature in get_unique_archstyle_features(df):
        rent_df[f'archstyle_{feature.lower().replace(" ", "")}'] = rent_df.architecturalstyle.str.contains(feature)

    # drop the original architecturalstyle column
    # and new columns that contain info about number of stories (redundant from "stories" column)
    cols = ['architecturalstyle', 'archstyle_onestory', 'archstyle_twostory', 'archstyle_3ormore']
    sale_df = sale_df.drop(columns=cols)
    rent_df = rent_df.drop(columns=cols)


    # LOT SIZE FEATURES #
    #####################

    # creating a boolean column for whether a property was listed with a negative value for lot size
    # This is likely an error in the listing, but could still be useful info, 
    # since only new construction was listed with negative values.
    sale_df['lotsizearea_listed_negative'] = (sale_df.lotsizearea < 0)
    rent_df['lotsizearea_listed_negative'] = (rent_df.lotsizearea < 0)

    # correcting the error by taking the absolute value of the lot size
    sale_df['lotsizearea'] = sale_df.lotsizearea.abs()
    rent_df['lotsizearea'] = rent_df.lotsizearea.abs()

    # lot size 0.2 or less
    sale_df['lotsizearea_small'] = sale_df.lotsizearea <= 0.2
    rent_df['lotsizearea_small'] = rent_df.lotsizearea <= 0.2

    return sale_df, rent_df



