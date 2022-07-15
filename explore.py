import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def build_year(train, target):
    
    plt.figure(figsize=(20,10))
    sns.swarmplot(data=train, y='yearbuilt', x='newconstructionyn')
    plt.title('New Construction by Build Year', fontsize=24)
    plt.ylabel('Build Year', fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=20)
    plt.show()
    
    plt.figure(figsize=(20,10))
    sns.countplot(data=train[train.yearbuilt > 2010], x='yearbuilt', hue=target)
    plt.title('New Construction by Build Year', fontsize=20)
    plt.ylabel('Number of Listings', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xlabel('Build Year', fontsize=20)
    plt.xticks(fontsize=16)
    plt.legend(['Existing', 'New Construction'], fontsize=20, loc='upper left')
    plt.show()

def built_last_two_years(train, target):
    plt.figure(figsize=(20,10))
    sns.countplot(data=train, x='built_last_two_years', hue=target)
    plt.title('Build Year Within 2 Years', fontsize=24)
    plt.legend(labels=['Existing', 'New Construction'], fontsize=20)
    plt.ylabel('Number of Listings\n', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], 
               labels=['Older Than\n2 Years', 
                       'Build Year Within\n2 Years'], 
               fontsize=20)
    plt.show()

def list_price(train, target):
    plt.figure(figsize=(8, 8))
    sns.barplot(data=train, 
                x=target, 
                y='originallistprice_scaled', 
                ci=None)
    plt.title('List Price\n(All Listings)', fontsize=18)
    plt.ylabel('Average\nScaled Price\n', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=16)
    plt.show()

    plt.figure(figsize=(16, 8))
    sns.barplot(data=train[train.yearbuilt>1990], 
                x='built_last_two_years', 
                y='originallistprice_scaled', 
                hue=target,
                ci=None)
    plt.title('List Price\n(By Whether the\nBuild Year is Recent)', fontsize=22)
    plt.ylabel('Average\nScaled Price\n', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], 
            labels=['Older Than\n2 Years',
                    'Build Year Within\n2 Years'],
            fontsize=16)
    plt.legend(fontsize=16, labels=['Existing', 'New Construction'],loc='upper left')
    plt.show()

def list_price_persqft(train, target):
    plt.figure(figsize=(8, 8))
    sns.barplot(data=train, 
                x=target, 
                y='originallistprice_persqft_scaled', 
                ci=None)
    plt.title('List Price-per-SqFt\n(All Listings)', fontsize=18)
    plt.ylabel('Average\nScaled Price-per-SqFt\n', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=16)
    plt.show()

    plt.figure(figsize=(16, 8))
    sns.barplot(data=train[train.yearbuilt>1990], 
                x='built_last_two_years', 
                y='originallistprice_persqft_scaled', 
                hue=target,
                ci=None)
    plt.title('List Price-per-Sqft\n(By Whether the\nBuild Year is Recent)', fontsize=22)
    plt.ylabel('Average\nScaled Price-Per_SqFt\n', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], 
            labels=['Older Than\n2 Years',
                    'Build Year Within\n2 Years'],
            fontsize=16)
    plt.legend(fontsize=16, labels=['Existing', 'New Construction'],loc='upper left')
    plt.show()

def previous_listings(train, target):
    previously_listed_rates = (pd.DataFrame(train.groupby(by=target).mean().previously_listed)
                             .reset_index()
                             .sort_values(by='previously_listed'))
    plt.figure(figsize=(8,8))
    sns.barplot(data=previously_listed_rates, 
                x=target,
                y='previously_listed')
    plt.title('Proportion of Properties\nwith Previous Listings', fontsize=18)
    plt.ylabel('Proportion\n', fontsize=16)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=16)
    plt.show()

def difference_in_price(sale_df):

    plt.figure(figsize=(12,8))
    sns.histplot(sale_df.originallistprice, color='lightblue')
    plt.title('Distribution\nof Price', fontsize=16)
    plt.show()

    sns.catplot(data=sale_df, 
            y='originallistprice', 
            x='newconstructionyn', 
            kind='box', 
            height=8, aspect=.75)
    plt.title('Distribution\nof Price', fontsize=16)
    plt.ylabel('List Price\n', fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=14)
    plt.show()

def show_median_prices(sale_df, target):
    median_price_new = sale_df[sale_df[target]].originallistprice.median()
    median_price_existing = sale_df[~sale_df[target]].originallistprice.median()
    print(f'Median Price for New Construction: ${median_price_new: ,.0f}')
    print(f'Median Price for Existing: ${median_price_existing: ,.0f}')
    print(f'Median difference in Price: ${median_price_new - median_price_existing: ,.0f}')

def difference_in_price_persqft(sale_df):

    plt.figure(figsize=(12,8))
    sns.histplot(sale_df.originallistprice_persqft, color='lightblue')
    plt.title('Distribution\nof Price-per-SqFt', fontsize=16)
    plt.show()

    sns.catplot(data=sale_df, 
            y='originallistprice_persqft', 
            x='newconstructionyn', 
            kind='box', 
            height=8, aspect=.75)
    plt.title('Distribution\nof Price-per-SqFt', fontsize=16)
    plt.ylabel('List Price\n', fontsize=14)
    plt.xlabel(None)
    plt.xticks(ticks=[False, True], labels=['Existing', 'New Construction'], fontsize=14)
    plt.show()

def show_median_price_persqft(sale_df, target):
    median_price_new = sale_df[sale_df[target]].originallistprice_persqft.median()
    median_price_existing = sale_df[~sale_df[target]].originallistprice_persqft.median()
    print(f'Median Price-per-SqFt for New Construction: ${median_price_new: ,.0f}')
    print(f'Median Price-per-SqFt for Existing: ${median_price_existing: ,.0f}')
    print(f'Median difference in Price-per-SqFt: ${median_price_new - median_price_existing: ,.0f}')