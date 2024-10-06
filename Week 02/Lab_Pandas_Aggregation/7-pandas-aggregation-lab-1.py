# -*- coding: utf-8 -*-
"""
Pandas dataframes

@author: Glenn Bruns
"""
import numpy as np
import pandas as pd


# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# =============================================================================
# Read data
# =============================================================================

# read 1994 census summary data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv")
df.set_index('usid', inplace=True)
df.drop('fnlwgt', axis=1, inplace=True)

# =============================================================================
# Simple aggregation
# =============================================================================

# print the average age
print(df['age'].mean())
# get the min, max, and avg value for each numeric column
df.select_dtypes(np.number).aggregate(['min','max','mean'])
# for a dataframe you get the aggregate for each column by default

# =============================================================================
# Aggregation with grouping
# =============================================================================

# how many people in each category of education?
# Try using pandas function value_counts().
df['education'].value_counts()
# for each native country, what is the average education num?
df.groupby('native_country')['education_num'].mean()
# repeat the previous problem, sorting the output by the average
# education num value
df.groupby('native_country')['education_num'].mean().sort_values()
# for each occupation, compute the median age
df.groupby('occupation')['age'].median()
# repeat the previous problem, but sort the output
df.groupby('occupation')['age'].median().sort_values()
# find average hours_per_week for those with age <= 40, and those with age > 40
# (this uses something labeled 'advanced' in the lecture)
df.assign(age_group=np.where(df['age'] <= 40, '<= 40', '> 40')).groupby('age_group')['hours_per_week'].mean()
# do the same, but for age groups < 40, 40-60, and > 60
df.groupby(pd.cut(df['age'], bins=[0, 40, 60, np.inf], labels=['< 40', '40-60', '> 60'], right=False, include_lowest=True), observed=True)['hours_per_week'].mean()
# get the rows of the data frame, but only for occupations
# with an average number of education_num > 10
# Hint: use filter
df.groupby('occupation').filter(lambda x: x['education_num'].mean() > 10)
# =============================================================================
# Vectorized string operations
# =============================================================================

# create a Pandas series containing the values in the native_country column.
# Name this series 'country'.
country = df['native_country']
# how many different values appear in the country series?
32561
# create a Series containing the unique country names in the series.
# Name this new series 'country_names'.
country_names = pd.Series(df['native_country'].unique())
# modify country_names so that underscore '_' is replaced
# by a hyphen '-' in every country name.  Use a vectorized operation.
# (See https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)
country_names = country_names.str.replace('_', '-')
# modify country_names to replace 'Holand' with 'Holland'.
country_names = country_names.str.replace('Holand', 'Holland', regex=False)
# modify country_names so that any characters after 5 are removed.
# Again, use a vectorized operation
country_names = country_names.str[:5]
# Suppose we were to use only the first two letters of each country.
# Write some code to determine if any two countries would then share
# a name.
abbr_names = country_names.str[:2]
if not abbr_names[abbr_names.duplicated()].empty:
    print("Duplication detected.")

# If you still have time, write code to determine which countries
# do not have a unique name when only the first two characters are
# used.  Hint: look into Pandas' Series.duplicated().

# =============================================================================
# Handling times and dates
# =============================================================================

# read gas prices data
gas = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/Gasoline_Retail_Prices_Weekly_Average_by_Region__Beginning_2007.csv")

# create a datetime series and make it the index of the dataset
gas['Date'] = pd.to_datetime(gas['Date'])
gas.set_index('Date', inplace=True)
# plot the gas prices for NY city
import matplotlib.pyplot as plt
nyc_gas = gas[['New York City Average ($/gal)']]
plt.figure(figsize=(10, 5)); plt.plot(nyc_gas.index, nyc_gas['New York City Average ($/gal)'], color='purple', linestyle='-'); plt.title('Gas Prices in New York City'); plt.xlabel('Date'); plt.ylabel('Price (USD/gallon)'); plt.grid(True)

# plot gas prices for NY city and Albany on the same plot
nyc_gas = gas[['New York City Average ($/gal)']]
albany_gas = gas[['Albany Average ($/gal)']]
plt.plot(nyc_gas.index, nyc_gas['New York City Average ($/gal)'], label='New York City', color='red', linestyle='-'); plt.plot(albany_gas.index, albany_gas['Albany Average ($/gal)'], label='Albany', color='green', linestyle='-'); plt.title('Gas Prices in New York City and Albany'); plt.xlabel('Date'); plt.ylabel('Price (USD/gallon)'); plt.legend(); plt.grid(True)
# if you still have time, see if you can find and plot California
# gas prices



