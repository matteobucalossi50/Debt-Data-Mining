
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# read dataset as dataframe
data = pd.read_csv('real_estate_db.csv', encoding='latin-1')

#Recatogorize the data
def typeconv(colname):
    for i in range(0,len(colname),1):
        data[colname[i]] = data[colname[i]].astype('object')


colname = ('UID','SUMLEVEL','COUNTYID','STATEID','zip_code','area_code')
typeconv(colname)

# shape of dataset
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# structure of the dataset
print("Dataset info:\n ")
print(data.info())

#Drop Duplicates
data = data.drop_duplicates()

# print null values per column
nulls = data.isna().sum()
print("Null values for each column:\n")
print(nulls)

# drop null column
data = data.drop('BLOCKID', axis=1)

# drop unnecessary columns
for col in data.columns:
    if col.endswith(('median', 'stdev', 'samples', 'weight', 'cdf')) | col.startswith('rent_gt'):
        data = data.drop(col, axis=1)
print("Dataset Shape after dropping columns:\n", data.shape)

#replacing all the missing values with mean of the column
data=data.fillna(data.mean()[:])

# print null values per column- to check if null values have been elimnated
nulls = data.isna().sum()
print("Null values for each column:\n")
print(nulls)



def remove_outliers(df):
    for name in df.columns:
        if is_numeric_dtype(df[name]):
            q1 = df[name].quantile(0.25)
            q3 = df[name].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            df = df[(df[name] > lower_bound) & (df[name] < upper_bound)]
    return df

# remove outliers - merge outlier table back to
data1 = remove_outliers(data.loc[:, "ALand":])
print(data1)
mergecol = []
for col in data1.columns:
    mergecol.append(col)
data = data.merge(data1, on=mergecol)
print(data)
print(data.loc[:, "pop":].head())

for name in data.columns:
    if is_numeric_dtype(data[name]):
        sns.boxplot(data[name])
        plt.show()



