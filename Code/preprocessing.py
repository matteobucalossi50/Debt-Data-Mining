import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# read dataset as dataframe
data = pd.read_csv('real_estate_db.csv', encoding='latin-1')

# shape of dataset
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# first dataset observations
print("Dataset first few rows:\n ")
print(data.head(5))

# structure of the dataset
print("Dataset info:\n ")
print(data.info())

# print null values per column
nulls = data.isnull().sum()
print(nulls)

# drop null column
data = data.drop('BLOCKID', axis=1)
print("Dataset Shape: ", data.shape)

# imputation
def impute(df):
    for column in df.columns:
            if column.endswith('median'):
                imputer = SimpleImputer('median')
                imputer = imputer.fit(df[:, column])
                df[:, column] = imputer.transform(df[:, column])
            else:
                imputer = SimpleImputer('mean')
                imputer = imputer.fit(df[:, column])
                df[:, column] = imputer.transform(df[:, column])
    return df

data = impute(data)

# delete if imputation works
data.dropna(inplace=True)
data.shape[0]


# outliers

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

# remove outliers from pop column (before it's geographic data)
data = remove_outliers(data.loc['pop':])

# these are checks for me if function worked - delete once we are done
print(sum(data['pop']> 55000))

for name in data.columns:
    if is_numeric_dtype(data[name]):
        sns.boxplot(data[name])
        plt.show()



