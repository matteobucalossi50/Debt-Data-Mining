#!/usr/bin/env python
# coding: utf-8

# In[18]:


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
print("Dataset Shape after dropping null columns:\n", data.shape)

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
data1 = remove_outliers(data.loc[:,"ALand":])
print(data1)
mergecol = []
for col in data1.columns:
    mergecol.append(col)
data = data.merge(data1, on=mergecol)
print(data)
print(data.loc[:,"pop":].head())

for name in data.columns:
    if is_numeric_dtype(data[name]):
        sns.boxplot(data[name])
        plt.show()




#=================================================================================================

print(data.iloc[:,10])
#features
X = data.iloc[:,10:]

normalized_X = preprocessing.normalize(X, axis=0)
scale_X = preprocessing.scale(X, axis=0)
#print(normalized_X[:5])
#print(scale_X[:5])
#print(sum(normalized_X[0]))
#print(sum(scale_X[0]))
print("normalize shape", normalized_X.shape)
print("scale shape", scale_X.shape)
#==============================================================================================


#feature selection- using heatmap

df1=pd.DataFrame(scale_X)
plt.figure(figsize=(100,50))
cor1 = df1.corr()
sns.heatmap(cor1, annot=True, cmap=plt.cm.Reds)
plt.savefig('corplot-heatmap.png')


#feature selection- feature importance from tree classifier
# here the values have been converted to integer since it does not accept float values
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
X = df.loc[:,2:]  #independent columns
X= X.astype(int) 
y = df.loc[:,0:2]
y= y.astype(int)
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#feature selection- using pca, three components are used
from sklearn.decomposition import PCA
X = df.loc[:,2:]  #independent columns 
y = df.loc[:,0:2]
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


# In[20]:


Monthly Mortgage and owner costs and monthly owner costs have set as 
the target variables. The remaining 68 values are used as features.
Features with the highest importance are: 
-gross rent 
-household income
-hc_mortgage_sample
-hc_mean
-hc_mortgage_mean
-hs_degree_female
-hc_mortgage_mean
-county
-hi_samples

