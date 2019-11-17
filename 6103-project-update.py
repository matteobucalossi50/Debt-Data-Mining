#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
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


# In[2]:


#loading data 
debtdata = pd.read_csv('real_estate_db.csv', encoding='latin-1')


# In[57]:


#information about the dataset
print(debtdata.info())
nulls = debtdata.isna().sum()
print("Null values for each column before any modifications:\n")
print(nulls)


# In[4]:


#creating new data frame- dropping all columns that are not needed
modd=debtdata.copy(deep=True)
modd = modd.drop(['SUMLEVEL','rent_median','rent_stdev','used_samples','hi_median','hi_stdev','hc_mortgage_median','hc_mortgage_stdev','hc_mortgage_samples'], axis=1)

modd  = modd.drop(['rent_gt_10','rent_gt_15','rent_gt_20','rent_gt_25','rent_gt_30','rent_gt_35','rent_gt_40','rent_gt_50'], axis=1)

modd  = modd.drop(['hc_median','hc_stdev','hc_samples','family_median','family_stdev','family_samples','rent_samples'], axis=1)

modd  = modd.drop(['male_age_median','male_age_stdev','male_age_samples','female_age_median','female_age_stdev','female_age_samples'], axis=1)

modd= modd.drop(['rent_sample_weight','family_sample_weight','universe_samples','hi_samples','hi_sample_weight','married_snp','pct_own','female_age_sample_weight','male_age_sample_weight','hc_sample_weight','hc_mortgage_sample_weight'],axis=1)


modd=modd.drop('BLOCKID',axis=1)
modd.info()


# In[5]:


#changing the datatypes of certain columns to object
modd['UID'] = modd['UID'].astype('object')
modd['COUNTYID'] = modd['COUNTYID'].astype('object')
modd['STATEID'] = modd['STATEID'].astype('object')
modd['zip_code'] = modd['zip_code'].astype('object')
modd['area_code'] = modd['area_code'].astype('object')
#so that the outlier function can ignore these variables
modd['lat'] = modd['lat'].astype('object')
modd['lng'] = modd['lng'].astype('object')
modd['ALand'] = modd['ALand'].astype('object')
modd['AWater'] = modd['AWater'].astype('object')
print(modd.info())


# ## Checking for null values:

# In[58]:


nulls = modd.isna().sum()
print("Null values for each column after modifications:\n")
print(nulls)


# ## Elimnating Outliers

# In[59]:


#function to remove outliers
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

# preparing data for elimnating outliers
#modd_w_in=modd.drop(["COUNTYID","STATEID","state","state_ab","city","place","type","primary","zip_code","area_code","lat","lng","ALand","AWater"],axis=1)
#m2=modd.loc[:,"pop":]
#mod_w_in=m1.append(m2, ignore_index=True)
#elimnating outliers
modd_w_out = remove_outliers(modd)


# In[60]:


#checking for duplicates
dup = modd_w_out[modd_w_out.duplicated(["UID","pop"])]
#there are no duplicates we move onto merging the data
#merging the data
#modd_merged = pd.merge(left=modd_w_out,right=modd_w_in, how='inner', left_on='UID', right_on='UID')
#final_modd_merged = pd.merge(left=modd,right=modd_merged, how='inner', left_on='UID', right_on='UID')
#final_modd_merged.info()
#print(final_modd_merged.head(2))
#geoginfo=modd_w_out.loc[:,"UID":"AWater"]
#maleinfo=modd_w_out.loc[:,["UID","male_pop","hs_degree_male"]]


# In[8]:


#inspecting final dataset
modd_w_out.info()


# In[61]:


#converting back the orignal data types so that we can perform analysis
modd_w_out['lat'] = modd_w_out['lat'].astype('float64')
modd_w_out['lng'] = modd_w_out['lng'].astype('float64')
modd_w_out['ALand'] = modd_w_out['ALand'].astype('int64') 
modd_w_out['AWater'] = modd_w_out['AWater'].astype('int64')
print(modd_w_out.info())


# ## Visualization

# In[10]:


plt.hist(modd_w_out["pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Population Distribution')
plt.show()


# In[11]:


plt.hist(modd_w_out["male_pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Male Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Male Population Distribution')
plt.show()


# In[12]:


plt.hist(modd_w_out["female_pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Female Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Female Population Distribution')
plt.show()


# In[13]:


plt.hist(modd_w_out["rent_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Population')
plt.title(r'Rent Distribution')
plt.show()


# In[14]:


plt.hist(modd_w_out["hi_mean"],bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Household income')
plt.title(r'Household income distribution')
plt.show()


# In[15]:


plt.hist(modd_w_out["family_mean"],bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Household income')
plt.title(r'Mean Family income')
plt.show()


# In[16]:


plt.hist(modd_w_out["hc_mortgage_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Mean Monthly Mortgage and Owner Cost')
plt.ylabel('Frequency')
plt.title(r'Histogram of Mean Monthly Mortgage and Owner Costs of specified geographic location')
plt.show()


# In[17]:


plt.hist(modd_w_out["hc_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Mean Monthly Owner Costs')
plt.ylabel('Frequency')
plt.title(r'Histogram of  mean Monthly Owner Costs  of specified geographic location')
plt.show()


# In[18]:


plt.hist(modd_w_out["home_equity_second_mortgage"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('  Percentage of homes with a second mortgage and home equity loan')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of homes with a second mortgage and home equity loan')
plt.show()


# In[19]:


plt.hist(modd_w_out["second_mortgage"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('percent of houses with a second mortgage')
plt.ylabel('Frequency')
plt.title(r'Histogram of percent of houses with a second mortgage')
plt.show()


# In[20]:


plt.hist(modd_w_out["home_equity"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of homes with a home equity loan.')
plt.ylabel('Frequency')
plt.title(r'Histogram of  Percentage of homes with a home equity loan.')
plt.show()


# In[21]:


plt.hist(modd_w_out["debt"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Percentage of homes with some type of debt.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of homes with some type of debt.')
plt.show()


# In[22]:


plt.hist(modd_w_out["hs_degree"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of people with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of people with at least high school degree.')
plt.show()


# In[23]:


plt.hist(modd_w_out["hs_degree_male"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of males with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of males with at least high school degree.')
plt.show()


# In[24]:


plt.hist(modd_w_out["hs_degree_female"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of females with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of females with at least high school degree.')
plt.show()


# In[25]:


plt.hist(modd_w_out["married"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of married people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of married people in the geographical area.')
plt.show()


# In[26]:


plt.hist(modd_w_out["separated"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of separated people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of separated people in the geographical area.')
plt.show()


# In[27]:


plt.hist(modd_w_out["divorced"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of divorced people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of divorced people in the geographical area.')
plt.show()


# ## Performing KNN 

# In[105]:


#splitting data into test and training data
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings("ignore")


#splitting data into training and testing data
X=(modd_w_out.loc[:,('rent_mean','home_equity','pop')])#variables to fit the data
y=(modd_w_out.loc[:,"debt"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


#scalling the data 
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#label encoding data to perform regrssion
le = preprocessing.LabelEncoder()
y_test = le.fit_transform(y_test)



classifier = KNeighborsRegressor(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = le.fit_transform(y_pred)

#printing the accuracy score
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")


print(classifier.predict([[1,2,3]]))# will change later!!!!


# ## SVR:

# In[107]:


#svr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
S = modd_w_out.loc[:,('rent_mean','home_equity','pop')].values
T = modd_w_out['debt']
t2 = le.fit_transform(T)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_S = StandardScaler()
sc_t = StandardScaler()
S2 = sc_S.fit_transform(S)


#fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(S2, t2)
y_pred = classifier.predict(X_test)
y_pred = le.fit_transform(y_pred)
y_test=le.fit_transform(y_test)


#printing the accuracy score
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print(regressor.predict([[1,2,3]]))

