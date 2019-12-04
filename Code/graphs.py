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




# In[61]:


#converting back the orignal data types so that we can perform analysis
modd_w_out['lat'] = modd_w_out['lat'].astype('float64')
modd_w_out['lng'] = modd_w_out['lng'].astype('float64')
modd_w_out['ALand'] = modd_w_out['ALand'].astype('int64')
modd_w_out['AWater'] = modd_w_out['AWater'].astype('int64')
new_nulls=modd_w_out.isna().sum()
#data=data.fillna(data.mean()[:])
print("FINAL DATA SET:",new_nulls)


# ## Visualization

# In[10]:


plt.hist(modd_w_out["pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Population Distribution')
plt.savefig("popdist")


# In[11]:


plt.hist(modd_w_out["male_pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Male Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Male Population Distribution')
plt.savefig("malepop")


# In[12]:


plt.hist(modd_w_out["female_pop"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Female Population')
plt.ylabel('Frequency')
plt.title(r'Histogram of Female Population Distribution')
plt.savefig("femaledist")


# In[13]:


plt.hist(modd_w_out["rent_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Population')
plt.title(r'Rent Distribution')
plt.savefig("rentdist")



# In[14]:


plt.hist(modd_w_out["hi_mean"],bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Household income')
plt.title(r'Household income distribution')
plt.savefig("ousehold")


# In[15]:


plt.hist(modd_w_out["family_mean"],bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Rent')
plt.ylabel('Family income')
plt.title(r'Mean Family income')
plt.savefig("familyinc")


# In[16]:


plt.hist(modd_w_out["hc_mortgage_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Mean Monthly Mortgage and Owner Cost')
plt.ylabel('Frequency')
plt.title(r'Histogram of Mean Monthly Mortgage and Owner Costs of specified geographic location')
plt.savefig("monthlymorg")


# In[17]:


plt.hist(modd_w_out["hc_mean"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Mean Monthly Owner Costs')
plt.ylabel('Frequency')
plt.title(r'Histogram of  mean Monthly Owner Costs  of specified geographic location')
plt.savefig("ownercost")


# In[18]:


plt.hist(modd_w_out["home_equity_second_mortgage"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('  Percentage of homes with a second mortgage and home equity loan')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of homes with a second mortgage and home equity loan')
plt.savefig("secondmorgage")


# In[19]:


plt.hist(modd_w_out["second_mortgage"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('percent of houses with a second mortgage')
plt.ylabel('Frequency')
plt.title(r'Histogram of percent of houses with a second mortgage')
plt.savefig("homes")


# In[20]:





# In[21]:


plt.hist(modd_w_out["debt"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel('Percentage of homes with some type of debt.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of homes with some type of debt.')
plt.savefig("debt")


# In[22]:


plt.hist(modd_w_out["hs_degree"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of people with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of people with at least high school degree.')
plt.savefig("highschool")


# In[23]:


plt.hist(modd_w_out["hs_degree_male"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of males with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of males with at least high school degree.')
plt.savefig("malehs")


# In[24]:


plt.hist(modd_w_out["hs_degree_female"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of females with at least high school degree.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of females with at least high school degree.')
plt.savefig("femalehs")


# In[25]:


plt.hist(modd_w_out["married"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of married people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of married people in the geographical area.')
plt.savefig("married")


# In[26]:


plt.hist(modd_w_out["separated"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of separated people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of separated people in the geographical area.')
plt.savefig("seperated")


# In[27]:


plt.hist(modd_w_out["divorced"], bins = 15, color = 'blue', edgecolor = 'black')
plt.xlabel(' Percentage of divorced people in the geographical area.')
plt.ylabel('Frequency')
plt.title(r'Histogram of Percentage of divorced people in the geographical area.')
plt.savefig("divorced")