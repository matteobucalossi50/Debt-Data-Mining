# Matteo Individual Code

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import tree
import collections
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
import webbrowser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#loading data
debtdata = pd.read_csv('real_estate_db.csv', encoding='latin-1')

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


#information about the dataset
print(debtdata.info())
nulls = debtdata.isna().sum()
print("Null values for each column before any modifications:\n")
print(nulls)



#dropping all columns that are not needed
modd=debtdata.copy(deep=True)
modd = modd.drop(['SUMLEVEL','rent_median','rent_stdev','used_samples','hi_median','hi_stdev','hc_mortgage_median','hc_mortgage_stdev','hc_mortgage_samples'], axis=1)

modd  = modd.drop(['rent_gt_10','rent_gt_15','rent_gt_20','rent_gt_25','rent_gt_30','rent_gt_35','rent_gt_40','rent_gt_50'], axis=1)

modd  = modd.drop(['hc_median','hc_stdev','hc_samples','family_median','family_stdev','family_samples','rent_samples'], axis=1)

modd  = modd.drop(['male_age_median','male_age_stdev','male_age_samples','female_age_median','female_age_stdev','female_age_samples'], axis=1)

modd= modd.drop(['rent_sample_weight','family_sample_weight','universe_samples','hi_samples','hi_sample_weight','married_snp','pct_own','female_age_sample_weight','male_age_sample_weight','hc_sample_weight','hc_mortgage_sample_weight'],axis=1)


modd=modd.drop('BLOCKID',axis=1)
modd.info()



#changing the datatypes to object
modd['UID'] = modd['UID'].astype('object')
modd['COUNTYID'] = modd['COUNTYID'].astype('object')
modd['STATEID'] = modd['STATEID'].astype('object')
modd['zip_code'] = modd['zip_code'].astype('object')
modd['area_code'] = modd['area_code'].astype('object')
modd['lat'] = modd['lat'].astype('object')
modd['lng'] = modd['lng'].astype('object')
modd['ALand'] = modd['ALand'].astype('object')
modd['AWater'] = modd['AWater'].astype('object')
print(modd.info())

## Checking for null values:

nulls = modd.isna().sum()
print("Null values for each column after modifications:\n")
print(nulls)


## Elimnating Outliers


# function to remove outliers
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


modd_w_out = remove_outliers(modd)


#checking for duplicates
dup = modd_w_out[modd_w_out.duplicated(["UID","pop"])]


#inspecting final dataset
modd_w_out.info()

#converting back the orignal data types so that we can perform analysis
modd_w_out['lat'] = modd_w_out['lat'].astype('float64')
modd_w_out['lng'] = modd_w_out['lng'].astype('float64')
modd_w_out['ALand'] = modd_w_out['ALand'].astype('int64')
modd_w_out['AWater'] = modd_w_out['AWater'].astype('int64')
print(modd_w_out.info())


## Models

## Debt

numfeatures = (
'pop',
'male_pop',
'female_pop',
'hi_mean',
'family_mean',
'hc_mean',
'home_equity_second_mortgage',
'second_mortgage',
'home_equity',
'rent_mean',
'hs_degree',
'hs_degree_male',
'hs_degree_female',
'male_age_mean',
'female_age_mean',
'married',
'separated',
'divorced'
)


# Recategorize debt column to categorical for analysis

median_debt = float(modd_w_out.loc[:,"debt"].median())
LowerQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.25))
UpperQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.75))


modd_w_out['debt_cat_logit'] = np.where(modd_w_out['debt'] <= median_debt, 'Low', 'High')


modd_w_out['debt_cat_dt'] = np.where(modd_w_out['debt'] <= LowerQ_debt, 'Low',
                                  np.where(modd_w_out['debt'] <= median_debt, 'MLow',
                                           np.where(modd_w_out['debt'] < UpperQ_debt, 'MHigh', 'High')))


## Ensambling:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
model3 = SVMclassifier = SVC(kernel = 'linear')
model4 = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3)], voting='hard')
model.fit(X_train,y_train)
print("Ensambling Accuracy : ", model.score(X_test, y_test) * 100)



## Ensambling (Random Forest):
from sklearn.pipeline import Pipeline

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
ry=(modd_w_out.loc[:,"debt"])# target variable
RFregressor = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestRegressor', RandomForestRegressor(n_estimators=100))])
rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

RFR = RFregressor.fit(rX_train, ry_train)

print("Random Forest Regression Accuracy : ", RFR.score(rX_test, ry_test) * 100)

f_importances = pd.Series(RFregressor.named_steps['RandomForestRegressor'].feature_importances_, numfeatures)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.show()


X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_dt"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
RFclassifier = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestClassifier', RandomForestClassifier(n_estimators=100))])

RFC = RFclassifier.fit(X_train, y_train)

print("Random Forest Classifier Accuracy : ", RFC.score(X_test, y_test) * 100)

f_importances = pd.Series(RFclassifier.named_steps['RandomForestClassifier'].feature_importances_, numfeatures)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.show()





# Rent

numfeatures = (
'pop',
'male_pop',
'female_pop',
'hi_mean',
'family_mean',
'hc_mean',
'home_equity_second_mortgage',
'second_mortgage',
'home_equity',
'debt',
'hs_degree',
'hs_degree_male',
'hs_degree_female',
'male_age_mean',
'female_age_mean',
'married',
'separated',
'divorced'
)


# Recategorize debt column to categorical for analysis

median_debt = float(modd_w_out.loc[:,"rent_mean"].median())
LowerQ_debt = float(modd_w_out.loc[:,"rent_mean"].quantile(0.25))
UpperQ_debt = float(modd_w_out.loc[:,"rent_mean"].quantile(0.75))


modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')


modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                  np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                           np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh', 'High')))



## Ensambling:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"rent_cat_logit"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
model3 = SVMclassifier = SVC(kernel = 'linear')
model4 = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3)], voting='hard')
model.fit(X_train,y_train)
print("Ensambling Accuracy : ", model.score(X_test, y_test) * 100)



## Ensambling (Random Forest):
from sklearn.pipeline import Pipeline

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
ry=(modd_w_out.loc[:,"rent_mean"])# target variable
RFregressor = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestRegressor', RandomForestRegressor(n_estimators=100))])
rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

RFR = RFregressor.fit(rX_train, ry_train)

print("Random Forest Regression Accuracy : ", RFR.score(rX_test, ry_test) * 100)

f_importances = pd.Series(RFregressor.named_steps['RandomForestRegressor'].feature_importances_, numfeatures)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.show()


X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"rent_cat_dt"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
RFclassifier = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestClassifier', RandomForestClassifier(n_estimators=100))])

RFC = RFclassifier.fit(X_train, y_train)

print("Random Forest Classifier Accuracy : ", RFC.score(X_test, y_test) * 100)

f_importances = pd.Series(RFclassifier.named_steps['RandomForestClassifier'].feature_importances_, numfeatures)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.show()



## Classes (not eventually used)

# import packages

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
warnings.filterwarnings("ignore")


# create class for preprocessing

class ourPreprocessing():
    def __init__(self, data):
        self.data = data

    def getdf(self):
        return self.data

    def inf(self):
        print(self.data.info())
        nulls = self.data.isna().sum()
        print("Null values for each column before any modifications:\n")
        print(nulls)

    def coldrop(self):
        self.data = self.data.drop(
            ['SUMLEVEL', 'rent_median', 'rent_stdev', 'used_samples', 'hi_median', 'hi_stdev', 'hc_mortgage_median',
             'hc_mortgage_stdev', 'hc_mortgage_samples'], axis=1)

        self.data = self.data.drop(
            ['rent_gt_10', 'rent_gt_15', 'rent_gt_20', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40',
             'rent_gt_50'], axis=1)

        self.data = self.data.drop(
            ['hc_median', 'hc_stdev', 'hc_samples', 'family_median', 'family_stdev', 'family_samples', 'rent_samples'],
            axis=1)

        self.data = self.data.drop(
            ['male_age_median', 'male_age_stdev', 'male_age_samples', 'female_age_median', 'female_age_stdev',
             'female_age_samples'], axis=1)

        self.data = self.data.drop(
            ['rent_sample_weight', 'family_sample_weight', 'universe_samples', 'hi_samples', 'hi_sample_weight',
             'married_snp', 'pct_own', 'female_age_sample_weight', 'male_age_sample_weight', 'hc_sample_weight',
             'hc_mortgage_sample_weight'], axis=1)

        self.data = self.data.drop('BLOCKID', axis=1)
        print(self.data.info())
        return self.data

    def coltypes(self):
        self.data['UID'] = self.data['UID'].astype('object')
        self.data['COUNTYID'] = self.data['COUNTYID'].astype('object')
        self.data['STATEID'] = self.data['STATEID'].astype('object')
        self.data['zip_code'] = self.data['zip_code'].astype('object')
        self.data['area_code'] = self.data['area_code'].astype('object')
        self.data['lat'] = self.data['lat'].astype('object')
        self.data['lng'] = self.data['lng'].astype('object')
        self.data['ALand'] = self.data['ALand'].astype('object')
        self.data['AWater'] = self.data['AWater'].astype('object')
        print(self.data.info())

        print("Null values for each column after modifications:\n")
        print(self.data.isna().sum())

        return self.data

    def remove_outliers(self):
        for name in self.data.columns:
            if is_numeric_dtype(self.data[name]):
                q1 = self.data[name].quantile(0.25)
                q3 = self.data[name].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                self.data = self.data[(self.data[name] > lower_bound) & (self.data[name] < upper_bound)]
        return self.data

    def check_dupl(self):
        print(self.data[self.data.duplicated(["UID", "pop"])])
        print(self.data.info())

    def coltypes_back(self):
        self.data['lat'] = self.data['lat'].astype('float64')
        self.data['lng'] = self.data['lng'].astype('float64')
        self.data['ALand'] = self.data['ALand'].astype('int64')
        self.data['AWater'] = self.data['AWater'].astype('int64')
        print(self.data.info())
        return self.data



class ourViz:
    def __init__(self, data):
        self.data = data

    def hists(self):
        plt.hist(self.data["pop"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Population')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Population Distribution')
        plt.show()

        plt.hist(self.data["male_pop"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Male Population')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Male Population Distribution')
        plt.show()

        plt.hist(self.data["female_pop"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Female Population')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Female Population Distribution')
        plt.show()

        plt.hist(self.data["rent_mean"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Rent')
        plt.ylabel('Population')
        plt.title(r'Rent Distribution')
        plt.show()

        plt.hist(self.data["hi_mean"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Rent')
        plt.ylabel('Household income')
        plt.title(r'Household income distribution')
        plt.show()

        plt.hist(self.data["family_mean"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Rent')
        plt.ylabel('Household income')
        plt.title(r'Mean Family income')
        plt.show()

        plt.hist(self.data["hc_mortgage_mean"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Mean Monthly Mortgage and Owner Cost')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Mean Monthly Mortgage and Owner Costs of specified geographic location')
        plt.show()

        plt.hist(self.data["hc_mean"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Mean Monthly Owner Costs')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of  mean Monthly Owner Costs  of specified geographic location')
        plt.show()

        plt.hist(self.data["home_equity_second_mortgage"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of homes with a second mortgage and home equity loan')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of homes with a second mortgage and home equity loan')
        plt.show()

        plt.hist(self.data["second_mortgage"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percent of houses with a second mortgage')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of percent of houses with a second mortgage')
        plt.show()

        plt.hist(self.data["home_equity"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of homes with a home equity loan.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of  Percentage of homes with a home equity loan.')
        plt.show()

        plt.hist(self.data["debt"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of homes with some type of debt.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of homes with some type of debt.')
        plt.show()

        plt.hist(self.data["hs_degree"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of people with at least high school degree.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of people with at least high school degree.')
        plt.show()

        plt.hist(self.data["hs_degree_male"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of males with at least high school degree.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of males with at least high school degree.')
        plt.show()

        plt.hist(self.data["hs_degree_female"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of females with at least high school degree.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of females with at least high school degree.')
        plt.show()

        plt.hist(self.data["married"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of married people in the geographical area.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of married people in the geographical area.')
        plt.show()

        plt.hist(self.data["separated"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of separated people in the geographical area.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of separated people in the geographical area.')
        plt.show()

        plt.hist(self.data["divorced"], bins=15, color='blue', edgecolor='black')
        plt.xlabel('Percentage of divorced people in the geographical area.')
        plt.ylabel('Frequency')
        plt.title(r'Histogram of Percentage of divorced people in the geographical area.')
        plt.show()




# instantiate class object and apply modules
debtdata = pd.read_csv('real_estate_db.csv', encoding='latin-1')

data = ourPreprocessing(debtdata)
data.inf()
data.coldrop()
data.coltypes()
data.remove_outliers()
data.coltypes_back()
data.check_dupl()
data.getdf()
