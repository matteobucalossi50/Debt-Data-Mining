
#Regressions

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

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

numfeatures = (
'pop',
'male_pop',
'female_pop',
'rent_mean',
'hi_mean',
'family_mean',
'hc_mortgage_mean',
'hc_mean',
'home_equity_second_mortgage',
'second_mortgage',
'home_equity',
'hs_degree',
'hs_degree_male',
'hs_degree_female',
'male_age_mean',
'female_age_mean',
'married',
'divorced',
'separated'
)

#'debt_cdf',
#'home_equity_cdf',

# Recategorize debt column to categorical for analysis

median_debt = float(modd_w_out.loc[:,"debt"].median())
LowerQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.25))
UpperQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.75))


modd_w_out['debt_cat_logit'] = np.where(modd_w_out['debt'] <= median_debt, 'Low', 'High')


modd_w_out['debt_cat_dt'] = np.where(modd_w_out['debt'] <= LowerQ_debt, 'Low',
                                  np.where(modd_w_out['debt'] <= median_debt, 'MLow',
                                           np.where(modd_w_out['debt'] < UpperQ_debt, 'MHigh', 'High')))









# ## KNN:

#splitting data into training and testing data
X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
#y=(modd_w_out.loc[:,"debt_cat_dt"])# target variable
ry = (modd_w_out.loc[:,"debt"])# target variable

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

#scalling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
rX_train = scaler.transform(rX_train)
rX_test = scaler.transform(rX_test)

#Running a Model for Regression and Classification
KNNclassifier = KNeighborsClassifier(n_neighbors=9)
KNNregressor = KNeighborsRegressor(n_neighbors=9)
knnC = KNNclassifier.fit(X_train, y_train)
knnR = KNNregressor.fit(rX_train, ry_train)

#Classification Prediction Values
y_pred = KNNclassifier.predict(X_test)

#Classification Analysis
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("\n")
print(classification_report(y_test, y_pred))
print("\n")

#printing the accuracy score
print("KNN Classifier Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("KNN Regressor Accuracy : ", knnR.score(rX_test, ry_test) * 100)
print("\n")

# Displaying confusion matrix
class_names = modd_w_out.loc[:,"debt_cat_logit"].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()











# ## SVR:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
#y=(modd_w_out.loc[:,"debt_cat_dt"])# target variable
yr=(modd_w_out.loc[:,"debt"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
rX_train, rX_test, ry_train, ry_test = train_test_split(X, yr, test_size=0.25, random_state=4)

class_le = LabelEncoder()
y = class_le.fit_transform(y)

#scalling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
rX_train = scaler.transform(rX_train)
rX_test = scaler.transform(rX_test)

#fitting the SVR to the dataset
SVMclassifier = SVC(kernel = 'linear')
SVMregressor = SVR(kernel = 'rbf')
svc = SVMclassifier.fit(X_train, y_train)
svm = SVMregressor.fit(rX_train, ry_train)

#Classification Prediction Values
y_pred = SVMclassifier.predict(X_test)

#Classification Analysis
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("\n")
print(classification_report(y_test, y_pred))
print("\n")

#printing the accuracy score
print("SVM Classifier Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("SVM Regression Accuracy : ", svm.score(rX_test, ry_test) * 100)
print("\n")

#print(regressor.predict([[1,2,3,4,5,6,7,8]]))

def coef_values(coef, names):
    imp = coef
    print(imp)
    imp,names = zip(*sorted(zip(imp.ravel(),names)))

    imp_pos_10 = imp[-10:]
    names_pos_10 = names[-10:]
    imp_neg_10 = imp[:10]
    names_neg_10 = names[:10]

    imp_top_20 = imp_neg_10+imp_pos_10
    names_top_20 =  names_neg_10+names_pos_10

    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()

# get the column names
features_names = (modd_w_out.loc[:,numfeatures]).columns

# call the function
coef_values(SVMclassifier.coef_, features_names)

#Plot Confusion Matrix

class_names = modd_w_out.loc[:,"debt_cat_logit"].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()










## # Logistic Regression:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

clf = LogisticRegression()

# performing training
clf.fit(X_train, y_train)

#%%-----------------------------------------------------------------------
# make predictions
# predicton on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

#%%-----------------------------------------------------------------------
# calculate metrics
print("\n")

print("Logistic Regression Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

# %%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = modd_w_out.loc[:,"debt_cat_logit"].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()



















## # Descision Tree:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
ry=(modd_w_out.loc[:,"debt"])# target variable
y=(modd_w_out.loc[:,"debt_cat_dt"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

#scalling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
rX_train = scaler.transform(rX_train)
rX_test = scaler.transform(rX_test)

treeRegressor = DecisionTreeRegressor(max_depth=5)
treeClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=0)
dtRegressor = treeRegressor.fit(rX_train, ry_train)
dtClassifier = treeClassifier.fit(X_train, y_train)

# predicton on test using entropy
y_pred_entropy = treeClassifier.predict(X_test)
print("Decision Tree Classifier Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print("Decision Tree Regression Accuracy : ", dtRegressor.score(rX_test, ry_test) * 100)

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = modd_w_out.loc[:,"debt_cat_dt"].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree

dot_data1 = export_graphviz(dtRegressor, filled=True, rounded=True, class_names=class_names, feature_names=modd_w_out.loc[:,numfeatures].columns, out_file=None)
dot_data2 = export_graphviz(dtClassifier, filled=True, rounded=True, class_names=class_names, feature_names=modd_w_out.loc[:,numfeatures].columns, out_file=None)



graph1 = graph_from_dot_data(dot_data1)
graph2 = graph_from_dot_data(dot_data2)
graph1.write_pdf("decision_tree_entropy1.pdf")
graph2.write_pdf("decision_tree_entropy2.pdf")
webbrowser.open_new(r'decision_tree_entropy1.pdf')
webbrowser.open_new(r'decision_tree_entropy2.pdf')











# ## Ensambling:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
model3 = SVMclassifier = SVC(kernel = 'linear')
model4 = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
#    ,('SVM', model3) ,('KNN', model4)
model.fit(X_train,y_train)
print("Ensambling Accuracy : ", model.score(X_test, y_test) * 100)

# ## Ensambling (Bagging):

bag1 = modd_w_out.sample(2000)

X1=(bag1.loc[:,numfeatures])#variables to fit the data
y1=(bag1.loc[:,"debt_cat_logit"])# target variable
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=4)

model1 = DecisionTreeClassifier()
model1.fit(X1_train,y1_train)
pred1=model1.predict_proba(X1_test)

bag2 = modd_w_out.sample(2000)

X2=(bag2.loc[:,numfeatures])#variables to fit the data
y2=(bag2.loc[:,"debt_cat_logit"])# target variable
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=4)

model2 = DecisionTreeClassifier()
model2.fit(X2_train,y2_train)
pred2=model1.predict_proba(X2_test)

finalpred=(pred1+pred2)/2

# ## Ensambling (Random Forest):

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
ry=(modd_w_out.loc[:,"debt"])# target variable
RFregressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

RFR = RFregressor.fit(rX_train, ry_train)

print("Random Forest Regression Accuracy : ", RFR.score(rX_test, ry_test) * 100)