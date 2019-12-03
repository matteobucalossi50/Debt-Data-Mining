# ## Ensambling:

X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
model3 = SVMclassifier = SVC(kernel = 'linear')
model4 = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3)], voting='hard')
#    ,('SVM', model3) ,('KNN', model4)
model.fit(X_train,y_train)
print("Ensambling Accuracy : ", model.score(X_test, y_test) * 100)



# ## Ensambling (Random Forest):
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
#plt.savefig('RFfeatures.png', bbox_inches='tight', pad_inches=0.25)
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
#plt.savefig('RFfeatures.png', bbox_inches='tight', pad_inches=0.25)
plt.show()
