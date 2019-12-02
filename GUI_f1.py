#!/usr/bin/env python
# coding: utf-8

# In[37]:


import tkinter
from tkinter import font
from tkinter import * 
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
finaldebtdata = pd.read_csv('real_estate_db.csv', encoding='latin-1')
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
class pre():
    def load_data(debtdata):
        #information about the dataset
        modd=debtdata.copy(deep=True)
        modd = modd.drop(['SUMLEVEL','rent_median','rent_stdev','used_samples','hi_median','hi_stdev','hc_mortgage_median','hc_mortgage_stdev','hc_mortgage_samples'], axis=1)

        modd  = modd.drop(['rent_gt_10','rent_gt_15','rent_gt_20','rent_gt_25','rent_gt_30','rent_gt_35','rent_gt_40','rent_gt_50'], axis=1)

        modd  = modd.drop(['hc_median','hc_stdev','hc_samples','family_median','family_stdev','family_samples','rent_samples'], axis=1)

        modd  = modd.drop(['male_age_median','male_age_stdev','male_age_samples','female_age_median','female_age_stdev','female_age_samples'], axis=1)

        modd= modd.drop(['rent_sample_weight','family_sample_weight','universe_samples','hi_samples','hi_sample_weight','married_snp','pct_own','female_age_sample_weight','male_age_sample_weight','hc_sample_weight','hc_mortgage_sample_weight'],axis=1)
        modd=modd.drop('BLOCKID',axis=1)
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
        modd_w_out['lat'] = modd_w_out['lat'].astype('float64')
        modd_w_out['lng'] = modd_w_out['lng'].astype('float64')
        modd_w_out['ALand'] = modd_w_out['ALand'].astype('int64')
        modd_w_out['AWater'] = modd_w_out['AWater'].astype('int64')
        nulls = modd_w_out.isna().sum()
        return nulls,modd_w_out    
    def print_data():
        nulls1,modd_w_out1=pre.load_data(finaldebtdata)
        master2 = Tk() 
        master2.title("LOADING THE DATA AND EXPLORING THE VALUES")
        master2.geometry("600x600")
        texn = tkinter.Text()
        texn.insert(tkinter.END, nulls1)
        texn.see(tkinter.END)
        texn.pack()
        next22=Button(master2,text="NEXT",command=model1.knnstats)
        next22.pack()
        master2.mainloop()
    
class model1():       
    def KNN():
           
            nulls1,modd_w_out=pre.load_data(finaldebtdata)
            median_debt = float(modd_w_out.loc[:,"debt"].median())
            LowerQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.25))
            UpperQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.75))


            modd_w_out['debt_cat_logit'] = np.where(modd_w_out['debt'] <= median_debt, 'Low', 'High')


            modd_w_out['debt_cat_dt'] = np.where(modd_w_out['debt'] <= LowerQ_debt, 'Low',
                                              np.where(modd_w_out['debt'] <= median_debt, 'MLow',
                                                       np.where(modd_w_out['debt'] < UpperQ_debt, 'MHigh', 'High')))



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
                'separated',
                'divorced'
                    )
            #splitting data into training and testing data
            X=(modd_w_out.loc[:,numfeatures])#variables to fit the data
            y=(modd_w_out.loc[:,"debt_cat_logit"])# target variable
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
            #best KNN value, find new value! 
            KNNclassifier = KNeighborsClassifier(n_neighbors=9)
            KNNregressor = KNeighborsRegressor(n_neighbors=9)
            knnC = KNNclassifier.fit(X_train, y_train)
            knnR = KNNregressor.fit(rX_train, ry_train)

            #Classification Prediction Values
            y_pred_c = KNNclassifier.predict(X_test)
            y_pred_r=KNNregressor.predict(X_test)
            #Classification Analysis
            conf_matrix = confusion_matrix(y_test, y_pred_c)
            clfreport=classification_report(y_test, y_pred_c)
            acc_KNN=accuracy_score(y_test, y_pred_c) * 100
            acr_KNN= knnR.score(rX_test, ry_test) * 100
            class_names = modd_w_out.loc[:,"debt_cat_logit"].unique()

            df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

            plt.figure(figsize=(5,5))
            hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
            hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
            hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
            plt.ylabel('True label',fontsize=20)
            plt.xlabel('Predicted label',fontsize=20)
            plt.tight_layout()
            plt.savefig('KNN.png')
            return conf_matrix,clfreport,acc_KNN,acr_KNN

    def knnstats():
            conf_matrix,clfreport,acc_KNN,acr_KNN=model1.KNN()
            master3 = Tk() 
            master3.title("K Nearest Neighbour")
            master3.geometry("600x600")
            tex = tkinter.Text()
            tex.insert(tkinter.END,"Confusion Matrix\n")
            tex.insert(tkinter.END,conf_matrix)
            tex.insert(tkinter.END, "\nClassification Report\n")
            tex.insert(tkinter.END,clfreport)
            tex.insert(tkinter.END, "Accuracy Score for Classification\n")
            tex.insert(tkinter.END,acc_KNN)
            tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
            tex.insert(tkinter.END,acr_KNN)
            tex.see(tkinter.END)
            tex.pack()
            Button(master3,text="DISPLAY KNN CONFUSION MATRIX",command=model1.dispimgKNN).pack()

            # print("Predicted value for KNN: ",y_pred_r )
            master3.mainloop()
    def dispimgKNN():
            master4 = Toplevel() 
            master4.title("K Nearest Neighbour Confusion Matrix")
            master4.geometry("600x600")
            # Adding widgets to the root window 
            Label(master4, text = 'KNN Confusion matrix', font =('Verdana', 15)).pack(side = TOP, pady = 10) 
            # Creating a photoimage object to use image 
            knnimg = PhotoImage(file = r"KNN.png") 
            # here, image option is used to 
            # set image on button 
            Button(master4,  image = knnimg).pack(side = TOP)
            
            Button(master4,text="Perform SVM", command=model2.svmstats).pack()
            master4.mainloop()
            
            
class model2():       
    def SVM():
           
            nulls1,modd_w_out=pre.load_data(finaldebtdata)
            median_debt = float(modd_w_out.loc[:,"debt"].median())
            LowerQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.25))
            UpperQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.75))


            modd_w_out['debt_cat_logit'] = np.where(modd_w_out['debt'] <= median_debt, 'Low', 'High')


            modd_w_out['debt_cat_dt'] = np.where(modd_w_out['debt'] <= LowerQ_debt, 'Low',
                                              np.where(modd_w_out['debt'] <= median_debt, 'MLow',
                                                       np.where(modd_w_out['debt'] < UpperQ_debt, 'MHigh', 'High')))



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
                'separated',
                'divorced'
                    )            
            
            
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
            SVMclassifier = SVC(kernel = 'linear')# for classifiers
            SVMregressor = SVR(kernel = 'rbf')
            svc = SVMclassifier.fit(X_train, y_train)
            svm = SVMregressor.fit(rX_train, ry_train)

            #Classification Prediction Values
            y_pred_c = SVMclassifier.predict(X_test)
            y_pred_r=SVMregressor.predict(X_test)
            #print(regressor.predict([[0.5,0.6,0.3]]))

            #Classification Analysis
            conf_matrix = confusion_matrix(y_test, y_pred_c)

            svm_report=classification_report(y_test, y_pred_c)

            #printing the accuracy score
            svm_c_acc= accuracy_score(y_test, y_pred_c) * 100
            svm_r_acc= svm.score(rX_test, ry_test) * 100


            print("Predicted value for SVR: ",y_pred_r )
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
                plt.savefig("imp")

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
            plt.savefig('svmfig')
            return conf_matrix,svm_report,svm_c_acc,svm_r_acc

    def svmstats():
            conf_matrix,clfreport,acc_SVM,acr_SVM=model2.SVM()
            master6 = Tk() 
            master6.title("K Nearest Neighbour")
            master6.geometry("600x600")
            tex = tkinter.Text()
            tex.insert(tkinter.END,"Confusion Matrix\n")
            tex.insert(tkinter.END,conf_matrix)
            tex.insert(tkinter.END, "\nClassification Report\n")
            tex.insert(tkinter.END,clfreport)
            tex.insert(tkinter.END, "Accuracy Score for Classification\n")
            tex.insert(tkinter.END,acc_SVM)
            tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
            tex.insert(tkinter.END,acr_SVM)
            tex.see(tkinter.END)
            tex.pack()
            Button(master6,text="DISPLAY SVM CONFUSION MATRIX",command=model2.dispimgSVM).pack()

            # print("Predicted value for KNN: ",y_pred_r )
            master6.mainloop()    
    def dispimgSVM():
            master5 = Toplevel() 
            master5.title("K Nearest Neighbour Confusion Matrix")
            master5.geometry("600x600")
            # Adding widgets to the root window 
            Label(master5, text = 'SVM Confusion matrix', font =('Verdana', 15)).pack(side = TOP, pady = 10) 
            # Creating a photoimage object to use image 
            knnimg = PhotoImage(file = r"KNN.png") 
            #Label(master5, text = 'FEATURES DISPLAYED BY IMPORTANCE', font =('Verdana', 15)).pack(side = TOP, pady = 10) 
            # Creating a photoimage object to use image 
            #imp = PhotoImage(file = r"imp.png") 
            # here, image option is used to 
            # set image on button 
            Button(master5,  image = knnimg).pack(side = LEFT)
            #Button(master5,  image = imp).pack(side = RIGHT)
            nextsvr=Button(master5,text="PERFORM LOGISTIC REGRESSION", command=model3.lrstats)
            nextsvr.pack()
            master5.mainloop()
            
class model3():       
    def LR():
           
            nulls1,modd_w_out=pre.load_data(finaldebtdata)
            median_debt = float(modd_w_out.loc[:,"debt"].median())
            LowerQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.25))
            UpperQ_debt = float(modd_w_out.loc[:,"debt"].quantile(0.75))


            modd_w_out['debt_cat_logit'] = np.where(modd_w_out['debt'] <= median_debt, 'Low', 'High')


            modd_w_out['debt_cat_dt'] = np.where(modd_w_out['debt'] <= LowerQ_debt, 'Low',
                                              np.where(modd_w_out['debt'] <= median_debt, 'MLow',
                                                       np.where(modd_w_out['debt'] < UpperQ_debt, 'MHigh', 'High')))



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
                'separated',
                'divorced'
                    )  
            
            
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

            lracc= accuracy_score(y_test, y_pred) * 100


            roc= roc_auc_score(y_test,y_pred_score[:,1]) * 100

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
            plt.savefig('LRconfusionmatrix')
            return lracc,roc
    def lrstats():
            lracc,roc=model3.LR()
            master7 = Tk() 
            master7.title("Linear Regression")
            master7.geometry("600x600")
            tex = tkinter.Text()
            tex.insert(tkinter.END,"Linear Regression Accuracy Score:\n")
            tex.insert(tkinter.END,lracc)
            tex.insert(tkinter.END, "\nROC\n")
            tex.insert(tkinter.END,roc)
            tex.see(tkinter.END)
            tex.pack()
            Button(master7,text="DISPLAY LR CONFUSION MATRIX",command=model3.dispimgLR).pack()

            # print("Predicted value for KNN: ",y_pred_r )
            master7.mainloop()    
    def dispimgLR():
            master8 = Toplevel() 
            master8.title("LINEAR REGRESSION CONFUSION MATRIX")
            master8.geometry("600x600")
            # Adding widgets to the root window 
            Label(master8, text = 'LINEAR REGRESSION', font =('Verdana', 15)).pack(side = TOP, pady = 10) 
            # Creating a photoimage object to use image 
            lrimg = PhotoImage(file = r"LRconfusionmatrix.png")  
            # here, image option is used to 
            # set image on button 
            Button(master8,  image = lrimg).pack(side = TOP)
            nextsvr=Button(master8,"TREES")
            master8.mainloop()


# In[38]:


pre.print_data()


# In[ ]:




