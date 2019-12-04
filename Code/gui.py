import tkinter
import zmq.eventloop.ioloop
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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


class debt:
    class pre:
        def load_data(debtdata):
            # information about the dataset
            modd = debtdata.copy(deep=True)
            modd = modd.drop(
                ['SUMLEVEL', 'rent_median', 'rent_stdev', 'used_samples', 'hi_median', 'hi_stdev', 'hc_mortgage_median',
                 'hc_mortgage_stdev', 'hc_mortgage_samples'], axis=1)

            modd = modd.drop(
                ['rent_gt_10', 'rent_gt_15', 'rent_gt_20', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40',
                 'rent_gt_50'], axis=1)

            modd = modd.drop(['hc_median', 'hc_stdev', 'hc_samples', 'family_median', 'family_stdev', 'family_samples',
                              'rent_samples'], axis=1)

            modd = modd.drop(
                ['male_age_median', 'male_age_stdev', 'male_age_samples', 'female_age_median', 'female_age_stdev',
                 'female_age_samples'], axis=1)

            modd = modd.drop(
                ['rent_sample_weight', 'family_sample_weight', 'universe_samples', 'hi_samples', 'hi_sample_weight',
                 'married_snp', 'pct_own', 'female_age_sample_weight', 'male_age_sample_weight', 'hc_sample_weight',
                 'hc_mortgage_sample_weight'], axis=1)
            modd = modd.drop('BLOCKID', axis=1)
            modd['UID'] = modd['UID'].astype('object')
            modd['COUNTYID'] = modd['COUNTYID'].astype('object')
            modd['STATEID'] = modd['STATEID'].astype('object')
            modd['zip_code'] = modd['zip_code'].astype('object')
            modd['area_code'] = modd['area_code'].astype('object')

            # so that the outlier function can ignore these variables
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
            nulls = modd_w_out.info()
            return nulls, modd_w_out

        def print_data(self):
            nulls1, modd_w_out1 = debt.pre.load_data(finaldebtdata)
            master2 = Tk()
            master2.title("LOADING THE DATA AND EXPLORING THE VALUES")
            master2.geometry("600x600")
            texn = tkinter.Text(master=master2)
            texn.insert(tkinter.END, nulls1)
            texn.see(tkinter.END)
            texn.pack()
            master2.mainloop()

    class model1:
        class KNN:
            def out(self):
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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
                # splitting data into training and testing data
                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "debt_cat_logit"])  # target variable
                ry = (modd_w_out.loc[:, "debt"])  # target variable
                class_le = LabelEncoder()
                y = class_le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                # Running a Model for Regression and Classification
                # best KNN value, find new value!
                KNNclassifier = KNeighborsClassifier(n_neighbors=9)
                KNNregressor = KNeighborsRegressor(n_neighbors=9)
                knnC = KNNclassifier.fit(X_train, y_train)
                knnR = KNNregressor.fit(rX_train, ry_train)

                # Classification Prediction Values
                y_pred_c = KNNclassifier.predict(X_test)
                y_pred_r = KNNregressor.predict(X_test)
                # Classification Analysis
                conf_matrix = confusion_matrix(y_test, y_pred_c)
                clfreport = classification_report(y_test, y_pred_c)
                acc_KNN = accuracy_score(y_test, y_pred_c) * 100
                acr_KNN = knnR.score(rX_test, ry_test) * 100
                class_names = modd_w_out.loc[:, "debt_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('KNN.png')
                return conf_matrix, clfreport, acc_KNN, acr_KNN

        class knnstats:
            def __init__(self):
                conf_matrix, clfreport, acc_KNN, acr_KNN = debt.model1.KNN.out(self)
                master3 = Tk()
                master3.title("K Nearest Neighbour")
                master3.geometry("600x600")
                tex = tkinter.Text(master=master3)
                tex.insert(tkinter.END, "Confusion Matrix\n")
                tex.insert(tkinter.END, conf_matrix)
                tex.insert(tkinter.END, "\nClassification Report\n")
                tex.insert(tkinter.END, clfreport)
                tex.insert(tkinter.END, "Accuracy Score for Classification\n")
                tex.insert(tkinter.END, acc_KNN)
                tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
                tex.insert(tkinter.END, acr_KNN)
                tex.see(tkinter.END)
                tex.pack()
                Button(master3, text="DISPLAY KNN CONFUSION MATRIX", command=debt.model1.dispimgKNN).pack()

                master3.mainloop()

        class dispimgKNN:
            def __init__(self):
                master4 = Toplevel()
                master4.title("K Nearest Neighbour Confusion Matrix")
                master4.geometry("600x600")
                # Adding widgets to the root window
                Label(master4, text='KNN Confusion matrix', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                knnimg = PhotoImage(file=r"KNN.png")
                # here, image option is used to
                # set image on button
                Button(master4, image=knnimg).pack(side=TOP)

                master4.mainloop()

    class model2:
        class SVM:
            def out(self):
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "debt_cat_logit"])  # target variable
                # y=(modd_w_out.loc[:,"debt_cat_dt"])# target variable
                yr = (modd_w_out.loc[:, "debt"])  # target variable
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, yr, test_size=0.25, random_state=4)

                class_le = LabelEncoder()
                y = class_le.fit_transform(y)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                # fitting the SVR to the dataset
                SVMclassifier = SVC(kernel='linear')  # for classifiers
                SVMregressor = SVR(kernel='rbf')
                svc = SVMclassifier.fit(X_train, y_train)
                svm = SVMregressor.fit(rX_train, ry_train)

                # Classification Prediction Values
                y_pred_c = SVMclassifier.predict(X_test)
                y_pred_r = SVMregressor.predict(X_test)
                # print(regressor.predict([[0.5,0.6,0.3]]))

                # Classification Analysis
                conf_matrix = confusion_matrix(y_test, y_pred_c)

                svm_report = classification_report(y_test, y_pred_c)

                # printing the accuracy score
                svm_c_acc = accuracy_score(y_test, y_pred_c) * 100
                svm_r_acc = svm.score(rX_test, ry_test) * 100



                def coef_values(coef, names):
                    imp = coef
                    print(imp)
                    imp, names = zip(*sorted(zip(imp.ravel(), names)))

                    imp_pos_10 = imp[-10:]
                    names_pos_10 = names[-10:]
                    imp_neg_10 = imp[:10]
                    names_neg_10 = names[:10]

                    imp_top_20 = imp_neg_10 + imp_pos_10
                    names_top_20 = names_neg_10 + names_pos_10

                    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
                    plt.yticks(range(len(names_top_20)), names_top_20)

                plt.savefig("imp")

                # get the column names
                features_names = (modd_w_out.loc[:, numfeatures]).columns

                # call the function
                coef_values(SVMclassifier.coef_, features_names)

                # Plot Confusion Matrix

                class_names = modd_w_out.loc[:, "debt_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                # Show heat map
                plt.tight_layout()
                plt.savefig('svmfig')
                return conf_matrix, svm_report, svm_c_acc, svm_r_acc

        class svmstats:
            def __init__(self):
                conf_matrix, clfreport, acc_SVM, acr_SVM = debt.model2.SVM.out(self)
                master6 = Tk()
                master6.title("K Nearest Neighbour")
                master6.geometry("600x600")
                tex = tkinter.Text(master=master6)
                tex.insert(tkinter.END, "Confusion Matrix\n")
                tex.insert(tkinter.END, conf_matrix)
                tex.insert(tkinter.END, "\nClassification Report\n")
                tex.insert(tkinter.END, clfreport)
                tex.insert(tkinter.END, "Accuracy Score for Classification\n")
                tex.insert(tkinter.END, acc_SVM)
                tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
                tex.insert(tkinter.END, acr_SVM)
                tex.see(tkinter.END)
                tex.pack()
                Button(master6, text="DISPLAY SVM CONFUSION MATRIX", command=debt.model2.dispimgSVM).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master6.mainloop()

        class dispimgSVM:
            def __init__(self):
                master5 = Toplevel()
                master5.title("K Nearest Neighbour Confusion Matrix")
                master5.geometry("600x600")
                # Adding widgets to the root window
                Label(master5, text='SVM Confusion matrix', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                knnimg = PhotoImage(file=r"KNN.png")
                # Label(master5, text = 'FEATURES DISPLAYED BY IMPORTANCE', font =('Verdana', 15)).pack(side = TOP, pady = 10)
                # Creating a photoimage object to use image
                # imp = PhotoImage(file = r"imp.png")
                # here, image option is used to
                # set image on button
                Button(master5, image=knnimg).pack()
                # Button(master5,  image = imp).pack(side = RIGHT)
                master5.mainloop()

    class model3:
        class LR:
            def out(self):
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "debt_cat_logit"])  # target variable
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

                clf = LogisticRegression()

                # performing training
                clf.fit(X_train, y_train)

                # %%-----------------------------------------------------------------------
                # make predictions
                # predicton on test
                y_pred = clf.predict(X_test)

                y_pred_score = clf.predict_proba(X_test)

                # %%-----------------------------------------------------------------------
                # calculate metrics
                print("\n")

                lracc = accuracy_score(y_test, y_pred) * 100

                roc = roc_auc_score(y_test, y_pred_score[:, 1]) * 100

                # %%-----------------------------------------------------------------------
                # confusion matrix

                conf_matrix = confusion_matrix(y_test, y_pred)
                class_names = modd_w_out.loc[:, "debt_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('LRconfusionmatrix')
                return lracc, roc

        class lrstats:
            def __init__(self):
                lracc, roc = debt.model3.LR.out(self)
                master7 = Tk()
                master7.title("Logistic Regression")
                master7.geometry("600x600")
                tex = tkinter.Text(master=master7)
                tex.insert(tkinter.END, "Logistic Regression Accuracy Score:\n")
                tex.insert(tkinter.END, lracc)
                tex.insert(tkinter.END, "\nROC\n")
                tex.insert(tkinter.END, roc)
                tex.see(tkinter.END)
                tex.pack()
                Button(master7, text="DISPLAY LR CONFUSION MATRIX", command=debt.model3.dispimgLR).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master7.mainloop()

        class dispimgLR:
            def __init__(self):
                master8 = Toplevel()
                master8.title("LINEAR REGRESSION CONFUSION MATRIX")
                master8.geometry("600x600")
                # Adding widgets to the root window
                Label(master8, text='LINEAR REGRESSION', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                lrimg = PhotoImage(file=r"LRconfusionmatrix.png")
                # here, image option is used to
                # set image on button
                Button(master8, image=lrimg).pack(side=TOP)
                master8.mainloop()

    class model4:
        class DT:
            def out(self):
                ## # Descision Tree:
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                ry = (modd_w_out.loc[:, "debt"])  # target variable
                y = (modd_w_out.loc[:, "debt_cat_dt"])  # target variable
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                treeRegressor = DecisionTreeRegressor(max_depth=5)
                treeClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
                dtRegressor = treeRegressor.fit(rX_train, ry_train)
                dtClassifier = treeClassifier.fit(X_train, y_train)

                # predicton on test using entropy
                y_pred_entropy = treeClassifier.predict(X_test)
                acc_c_sc = accuracy_score(y_test, y_pred_entropy) * 100
                acc_r_sc = dtRegressor.score(rX_test, ry_test) * 100

                # confusion matrix for entropy model
                conf_matrix = confusion_matrix(y_test, y_pred_entropy)
                class_names = modd_w_out.loc[:, "debt_cat_dt"].unique()
                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('dtmatrix')

                # %%-----------------------------------------------------------------------
                # display decision tree

                dot_data1 = export_graphviz(dtRegressor, filled=True, rounded=True, class_names=class_names,
                                            feature_names=modd_w_out.loc[:, numfeatures].columns, out_file=None)
                dot_data2 = export_graphviz(dtClassifier, filled=True, rounded=True, class_names=class_names,
                                            feature_names=modd_w_out.loc[:, numfeatures].columns, out_file=None)

                graph1 = graph_from_dot_data(dot_data1)
                graph2 = graph_from_dot_data(dot_data2)
                graph1.write_pdf("decision_tree_entropy1.pdf")
                graph2.write_pdf("decision_tree_entropy2.pdf")
                return acc_c_sc, acc_r_sc

        class dtstats:
            def __init__(self):
                clas_acc, reg_acc = debt.model4.DT.out(self)
                master9 = Tk()
                master9.title("Decision Trees")
                master9.geometry("600x600")
                tex = tkinter.Text(master=master9)
                tex.insert(tkinter.END, "Decision Trees Classification Accuracy Score:\n")
                tex.insert(tkinter.END, clas_acc)
                tex.insert(tkinter.END, "\nDecision Trees Regression Accuracy Score:\n")
                tex.insert(tkinter.END, reg_acc)
                tex.see(tkinter.END)
                tex.pack()
                Button(master9, text="DISPLAY DT CONFUSION MATRIX", command=debt.model4.dispimgDT).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master9.mainloop()

        class dispimgDT:
            def __init__(self):
                master10 = Toplevel()
                master10.title("DESCION TREE CONFUSION MATRIX")
                master10.geometry("600x600")
                # Adding widgets to the root window
                Label(master10, text='DESCISION TREES', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                lrimg = PhotoImage(file=r"dtmatrix.png")
                # here, image option is used to
                # set image on button
                Button(master10, image=lrimg).pack(side=TOP)
                master10.mainloop()

    class model5:
        class ensembling:
            def out(self):
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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
                # ## Ensambling:
                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "debt_cat_logit"])  # target variable
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

                from sklearn.ensemble import VotingClassifier
                modela = LogisticRegression(random_state=1)
                modelb = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
                modelc = SVMclassifier = SVC(kernel='linear')
                modeld = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
                model = VotingClassifier(estimators=[('lr', modela), ('dt', modelb)], voting='hard')
                #    ,('SVM', model3) ,('KNN', model4)
                model.fit(X_train, y_train)
                ensemacc = model.score(X_test, y_test) * 100
                return ensemacc

        class estats:
            def __init__(self):
                acc = debt.model5.ensembling.out(self)
                master11 = Tk()
                master11.title("ENSEMBALING")
                master11.geometry("600x600")
                tex = tkinter.Text(master=master11)
                tex.insert(tkinter.END, "ENSEMBALING ACCURACY SCORE:\n")
                tex.insert(tkinter.END, acc)
                tex.see(tkinter.END)
                tex.pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master11.mainloop()

        class rf:
            def out(self):
                nulls1, modd_w_out = debt.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "debt"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "debt"].quantile(0.75))

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
                'divorced',
                )
                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                ry = (modd_w_out.loc[:, "debt"])  # target variable
                RFregressor = RandomForestRegressor(n_estimators=100, random_state=0)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                RFR = RFregressor.fit(rX_train, ry_train)

                rfacc = RFR.score(rX_test, ry_test) * 100

                return rfacc

        class rfstats:
            def __init__(self):
                racc = debt.model5.rf.out(self)
                master12 = Tk()
                master12.title("RANDOM FOREST")
                master12.geometry("600x600")
                tex = tkinter.Text(master=master12)
                tex.insert(tkinter.END, "RANDOM FOREST ACCURACY SCORE:\n")
                tex.insert(tkinter.END, racc)
                tex.see(tkinter.END)
                tex.pack()
                master12.mainloop()

    class mainwindow:
        def __init__(self):
            masterfinal1 = Tk()
            frame = Frame(masterfinal1, width=200, height=250)
            frame.pack()
            masterfinal1.title("DEBT AND MORTGAGE AND DATA ANALYSIS")
            masterfinal1.geometry("600x600")
            Label(frame, text="RENT DATA ANALYSIS", font=('Verdana', 30), fg="white", bg="black").grid(row=1)
            Button(frame, text="K NEAREST NEIGHBOUR ",bg="light cyan", command=debt.model1.knnstats).grid(row=2)
            Button(frame, text="SUPPORT VECTOR MACHINES ",bg="sky blue", command=debt.model2.svmstats).grid(row=3)
            Button(frame, text="LOGISTIC REGRESSION", bg="snow2",command=debt.model3.lrstats).grid(row=4)
            Button(frame, text="DECISON TREES", bg="light steel blue",command=debt.model4.dtstats).grid(row=5)
            Button(frame, text="ENSEMBALING", bg="pink",command=debt.model5.estats).grid(row=6)
            Button(frame, text="RANDOM FOREST",bg="plum2", command=debt.model5.rfstats).grid(row=7)
            masterfinal1.configure(background='white')
            masterfinal1.mainloop()



class rent:
    class pre:
        def load_data(debtdata):
            # information about the dataset
            modd = debtdata.copy(deep=True)
            modd = modd.drop(
                ['SUMLEVEL', 'rent_median', 'rent_stdev', 'used_samples', 'hi_median', 'hi_stdev', 'hc_mortgage_median',
                 'hc_mortgage_stdev', 'hc_mortgage_samples'], axis=1)

            modd = modd.drop(
                ['rent_gt_10', 'rent_gt_15', 'rent_gt_20', 'rent_gt_25', 'rent_gt_30', 'rent_gt_35', 'rent_gt_40',
                 'rent_gt_50'], axis=1)

            modd = modd.drop(['hc_median', 'hc_stdev', 'hc_samples', 'family_median', 'family_stdev', 'family_samples',
                              'rent_samples'], axis=1)

            modd = modd.drop(
                ['male_age_median', 'male_age_stdev', 'male_age_samples', 'female_age_median', 'female_age_stdev',
                 'female_age_samples'], axis=1)

            modd = modd.drop(
                ['rent_sample_weight', 'family_sample_weight', 'universe_samples', 'hi_samples', 'hi_sample_weight',
                 'married_snp', 'pct_own', 'female_age_sample_weight', 'male_age_sample_weight', 'hc_sample_weight',
                 'hc_mortgage_sample_weight'], axis=1)
            modd = modd.drop('BLOCKID', axis=1)
            modd['UID'] = modd['UID'].astype('object')
            modd['COUNTYID'] = modd['COUNTYID'].astype('object')
            modd['STATEID'] = modd['STATEID'].astype('object')
            modd['zip_code'] = modd['zip_code'].astype('object')
            modd['area_code'] = modd['area_code'].astype('object')

            # so that the outlier function can ignore these variables
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
            return nulls, modd_w_out

        def print_data(self):
            nulls1, modd_w_out1 = rent.pre.load_data(finaldebtdata)
            master2 = Tk()
            master2.title("LOADING THE DATA AND EXPLORING THE VALUES")
            master2.geometry("600x600")
            texn = tkinter.Text(master=master2)
            texn.insert(tkinter.END, nulls1)
            texn.see(tkinter.END)
            texn.pack()
            master2.mainloop()

    class model1:
        class KNN:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "rent_cat_logit"])  # target variable
                ry = (modd_w_out.loc[:, "rent_mean"])  # target variable
                class_le = LabelEncoder()
                y = class_le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                # Running a Model for Regression and Classification
                # best KNN value, find new value!
                KNNclassifier = KNeighborsClassifier(n_neighbors=9)
                KNNregressor = KNeighborsRegressor(n_neighbors=9)
                knnC = KNNclassifier.fit(X_train, y_train)
                knnR = KNNregressor.fit(rX_train, ry_train)

                # Classification Prediction Values
                y_pred_c = KNNclassifier.predict(X_test)
                y_pred_r = KNNregressor.predict(X_test)
                # Classification Analysis
                conf_matrix = confusion_matrix(y_test, y_pred_c)
                clfreport = classification_report(y_test, y_pred_c)
                acc_KNN = accuracy_score(y_test, y_pred_c) * 100
                acr_KNN = knnR.score(rX_test, ry_test) * 100
                class_names = modd_w_out.loc[:, "rent_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('KNN.png')
                return conf_matrix, clfreport, acc_KNN, acr_KNN

        class knnstats():
            def __init__(self):
                conf_matrix, clfreport, acc_KNN, acr_KNN = rent.model1.KNN.out(self)
                master3 = Tk()
                master3.title("K Nearest Neighbour")
                master3.geometry("600x600")
                tex = tkinter.Text(master=master3)
                tex.insert(tkinter.END, "Confusion Matrix\n")
                tex.insert(tkinter.END, conf_matrix)
                tex.insert(tkinter.END, "\nClassification Report\n")
                tex.insert(tkinter.END, clfreport)
                tex.insert(tkinter.END, "Accuracy Score for Classification\n")
                tex.insert(tkinter.END, acc_KNN)
                tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
                tex.insert(tkinter.END, acr_KNN)
                tex.see(tkinter.END)
                tex.pack()
                Button(master3, text="DISPLAY KNN CONFUSION MATRIX", command=rent.model1.dispimgKNN).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master3.mainloop()

        class dispimgKNN():
            def __init__(self):
                master4 = Toplevel()
                master4.title("K Nearest Neighbour Confusion Matrix")
                master4.geometry("600x600")
                # Adding widgets to the root window
                Label(master4, text='KNN Confusion matrix', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                knnimg = PhotoImage(file=r"KNN.png")
                # here, image option is used to
                # set image on button
                Button(master4, image=knnimg).pack(side=TOP)

                master4.mainloop()

    class model2:
        class SVM:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "rent_cat_logit"])  # target variable
                yr = (modd_w_out.loc[:, "rent_mean"])  # target variable
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, yr, test_size=0.25, random_state=4)

                class_le = LabelEncoder()
                y = class_le.fit_transform(y)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                # fitting the SVR to the dataset
                SVMclassifier = SVC(kernel='linear')  # for classifiers
                SVMregressor = SVR(kernel='rbf')
                svc = SVMclassifier.fit(X_train, y_train)
                svm = SVMregressor.fit(rX_train, ry_train)

                # Classification Prediction Values
                y_pred_c = SVMclassifier.predict(X_test)
                y_pred_r = SVMregressor.predict(X_test)
                # print(regressor.predict([[0.5,0.6,0.3]]))

                # Classification Analysis
                conf_matrix = confusion_matrix(y_test, y_pred_c)

                svm_report = classification_report(y_test, y_pred_c)

                # printing the accuracy score
                svm_c_acc = accuracy_score(y_test, y_pred_c) * 100
                svm_r_acc = svm.score(rX_test, ry_test) * 100

                print("Predicted value for SVR: ", y_pred_r)

                # print(regressor.predict([[1,2,3,4,5,6,7,8]]))

                def coef_values(coef, names):
                    imp = coef
                    print(imp)
                    imp, names = zip(*sorted(zip(imp.ravel(), names)))

                    imp_pos_10 = imp[-10:]
                    names_pos_10 = names[-10:]
                    imp_neg_10 = imp[:10]
                    names_neg_10 = names[:10]

                    imp_top_20 = imp_neg_10 + imp_pos_10
                    names_top_20 = names_neg_10 + names_pos_10

                    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
                    plt.yticks(range(len(names_top_20)), names_top_20)
                    plt.savefig("imp")

                # get the column names
                features_names = (modd_w_out.loc[:, numfeatures]).columns

                # call the function
                coef_values(SVMclassifier.coef_, features_names)

                # Plot Confusion Matrix

                class_names = modd_w_out.loc[:, "rent_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                # Show heat map
                plt.tight_layout()
                plt.savefig('svmfig')
                return conf_matrix, svm_report, svm_c_acc, svm_r_acc

        class svmstats:
            def __init__(self):
                conf_matrix, clfreport, acc_SVM, acr_SVM = rent.model2.SVM.out(self)
                master6 = Tk()
                master6.title("K Nearest Neighbour")
                master6.geometry("600x600")
                tex = tkinter.Text(master=master6)
                tex.insert(tkinter.END, "Confusion Matrix\n")
                tex.insert(tkinter.END, conf_matrix)
                tex.insert(tkinter.END, "\nClassification Report\n")
                tex.insert(tkinter.END, clfreport)
                tex.insert(tkinter.END, "Accuracy Score for Classification\n")
                tex.insert(tkinter.END, acc_SVM)
                tex.insert(tkinter.END, "\nAccuracy Score for Regression\n")
                tex.insert(tkinter.END, acr_SVM)
                tex.see(tkinter.END)
                tex.pack()
                Button(master6, text="DISPLAY SVM CONFUSION MATRIX", command=rent.model2.dispimgSVM).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master6.mainloop()

        class dispimgSVM:
            def __init__(self):
                master5 = Toplevel()
                master5.title("K Nearest Neighbour Confusion Matrix")
                master5.geometry("600x600")
                # Adding widgets to the root window
                Label(master5, text='SVM Confusion matrix', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                knnimg = PhotoImage(file=r"KNN.png")
                # Label(master5, text = 'FEATURES DISPLAYED BY IMPORTANCE', font =('Verdana', 15)).pack(side = TOP, pady = 10)
                # Creating a photoimage object to use image
                # imp = PhotoImage(file = r"imp.png")
                # here, image option is used to
                # set image on button
                Button(master5, image=knnimg).pack()
                # Button(master5,  image = imp).pack(side = RIGHT)
                master5.mainloop()

    class model3:
        class LR:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "rent_cat_logit"])  # target variable

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

                clf = LogisticRegression()

                # performing training
                clf.fit(X_train, y_train)

                # %%-----------------------------------------------------------------------
                # make predictions
                # predicton on test
                y_pred = clf.predict(X_test)

                y_pred_score = clf.predict_proba(X_test)

                # %%-----------------------------------------------------------------------
                # calculate metrics

                lracc = accuracy_score(y_test, y_pred) * 100

                roc = roc_auc_score(y_test, y_pred_score[:, 1]) * 100

                # %%-----------------------------------------------------------------------
                # confusion matrix

                conf_matrix = confusion_matrix(y_test, y_pred)
                class_names = modd_w_out.loc[:, "rent_cat_logit"].unique()

                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('LRconfusionmatrix')
                return lracc, roc

        class lrstats:
            def __init__(self):
                lracc, roc = rent.model3.LR.out(self)
                master7 = Tk()
                master7.title("Linear Regression")
                master7.geometry("600x600")
                tex = tkinter.Text(master=master7)
                tex.insert(tkinter.END, "Linear Regression Accuracy Score:\n")
                tex.insert(tkinter.END, lracc)
                tex.insert(tkinter.END, "\nROC\n")
                tex.insert(tkinter.END, roc)
                tex.see(tkinter.END)
                tex.pack()
                Button(master7, text="DISPLAY LR CONFUSION MATRIX", command=rent.model3.dispimgLR).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master7.mainloop()

        class dispimgLR:
            def __init__(self):
                master8 = Toplevel()
                master8.title("LINEAR REGRESSION CONFUSION MATRIX")
                master8.geometry("600x600")
                # Adding widgets to the root window
                Label(master8, text='LINEAR REGRESSION', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                lrimg = PhotoImage(file=r"LRconfusionmatrix.png")
                # here, image option is used to
                # set image on button
                Button(master8, image=lrimg).pack(side=TOP)
                master8.mainloop()

    class model4:
        class DT:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                ry = (modd_w_out.loc[:, "rent_mean"])  # target variable
                y = (modd_w_out.loc[:, "rent_cat_dt"])  # target variable

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                # scalling the data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                rX_train = scaler.transform(rX_train)
                rX_test = scaler.transform(rX_test)

                treeRegressor = DecisionTreeRegressor(max_depth=5)
                treeClassifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
                dtRegressor = treeRegressor.fit(rX_train, ry_train)
                dtClassifier = treeClassifier.fit(X_train, y_train)

                # predicton on test using entropy
                y_pred_entropy = treeClassifier.predict(X_test)
                acc_c_sc = accuracy_score(y_test, y_pred_entropy) * 100
                acc_r_sc = dtRegressor.score(rX_test, ry_test) * 100

                # confusion matrix for entropy model
                conf_matrix = confusion_matrix(y_test, y_pred_entropy)
                class_names = modd_w_out.loc[:, "rent_cat_dt"].unique()
                df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

                plt.figure(figsize=(5, 5))
                hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
                hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
                plt.ylabel('True label', fontsize=20)
                plt.xlabel('Predicted label', fontsize=20)
                plt.tight_layout()
                plt.savefig('dtmatrix')

                # %%-----------------------------------------------------------------------
                # display decision tree

                dot_data1 = export_graphviz(dtRegressor, filled=True, rounded=True, class_names=class_names,
                                            feature_names=modd_w_out.loc[:, numfeatures].columns, out_file=None)
                dot_data2 = export_graphviz(dtClassifier, filled=True, rounded=True, class_names=class_names,
                                            feature_names=modd_w_out.loc[:, numfeatures].columns, out_file=None)

                graph1 = graph_from_dot_data(dot_data1)
                graph2 = graph_from_dot_data(dot_data2)
                graph1.write_pdf("decision_tree_entropy1.pdf")
                graph2.write_pdf("decision_tree_entropy2.pdf")

                return acc_c_sc, acc_r_sc

        class dtstats:
            def __init__(self):
                clas_acc, reg_acc = rent.model4.DT.out(self)
                master9 = Tk()
                master9.title("Decision Trees")
                master9.geometry("600x600")
                tex = tkinter.Text(master=master9)
                tex.insert(tkinter.END, "Decision Trees Classification Accuracy Score:\n")
                tex.insert(tkinter.END, clas_acc)
                tex.insert(tkinter.END, "\nDecision Trees Regression Accuracy Score:\n")
                tex.insert(tkinter.END, reg_acc)
                tex.see(tkinter.END)
                tex.pack()
                Button(master9, text="DISPLAY DT CONFUSION MATRIX", command=rent.model4.dispimgDT).pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master9.mainloop()

        class dispimgDT:
            def __init__(self):
                master10 = Toplevel()
                master10.title("DESCION TREE CONFUSION MATRIX")
                master10.geometry("600x600")
                # Adding widgets to the root window
                Label(master10, text='DESCISION TREES', font=('Verdana', 15)).pack(side=TOP, pady=10)
                # Creating a photoimage object to use image
                lrimg = PhotoImage(file=r"dtmatrix.png")
                # here, image option is used to
                # set image on button
                Button(master10, image=lrimg).pack(side=TOP)
                master10.mainloop()

    class model5:
        class ensembling:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                y = (modd_w_out.loc[:, "rent_cat_logit"])  # target variable

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

                from sklearn.ensemble import VotingClassifier
                modela = LogisticRegression(random_state=1)
                modelb = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
                modelc = SVMclassifier = SVC(kernel='linear')
                modeld = KNNclassifier = KNeighborsClassifier(n_neighbors=9)
                model = VotingClassifier(estimators=[('lr', modela), ('dt', modelb)], voting='hard')
                #    ,('SVM', model3) ,('KNN', model4)
                model.fit(X_train, y_train)
                ensemacc = model.score(X_test, y_test) * 100
                return ensemacc

        class estats:
            def __init__(self):
                acc = rent.model5.ensembling.out(self)
                master11 = Tk()
                master11.title("ENSEMBALING")
                master11.geometry("600x600")
                tex = tkinter.Text(master=master11)
                tex.insert(tkinter.END, "ENSEMBALING ACCURACY SCORE:\n")
                tex.insert(tkinter.END, acc)
                tex.see(tkinter.END)
                tex.pack()

                # print("Predicted value for KNN: ",y_pred_r )
                master11.mainloop()

        class rf:
            def out(self):
                nulls1, modd_w_out = rent.pre.load_data(finaldebtdata)
                median_debt = float(modd_w_out.loc[:, "rent_mean"].median())
                LowerQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.25))
                UpperQ_debt = float(modd_w_out.loc[:, "rent_mean"].quantile(0.75))

                modd_w_out['rent_cat_logit'] = np.where(modd_w_out['rent_mean'] <= median_debt, 'Low', 'High')

                modd_w_out['rent_cat_dt'] = np.where(modd_w_out['rent_mean'] <= LowerQ_debt, 'Low',
                                                     np.where(modd_w_out['rent_mean'] <= median_debt, 'MLow',
                                                              np.where(modd_w_out['rent_mean'] < UpperQ_debt, 'MHigh',
                                                                       'High')))

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

                X = (modd_w_out.loc[:, numfeatures])  # variables to fit the data
                ry = (modd_w_out.loc[:, "rent_mean"])  # target variable
                RFregressor = RandomForestRegressor(n_estimators=100, random_state=0)
                rX_train, rX_test, ry_train, ry_test = train_test_split(X, ry, test_size=0.25, random_state=4)

                RFR = RFregressor.fit(rX_train, ry_train)

                rfacc = RFR.score(rX_test, ry_test) * 100

                return rfacc

        class rfstats:
            def __init__(self):
                racc = rent.model5.rf.out(self)
                master12 = Tk()
                master12.title("RANDOM FOREST")
                master12.geometry("600x600")
                tex = tkinter.Text(master=master12)
                tex.insert(tkinter.END, "RANDOM FOREST ACCURACY SCORE:\n")
                tex.insert(tkinter.END, racc)
                tex.see(tkinter.END)
                tex.pack()
                master12.mainloop()

    class mainwindow:
        def __init__(self):
            masterfinal = Tk()
            frame = Frame(masterfinal, width=200, height=250)
            frame.grid(row=0)
            masterfinal.title("DEBT AND MORTGAGE AND DATA ANALYSIS")
            masterfinal.geometry("600x600")
            Label(frame, text="RENT DATA ANALYSIS", font=('Verdana', 30), fg="white", bg="black").grid(row=1)
            Button(frame, text="K NEAREST NEIGHBOUR ",bg="light cyan", command=rent.model1.knnstats).grid(row=2)
            Button(frame, text="SUPPORT VECTOR MACHINES ",bg="light sky blue", command=rent.model2.svmstats).grid(row=3)
            Button(frame, text="LOGISTIC REGRESSION",bg="gray",command=rent.model3.lrstats).grid(row=4)
            Button(frame, text="DECISON TREES", bg="light pink",command=rent.model4.dtstats).grid(row=5)
            Button(frame, text="ENSEMBALING", bg="khaki1", command=rent.model5.estats).grid(row=6)
            Button(frame, text="RANDOM FOREST", bg="thistle1", command=rent.model5.rfstats).grid(row=7)
            masterfinal.configure(background='white')
            masterfinal.mainloop()
class mainviz():
    class debt():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Debt Distribution', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"debt.png")

            # here, image option is used to
            # set image on button
            Button(master, image = self.photo).pack(side = TOP)
            master.mainloop()
    class pop():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of Population', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"popdist.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class malepop():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of Male Population', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"malepop.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()


    class femalepop():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of Female Population', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"femaledist.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()


    class divorce():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of number of people divorced', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"divorced.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class equity():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of equity', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"equity.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class mean_of_income():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of family income', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"familyinc.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()


    class femalehs():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of number of females with highschool degree', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"femalehs.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class highschool():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of people with highschool degree', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"highschool.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class homes():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of homes with a second mortgage', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"homes.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class household():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of household income', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"household.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class malehs():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of males with high school degree', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"malehs.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class married():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of number of married people', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"married.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class monthlymorg():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of mean of monthly morgage and owner cost', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"monthlymorg.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class ownercost():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of mean of monthly morgage', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"ownercost.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class rentdist():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of rent', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"rentdist.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class secondmorgage():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of houses with second mortgage', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"secondmorgage.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()

    class seperated():
        def __init__(self):
            master = Toplevel()
             # Adding widgets to the root window
            Label(master, text = 'Distribution of number of people who are seperated', font =(
              'Verdana', 15)).pack(side = TOP, pady = 10)

            # Creating a photoimage object to use image
            self.photo = PhotoImage(file = r"seperated.png")

            # here, image option is used to
            # set image on button
            Button(master,  image = self.photo).pack(side = TOP)
            master.mainloop()


    class main():
        def __init__(self):
            master = Tk()
            master.title("VISUALIZATION")
            master.geometry("600x600")
            disp1=Button(master,text="DISTRIBUTION OF DEBT",bg="light cyan",command=mainviz.debt)
            disp1.pack()
            disp2=Button(master,text="DISTRIBUTION OF POPULATION",bg="gainsboro",command=mainviz.pop)
            disp2.pack()
            disp3=Button(master,text="DISTRIBUTION OF MALE POPULATION",bg="azure",command=mainviz.malepop)
            disp3.pack()
            disp4=Button(master,text="DISTRIBUTION OF FEMALE POPULATION",bg="alice blue",command=mainviz.femalepop)
            disp4.pack()
            disp5=Button(master,text="DISTRIBUTION OF DIVORCE",bg="thistle",command=mainviz.divorce)
            disp5.pack()
            disp6=Button(master,text="DISTRIBUTION OF MEAN OF FAMILY INCOME ",bg="seashell2",command=mainviz.mean_of_income)
            disp6.pack()
            disp7=Button(master,text="DISTRIBUTION OF NUMBER OF FEMALES WHO HAVE A HIGHSCHOOL DEGREE",bg="thistle",command=mainviz.femalehs)
            disp7.pack()
            disp8=Button(master,text="DISTRIBUTION OF NUMBER WITH A HIGHSCHOOL DEGREE",bg="light pink",command=mainviz.highschool)
            disp8.pack()
            disp9=Button(master,text="DISTRIBUTION OF HOUSEHOLD INCOME",bg="light yellow",command=mainviz.household)
            disp9.pack()
            disp10=Button(master,text="DISTRIBUTION OF MALES WITH A HIGH SCHOOL DEGREE",bg="MistyRose2",command=mainviz.malehs)
            disp10.pack()
            disp11=Button(master,text="DISTRIBUTION OF NUMBER OF MARRIED PEOPLE",bg="SkyBlue2",command=mainviz.married)
            disp11.pack()
            disp12=Button(master,text="DISTRIBUTION OF MEAN OF MONTHLY MORTGAGE AND OWNER COST",bg="tan1",command=mainviz.monthlymorg)
            disp12.pack()
            disp12=Button(master,text="DISTRIBUTION OF MEAN OF MONTHLY MORTGAGE",bg="powder blue",command=mainviz.ownercost)
            disp12.pack()
            disp13=Button(master,text="DISTRIBUTION OF RENT",bg="pale violet red",command=mainviz.rentdist)
            disp13.pack()
            disp14=Button(master,text="DISTRIBUTION OF HOUSES WITH A SECOND MORTGAGE",bg="linen",command=mainviz.secondmorgage)
            disp14.pack()
            disp15=Button(master,text="DISTRIBUTION OF NUMBER OF PEOPLE WHO ARE SEPERATED",bg="rosy brown",command=mainviz.seperated)
            disp15.pack()
            master.configure(background='white')
            master.mainloop()


class lastwindow:
        masterfinal = Tk()
        frame = Frame(masterfinal, width=500, height=2500, bd=1, relief=RIDGE)
        frame.grid(row=0)
        # separator = Frame(height=2, bd=1, relief=SUNKEN)
        # separator.pack(fill=X, padx=5, pady=5)
        masterfinal.title("DEBT AND MORTGAGE AND DATA ANALYSIS")
        masterfinal.geometry("600x600")
        Label(frame, text="MAIN MENU",font =('Verdana', 30),fg="white",bg="black").grid(row=1, column=4)
        Button(frame, text="ANALYSIS ON RENT DATA",bg="misty rose", command=rent.mainwindow).grid(row=5, column=4)
        Button(frame, text="ANALYSIS ON DEBT DATA",bg="lavender", command=debt.mainwindow).grid(row=6, column=4)

        Button(frame, text="LOADING THE DATA AND VIZUALIZTION  ",bg="linen", command=mainviz.main).grid(row=4, column=4)
        masterfinal.configure(background='white')
        masterfinal.mainloop()