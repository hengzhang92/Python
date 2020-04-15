import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


titanic_df = pd.read_csv('titanic/titanic_processed.csv')
Features = list(titanic_df.columns[1:])
result_dict ={}

def summarize_classification(y_test,y_pred):
    acc = accuracy_score(y_test,y_pred,normalize =True)
    num_acc = accuracy_score(y_test,y_pred,normalize = False)
    prec = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    return{'accuracy': acc,
           'precision':prec,
           'recall':recall,
           'accuracy_count':num_acc}

def build_model(classifier_fn,name_of_y_col,names_of_x_cols,dataset,test_frac=0.2):
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    model = classifier_fn(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    train_summary = summarize_classification(y_train,y_pred_train)
    test_summary = summarize_classification(y_test,y_pred)
    pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
    return {'training': train_summary,
            'test': test_summary,
            'confusion_matrix': model_crosstab}

def compare_results():
    for key in result_dict:
        print('Clasification:' + key)
        print()
        print('Training data')
        for score in result_dict[key]['training']:
            print(score,result_dict[key]['training'][score])
        print()
        print('Test data')
        for score in result_dict[key]['test']:
            print(score,result_dict[key]['test'][score])
        print()

# logistic regression
def logistic_fun(x_train,y_train):
    model = LogisticRegression(solver = 'liblinear')
    model.fit(x_train,y_train)
    return model

def linear_discriminant_fn(x_train,y_train,solver='svd'):
    model= LinearDiscriminantAnalysis(solver=solver)
    model.fit(x_train,y_train)
    return model

def quadratic_discriminant_fn(x_train,y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train,y_train)
    return model

def sgd_fn(x_train,y_train,max_iter=1000,tol=1e-3)
    model=SGDClassifier(max_iter=max_iter,tol=tol)
    model.fit(x_train,y_train)
    return model

def linear_svc_fn(x_train,y_train,C=1.0,max_iter=1000,tol=1e-3):
    model = LinearSVC(C=C,max_iter=max_iter,tol=tol,dual=False)
    model.fit(x_train,y_train)
    return model

result_dict['survived - logistic'] = build_model(logistic_fun,'Survived',Features[0:-1],titanic_df)
result_dict['linear_discriminant_analysis'] = build_model(linear_discriminant_fn,'Survived',Features[0:-1],titanic_df)
result_dict['survived - quadratic_discriminant_analysis']=build_model(quadratic_discriminant_fn,'Survived',Features[0:-1],titanic_df)
result_dict['survived-linearsvc'] = build_model(linear_svc_fn,'Survived',Features[0:-1],titanic_df)
compare_results()
