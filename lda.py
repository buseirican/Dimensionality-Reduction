import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA #unsupervised
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pylab import *
from sklearn.ensemble import RandomForestClassifier 

def randomFOREST(X_train,X_test,y_train,y_test):
    classifier=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
    classifier.fit(X_train,y_train)
    print(classifier.score(X_test,y_test))
    #print(classifier.score(X_train,y_train))
    pred=classifier.predict(X_test)
    #print(confusion_matrix(y_test,pred))
    
def LogisticReg(X_train,X_test,y_train,y_test):
    logreg=LogisticRegression(solver="lbfgs",multi_class='auto',max_iter=1000)
    logreg.fit(X_train,y_train)
    print(logreg.score(X_test,y_test))
    #print(logreg.score(X_train,y_train))
    
def LinearReg(X_train,X_test,y_train,y_test):
    linreg=LinearRegression()
    linreg.fit(X_train,y_train)
    print(linreg.score(X_test,y_test))
    #print(linreg.score(X_train,y_train))
    
def fill_missing(df):
    num_vars=list(df.loc[:,df.dtypes!='object'].columns) #numeric olan attributeları liste yap
    #df[num_vars].isnull().sum() #print numerical attributes
    for var in num_vars:
       df[var]=df[var].fillna(df[var].median()) #filling missing values with median


    cat_vars=list(df.loc[:,df.dtypes =='object'].columns) #categorical valuelar icin list
    myimputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    myimputer = myimputer.fit(df[cat_vars])
    df[cat_vars] = myimputer.transform(df[cat_vars])

    #df[cat_vars].isnull().sum()

    df_cat = pd.get_dummies(df[cat_vars]) #converting cat values to numeric, 
                                      #we create dummy vars since pca applied on numeric
    df_all = [df[num_vars], df_cat]
    df_all = pd.concat(df_all, axis=1)
    return df_all    


dataset= pd.read_csv(r'C:\Users\berke\Desktop\mat.csv')
#dataset.info()
#print(dataset[['Clicked on Ad']].describe())
# score=dataset[['stroke']]
# score_matrix=score.values.reshape(-1,1)
# scaled=preprocessing.MinMaxScaler()
# scaled_score=scaled.fit_transform(score_matrix)
# plt.plot(scaled_score)


df = dataset.drop('age', 1) #goal scored column discarded if we use 0 it will discard row
dy = dataset['age'] #series of corresponding labels
df_all = fill_missing(df)


#print(df_all)

X_train,X_test,y_train,y_test = train_test_split(df_all,dy,test_size=0.2,random_state=0)


#PCA performs best with normalized feature set to do so we perform standard scalar normalization our feature set
sc=StandardScaler() #scaled data has zero mean,unit variance
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print("mean of scaled data:")
print(X_test.mean())
print("std of scaled data:")
print(X_test.std())


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA #classes are well seperated
lda=LDA(n_components=2) #number of projections can only be 1
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)


# per_var1=np.round(lda.explained_variance_ratio_*100,decimals=1)
# labels1=[str(x) for x in range(1,len(per_var1)+1)]
# plt.bar(x=range(1,len(per_var1)+1),height=per_var1,tick_label=labels1)
# plt.ylabel("Percentage of variance")
# plt.xlabel("Number of LDA components")
# plt.title("Scree Plot")


print("Accuracy using Random Forest:") 
%time randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with LDA:") 
%time randomFOREST(X_train_lda,X_test_lda,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with LDA:") 
%time LogisticReg(X_train_lda,X_test_lda,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with LDA:") 
%time LinearReg(X_train_lda,X_test_lda,y_train,y_test)

#plt.plot(X_train_pca['pca1'], X_train_pca['pca2'], 'ro')
#plt.show()

#plot each class separately
#plt.plot(pca_df.loc[df['churn']==1, 'pca1'], pca_df.loc[df['churn']==1, 'pca2'], 'ro')
#plt.plot(pca_df.loc[df['churn']==0, 'pca1'], pca_df.loc[df['churn']==0, 'pca2'], 'bx')








