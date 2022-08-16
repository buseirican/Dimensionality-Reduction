import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import math
from sklearn.decomposition import KernelPCA 
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 

#pca is linear method we need kernel to handle nonlinear data

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

dataset= pd.read_csv(r'C:\Users\berke\Desktop\car.csv')
dataset.info()
df = dataset.drop('selling_price', 1) #feature sets into X
dy = dataset['selling_price'] #series of corresponding labels
df_all = fill_missing(df)


print(df_all)

X_train,X_test,y_train,y_test = train_test_split(df_all,dy,test_size=0.2,random_state=0)
#PCA performs best with normalized feature set to do so we perform standard scalar normalization our feature set
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


kernel_pca=KernelPCA(kernel="rbf",n_components=2,fit_inverse_transform=(True)) 
X_train_kpca=kernel_pca.fit_transform(X_train) 
X_test_kpca=kernel_pca.transform(X_test)

# X_new=kernel_pca.inverse_transform(X_train_kpca)
# plt.scatter(X_train[:,0],X_train[:,1],alpha=0.2)
# plt.scatter(X_new[:,0],X_new[:,1],alpha=0.8)

#plt.scatter(X_train_kpca,X_train,"kernel pca")

# per_var1=np.round(kernel_pca.explained_variance_ratio_*100,decimals=1)
# labels1=[str(x) for x in range(1,len(per_var1)+1)]
# plt.bar(x=range(1,len(per_var1)+1),height=per_var1,tick_label=labels1)
# plt.ylabel("Percentage of variance")
# plt.xlabel("Number of Principal components")
# plt.title("Scree Plot")

#plt.scatter(X_train_kpca,y_train,"Kernel PCA")

print("Accuracy using Random Forest:") 
%time randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with Kernel PCA:") 
%time randomFOREST(X_train_kpca,X_test_kpca,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with Kernel PCA:") 
%time LogisticReg(X_train_kpca,X_test_kpca,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with Kernel PCA:") 
%time LinearReg(X_train_kpca,X_test_kpca,y_train,y_test)




