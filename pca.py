import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pylab import *
from sklearn.ensemble import RandomForestClassifier 
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import mean_squared_error

def randomFOREST(X_train,X_test,y_train,y_test):
    regressor=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
    regressor.fit(X_train,y_train)
    print(regressor.score(X_test,y_test)) #finding accuracy using sum of square error
    return regressor.score(X_test,y_test)
    
    
def LogisticReg(X_train,X_test,y_train,y_test):
    logreg=LogisticRegression(solver="lbfgs",multi_class='auto',max_iter=1000)
    logreg.fit(X_train,y_train)
    print(logreg.score(X_test,y_test)) #finding accuracy using sum of square error
    return logreg.score(X_test,y_test)
    
def LinearReg(X_train,X_test,y_train,y_test):
    linreg=LinearRegression()
    linreg.fit(X_train,y_train) 
    print(linreg.score(X_test,y_test)) #finding accuracy using sum of square error
    return linreg.score(X_test,y_test)
   
def fill_missing(df):
    num_vars=list(df.loc[:,df.dtypes!='object'].columns) #creating a list for numeric attributes
    #print(df[num_vars].isnull().sum()) #print numerical attributes
    for var in num_vars:
       df[var]=df[var].fillna(df[var].median()) #filling missing values with median

    cat_vars=list(df.loc[:,df.dtypes =='object'].columns) #list for categorical values
    print(cat_vars)
    myimputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') #filling missing categorical values with most frequent value of that feature
    myimputer = myimputer.fit(df[cat_vars])
    df[cat_vars] = myimputer.transform(df[cat_vars])


    df_cat = pd.get_dummies(df[cat_vars]) #converting cat values to numeric, 
                                          #we create dummy vars since pca applied on numeric values
    df_all = [df[num_vars], df_cat]
    df_all = pd.concat(df_all, axis=1)
    return df_all



dataset= pd.read_csv(r'C:\Users\berke\Desktop\mat.csv') #reading csv file into dataframe

# dataset["Goal Scored"].hist()
# plt.title("Histogram of Goal Scored")
# plt.xlabel("Goal Scored")
# plt.ylabel("Frequency")

df = dataset.drop('age', 1) #Goal Scored column dropped for train set
dy = dataset['age'] #creating test set and put Goal Score column into it
df_all = fill_missing(df) #function call to fill missing values

print("Variences")
print(dataset.var())#variences shown
print("***DESCRIBING DATASET***")
print(dataset.describe())
print("**MISSING VALUES OF DATASET**")
print(dataset.isnull().sum()) #missing values shown

print("****DATASET****")
print(dataset)
print("**SHAPE OF DATASET**")
print(dataset.shape)
print("Correlation between age and other features:")
print(dataset[dataset.columns[1:]].corr()["Goal Scored"][:]) #correlations shown

#we split our data into train and test set with percentage of %20for testing %80for training
#train dataset is for fitting the model and test dataset is for evaluating the model
X_train,X_test,y_train,y_test = train_test_split(df_all,dy,test_size=0.2,random_state=0)

#PCA performs best with normalized feature set to do so we perform standard scalar normalization our feature set
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA #unsupervised
pca=PCA(n_components=2)  #creating PCA object and selecting number of components for reduction
X_train_pca=pca.fit_transform(X_train) 
X_test_pca=pca.transform(X_test)

#plot for visualizing the reduction after applying pca
X_new=pca.inverse_transform(X_train_pca)
plt.scatter(X_train[:,0],X_train[:,1],alpha=0.2)
plt.scatter(X_new[:,0],X_new[:,1],alpha=0.8)
plt.title("AFTER APPLYING PCA")

#plot for visualize the variances for each principal component
per_var1=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels1=[str(x) for x in range(1,len(per_var1)+1)]
plt.bar(x=range(1,len(per_var1)+1),height=per_var1,tick_label=labels1)
plt.ylabel("Percentage of variance")
plt.xlabel("Number of Principal components")
plt.title("Scree Plot for Dataset-2")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA #classes are well seperated
lda=LDA(n_components=1)  #creating LDA object and selecting number components
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)


from sklearn.decomposition import KernelPCA 
kernel_pca=KernelPCA(kernel="rbf",n_components=2,fit_inverse_transform=True) #creating KERNEL PCA object
X_train_kpca=kernel_pca.fit_transform(X_train)
X_test_kpca=kernel_pca.transform(X_test)


from sklearn.manifold import TSNE
tsne= TSNE(n_components=2, random_state=1, n_iter=250, learning_rate=200.0, verbose=1, perplexity=40, metric='euclidean', init='random')
X_train_tsne=tsne.fit_transform(X_train)
X_test_tsne=tsne.fit_transform(X_test)

from sklearn.manifold import Isomap
ısomap= Isomap(n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='minkowski', p=2, metric_params=None)
X_train_ısomap=ısomap.fit_transform(X_train)
X_test_ısomap=ısomap.fit_transform(X_test)

from sklearn.manifold import LocallyLinearEmbedding
LLE= LocallyLinearEmbedding(n_neighbors=5, n_components=2, reg=0.001, eigen_solver='auto', max_iter=100, method='standard', neighbors_algorithm='auto', random_state=None, n_jobs=None)
X_train_LLE=LLE.fit_transform(X_train)
X_test_LLE=LLE.fit_transform(X_test)

from sklearn.decomposition import TruncatedSVD
svd= TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
X_train_svd=svd.fit_transform(X_train)
X_test_svd=svd.fit_transform(X_test)


print("Accuracy using Random Forest:") 
%time rf=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with PCA:") 
%time rf_pca= randomFOREST(X_train_pca,X_test_pca,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with PCA:") 
%time lg_pca=LogisticReg(X_train_pca,X_test_pca,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with PCA:") 
%time ling_pca=LinearReg(X_train_pca,X_test_pca,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf2=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with LDA:") 
%time rf_lda=randomFOREST(X_train_lda,X_test_lda,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg2=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with LDA:") 
%time lg_lda=LogisticReg(X_train_lda,X_test_lda,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling2=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with LDA:") 
%time ling_lda=LinearReg(X_train_lda,X_test_lda,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf3=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with Kernel PCA:") 
%time rf_kpca=randomFOREST(X_train_kpca,X_test_kpca,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg3=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with Kernel PCA:") 
%time lg_kpca=LogisticReg(X_train_kpca,X_test_kpca,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling3=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with Kernel PCA:") 
%time ling_kpca=LinearReg(X_train_kpca,X_test_kpca,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf4=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with LLE:") 
%time rf_lle=randomFOREST(X_train_LLE,X_test_LLE,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg4=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with LLE:") 
%time lg_lle=LogisticReg(X_train_LLE,X_test_LLE,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling4=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with LLE:") 
%time ling_lle=randomFOREST(X_train_LLE,X_test_LLE,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf5=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with ISOMAP:") 
%time rf_isomap=randomFOREST(X_train_ısomap,X_test_ısomap,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg5=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with ISOMAP:") 
%time lg_isomap=LogisticReg(X_train_ısomap,X_test_ısomap,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling5=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with ISOMAP:") 
%time ling_isomap=LinearReg(X_train_ısomap,X_test_ısomap,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf6=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with TSNE:") 
%time rf_tsne=randomFOREST(X_train_tsne,X_test_tsne,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg6=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with TSNE:") 
%time lg_tsne=LogisticReg(X_train_tsne,X_test_tsne,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling6=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with TSNE:") 
%time ling_tsne=LinearReg(X_train_tsne,X_test_tsne,y_train,y_test)


print("Accuracy using Random Forest:") 
%time rf7=randomFOREST(X_train,X_test,y_train,y_test)
print("Accuracy using Random Forest with SVD:") 
%time rf_svd=randomFOREST(X_train_svd,X_test_svd,y_train,y_test)
print("Accuracy using Logistic Regression:") 
%time lg7=LogisticReg(X_train,X_test,y_train,y_test)
print("Accuracy using Logistic Regression with SVD:") 
%time lg_svd=LogisticReg(X_train_svd,X_test_svd,y_train,y_test)
print("Accuracy using Linear Regression:") 
%time ling7=LinearReg(X_train,X_test,y_train,y_test)
print("Accuracy using Linear Regression with SVD:") 
%time ling_svd=LinearReg(X_train_svd,X_test_svd,y_train,y_test)


#randomforest plot
left=[1,2,3,4,5,6,7,8]
height=[rf,rf_pca,rf_lda,rf_kpca,rf_lle,rf_isomap,rf_tsne,rf_svd]
tick_label=["RF","PCA","LDA","KPCA","LLE","ISOMAP","TSNE","SVD"]
plt.bar(left,height,tick_label=tick_label,width=0.8,color="red")
plt.title("Accuracy Comparison for Random Forest")
plt.ylabel("Accuracy")
plt.xlabel("Dimensionality Reduction Methods")

#logistic regression plot
left=[1,2,3,4,5,6,7,8]
height=[lg,lg_pca,lg_lda,lg_kpca,lg_lle,lg_isomap,lg_tsne,lg_svd]
tick_label=["LG","PCA","LDA","KPCA","LLE","ISOMAP","TSNE","SVD"]
plt.bar(left,height,tick_label=tick_label,width=0.8,color="red")
plt.title("Accuracy Comparison for Logistic Regression")
plt.ylabel("Accuracy")
plt.xlabel("Dimensionality Reduction Methods")

#linear regression plot
left=[1,2,3,4,5,6,7,8]
height=[ling,ling_pca,ling_lda,ling_kpca,ling_lle,ling_isomap,ling_tsne,ling_svd]
tick_label=["Linear","PCA","LDA","KPCA","LLE","ISOMAP","TSNE","SVD"]
plt.bar(left,height,tick_label=tick_label,width=0.8,color="red")
plt.title("Accuracy Comparison for Linear Regression")
plt.ylabel("Accuracy")
plt.xlabel("Dimensionality Reduction Methods")







