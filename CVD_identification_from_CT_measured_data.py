

import pandas as pd
#from matplotlib import pyplot as plt
import seaborn as sns
 
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout

import autokeras as ak

df = pd.read_csv("FINAL-abnormal-normal-CTR-PA-A-new dataset-for-CLAF-2nd.csv")

#print(df.describe().T)  #Values need to be normalized before fitting. 
#print(df.isnull().sum())
#df = df.dropna(axis=1)

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'RT-PCR result':'Label'})
print(df.dtypes)
df.head()

#Understand the data 
sns.countplot(x="Label", data=df) #M - malignant   B - benign


####### Replace categorical values with numbers########
df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
X = df.drop(labels = ["Label"], axis=1) 
##########################################################

from sklearn.model_selection import KFold
# define the search

from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf')
model_SVC.fit(X, y)
kfold_validation=KFold(10)
from sklearn.model_selection import cross_val_score
results=cross_val_score(model_SVC,X,y,cv=kfold_validation)
print(results)
import numpy as np
print(np.mean(results))


#**************************************************************
import pandas as pd
#from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold 
import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
import numpy as np # linear algebra
import pandas as pd 


df = pd.read_csv("FINAL-abnormal-normal-CTR-PA-A-new dataset-for-CLAF-2nd.csv")

#print(df.describe().T)  #Values need to be normalized before fitting. 
#print(df.isnull().sum())
#df = df.dropna(axis=1)

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'RT-PCR result':'Label'})
print(df.dtypes)
df.head()

 
sns.countplot(x="Label", data=df) #M - malignant   B - benign


####### Replace categorical values with numbers########
df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values
#################################################################
#Define x and normalize values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label"], axis=1) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X1 = scaler.transform(X)

#from sklearn.svm import SVC
#model_SVC = SVC(kernel = 'rbf', random_state = 0)
#model_SVC = SVC(kernel = 'rbf')
#model_SVC.fit(X_train, y_train)


random_state = np.random.RandomState(0)
clf = RandomForestClassifier(random_state=random_state)
cv = StratifiedKFold(n_splits=10,shuffle=False)
#cv = KFold(n_splits=10,shuffle=False)


tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
for train,test in cv.split(X,y):
    prediction = clf.fit(X1[train],y[train]).predict_proba(X1[test])
    fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

print(aucs)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()




