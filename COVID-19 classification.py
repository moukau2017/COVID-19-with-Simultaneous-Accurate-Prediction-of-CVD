# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:19:35 2023

@author: mmoit
"""
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg19 import VGG19
#from keras.applications.xception import Xception

print(os.listdir("CT-image-classification-3class/"))

SIZE = 256  
train_images = []
train_labels = [] 

#for directory_path in glob.glob("CT-image-classification-3class/Training-image/*"):
for directory_path in glob.glob("CT-image-classification-2class/Training-image/*"):
    label = directory_path.split("/")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
       
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = [] 

#for directory_path in glob.glob("CT-image-classification-3class/Testing-image/*"):
for directory_path in glob.glob("CT-image-classification-2class/Testing-image/*"):
    fruit_label = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

                
test_images = np.array(test_images)
test_labels = np.array(test_labels)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

x_train, x_test = x_train / 255.0, x_test / 255.0




# VGG19 Model

vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))


for layer in vgg_model.layers:
	layer.trainable = False
    
vgg_model.summary()  

feature_extractor=vgg_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features 
#Now using ML models

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the model on training data

#SVM
from sklearn.svm import LinearSVC
model= LinearSVC(max_iter = 100)
#from sklearn.svm import SVC
#model= SVC(kernal = 'rbf' , random_state= 4)
#logistic regression
from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()


model.fit(X_for_training, y_train) 

X_test_feature = vgg_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)
#X_test_features1 = X_test_feature.reshape(X_test_feature.shape[0], -1)

prediction1 = model.predict(X_test_features)
 
prediction = le.inverse_transform(prediction1)


from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction)
print(cm)
sns.heatmap(cm, annot=True)

#ns_probs = [0 for _ in range(len(testy))]

prediction2 = model.decision_function(X_test_feature)[:,1]
 
#prediction3 = le.inverse_transform(prediction1)

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(test_labels_encoded, prediction2)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

prediction2 = model.decision_function(X_test_features)[:,1]
#prediction2 = le.inverse_transform(prediction1)

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, roc_auc_score


fpr, tpr, thresholds = roc_curve(test_labels_encoded, prediction2)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()


#RANDOMFOREST-ROC
prediction3 = model.predict_proba(X_test_features)[:,1]
#prediction4 = prediction1[: , 1]

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(test_labels_encoded, prediction3)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

#############################################################################
from keras.applications.vgg16 import VGG16
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))


for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  



feature_extractor=VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features 

###########################################################################################

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 100, random_state = 42)


#SVM
from sklearn.svm import LinearSVC
#model= LinearSVC(max_iter = 100)



#logistic regression
from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()


#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()


############################################################################################

model.fit(X_for_training, y_train) 

X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

prediction1 = model.predict(X_test_features) 
prediction = le.inverse_transform(prediction1)


from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction)
print(cm)
sns.heatmap(cm, annot=True)

#################################################################################################
#SVM AND LOGISTIC REGRESSION ROC

prediction2 = model.decision_function(X_test_features)[:,1]
#prediction2 = le.inverse_transform(prediction1)

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(test_labels_encoded, prediction2)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()


incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(20,20))
plt.bar(np.arange(3), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(3), test_labels) 

##################################################################################################

#RANDOMFOREST-ROC
prediction3 = model.predict_proba(X_test_features)[:,1]
#prediction4 = prediction1[: , 1]

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(test_labels_encoded, prediction3)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

