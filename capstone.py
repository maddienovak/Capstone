#PSU ARL 5 Capstone 

#input parameter from casting process plan -- in order to have sum and cost ???



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import sklearn 

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

temp_data = pd.read_csv(r"~/Desktop/DatasetFinalNumbers.csv")

#converting string to numbers

temp_data.loc[(temp_data['Process'] == '3D-Printing', 'Process_Type')] = 0 #3D-printing 
temp_data.loc[(temp_data['Process'] == 'Machining', 'Process_Type')] = 1 #Machining
temp_data.loc[(temp_data['Process'] == 'Welding', 'Process_Type')] = 2 #Welding
temp_data.loc[(temp_data['Process'] == 'Casting', 'Process_Type')] = 3 #3D-printing 

#changing from float to int
temp_data['Process_Type'] = temp_data['Process_Type'].apply(np.int64)
    

#removing time for now ... 
temp_data.drop(columns=['Time (hr)'])


#randSamp = temp_data.sample(5000)
Y = np.array(temp_data[['Process_Type']])

X = temp_data[[  
    'advanced_face', 
    'axis2_placement_3d', 
    'cartesian_point', 
    'circle', 
    'closed_shell',
    'cylindrical_surface',
    'direction',
    'edge_curve',
    'edge_loop', 
    'face_bound', 
    'face_outer_bound', 
    'line',  
    'oriented_edge',
    'plane', 
    'vector', 
    'vertex_point',
    'height',
    'edge thickness',
    'number of ribs',
    'cutting depth',
    'number of drills',
    'diameter of drills',
    'rib thickness',
    'rib length',
    'shape thickness',
    'number of spines',
    'degree of draft for spines',
    'spine height'

    #wants material to be a feature 

]]

X = np.array(X)


#logistic regression (test)

#spliting data into test and train 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

lr = LogisticRegression(random_state = 0)
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
Y_test['predicted'] = Y_pred
df_out = pd.merge(temp_data,Y_test[['predicted']],how = 'left',left_index = True, right_index = True)

#determining if model was relevant 
print("Accuracy:",metrics.accuracy_score(Y_test.astype(int), Y_pred.astype(int)))
print(confusion_matrix(Y_test,Y_pred))  
print(classification_report(Y_test,Y_pred))  
print(accuracy_score(Y_test, Y_pred))

###feature importance 

#visual heat-map
plt.figure(figsize=(25,5))
cor = temp_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor['Process_Type'])
relevant_features = cor_target[cor_target>0.5]
relevant_features

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
nof_list = np.arange(1,28)
high_score = 0 

nof = 0 
score_list = []
for n in range(len(nof_list)):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    model = LogisticRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,Y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,Y_train)
    score = model.score(X_test_rfe,Y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))



cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,Y)  
#Fitting the data to model
model.fit(X_rfe,Y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

#good baseline model


######


#using random forest 

rf = RandomForestClassifier(random_state = 0, n_estimators = 500, min_samples_split = 10, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 40, bootstrap = True)
rf.fit(X_train, Y_train)
#prediction
Y_pred = rf.predict(X_test)

print(rf.feature_importances_) 
auc = metrics.roc_auc_score(Y_test, Y_pred)
print(auc)
print("Accuracy:",metrics.accuracy_score(Y_test.astype(int), Y_pred.astype(int)))
print(confusion_matrix(Y_test,Y_pred))  
print(classification_report(Y_test,Y_pred))  
print(accuracy_score(Y_test, Y_pred))


#%%
Y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_proba)
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)

plt.plot(fpr, tpr, label = "Autopay Variables, auc = " + str(auc))
plt.legend(loc = 4)
plt.show()
