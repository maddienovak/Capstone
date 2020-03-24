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

temp_data = pd.read_csv(r"~/Desktop/capstone/DatasetFinalNumbers.csv")

#converting string to numbers
#process str to int 
temp_data.loc[(temp_data['Process'] == '3D-Printing', 'Process_Type')] = 0 #3D-printing 
temp_data.loc[(temp_data['Process'] == 'Machining', 'Process_Type')] = 1 #Machining
temp_data.loc[(temp_data['Process'] == 'Welding', 'Process_Type')] = 2 #Welding
temp_data.loc[(temp_data['Process'] == 'Casting', 'Process_Type')] = 3 #3D-printing 


#will need to be changed for our database
#material for each process 
temp_data.loc[(temp_data['Process'] == '3D-Printing', 'Material_Type')] = 0 #ABS
temp_data.loc[(temp_data['Process'] == 'Machining', 'Material_Type')] = 1 #Aluminum
temp_data.loc[(temp_data['Process'] == 'Welding', 'Material_Type')] = 1
temp_data.loc[(temp_data['Process'] == 'Casting', 'Material_Type')] = 1 

#changing from float to int
temp_data['Process_Type'] = temp_data['Process_Type'].apply(np.int64)
temp_data['Material_Type'] = temp_data['Material_Type'].apply(np.int64)   

#removing time for now ... 
temp_data.drop(columns=['Time (hr)'])
temp_data.drop(columns=['Material'])

#randSamp = temp_data.sample(5000)
Y = np.array(temp_data[['Process_Type']])

X = temp_data[[  
    'Process',
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

#visual heat-map
#1 2 3 4 5 7 8 9 13 26 27 28 29 30 
heatmap_data = temp_data.drop(axis=0, columns=['Name', 'Process','Material', 'Cost', 'Time (hr)', 'Number of Steps'])
plt.figure(figsize=(50,20))
cor = temp_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()


#multinomial logistic regression 

#re-do 

X = temp_data.drop(axis=0, columns=['Name', 'Process','Material', 'Cost', 'Time (hr)', 'Number of Steps'])
Y = temp_data.Process

#sizing 
print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

#need to install xgboost privately 
#conda install -c anaconda py-xgboost -- on python interactive

#like gradient boosting but optimizied 
from xgboost import XGBClassifier
xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', random_state=42, eval_metric="auc", num_class=4)
xgb.fit(X_train,y_train)

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

# Use trained model to predict output of test dataset
val = xgb.predict(X_test)

lb = preprocessing.LabelBinarizer()
lb.fit(y_test)

y_test_lb = lb.transform(y_test)
val_lb = lb.transform(val)

roc_auc_score(y_test_lb, val_lb, average='macro')

output = pd.DataFrame()
output['Expected Output'] = y_test
output['Predicted Output'] = val
output.head()

### FEATURE IMPORTANCE 

print(xgb.feature_importances_)
for col in temp_data.columns: 
    print(col) 

#visualization

pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()

#more detailed chart
plot_importance(xgb)
pyplot.show()

### CROSS VALIDATION
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = xgboost.XGBClassifier()
kfold = KFold(n_splits=10, random_state=)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
