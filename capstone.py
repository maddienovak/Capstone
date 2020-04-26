#PSU ARL 5 Capstone 

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

#reading in csv from dataset file
df = pd.read_csv(r"~/Desktop/capstone/dataset_2020.csv")
df.set_index('filename', inplace=True)

#Adding material type for model per requested
df.loc[(df['Process'] == '3D-Printing', 'Material_Type')] = 0 #ABS
df.loc[(df['Process'] == 'Machining', 'Material_Type')] = 1 #Aluminum
df.loc[(df['Process'] == 'Welding', 'Material_Type')] = 1
df.loc[(df['Process'] == 'Casting', 'Material_Type')] = 1 

#changing from float to int
#temp_data['Process_Type'] = temp_data['Process_Type'].apply(np.int64)
df['Material_Type'] = df['Material_Type'].apply(np.int64)   


#xgboost classifier 


X = df[[
    'length',
    'width', 
    'height',
    'linear_properties',
    'surface_properties',
    'volume_properties',
    'Geom_Vertex_Unknown'

]]
Y = df.Process

#sizing 
print(X.shape)
print(Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

X_test.set_index('filename', inplace=True)

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
output['Predicted Output'] = val
output['filename'] = X_test.index
output.reset_index(level=0, inplace=True)
df.reset_index(level=0, inplace=True)
output = output.merge(df, how='left', on='filename')
output.head()

##writing model outputs to csv 
output.to_csv(r"~/Desktop/capstone/pred_df.csv")

### CROSS VALIDATION
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = xgboost.XGBClassifier()
kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#cross validation is better with 10 splits than 5 

output.loc[(output['Predicted Output'] == '3D-Printing', 'Process_Type')] = 0
output.loc[(output['Predicted Output'] == 'Welding', 'Process_Type')] = 1
output.loc[(output['Predicted Output'] == 'Casting', 'Process_Type')] = 2
output.loc[(output['Predicted Output'] == 'Machining', 'Process_Type')] = 3

output['Process_Type'] = output['Process_Type'].apply(np.int64)

####OUTPUT ALL MANUFACTURING PLANS

 
for i in output['Process_Type']:
    if i == 0:
        three_d_plans.at[3,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_3D']))
        three_d_plans.at[7,'Time (min)'] = list(filter(lambda num: num != 0.0, output['total_3D_time']))
        three_d_plans.at[8,'Operation Description'] = list(filter(lambda num: num != 0.0, output['cost_3D']))
        three_d_plans.at[9,'Operation Description'] = list(filter(lambda num: num != 0.0, output['cost_3D']))
    elif i == 1:
        welding_plans.at[5,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_welding']))
        welding_plans.at[8,'Time (min)'] = list(filter(lambda num: num != 0.0, output['total_welding_time']))
        welding_plans.at[9,'Operation Description'] = list(filter(lambda num: num != 0.0, output['welding_cost']))
    elif i == 2:
        casting_plans.at[0,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_casting1']))
        casting_plans.at[19,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_casting20']))
        casting_plans.at[22,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_casting23']))
        casting_plans.at[23,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_casting24']))
        casting_plans.at[24,'Time (min)'] = list(filter(lambda num: num != 0.0, output['total_casting_time']))
        casting_plans.at[25,'Operation Description'] = list(filter(lambda num: num != 0.0, output['casting_cost']))
    elif i == 3: 
        machining_plans.at[4,'Time (min)'] = list(filter(lambda num: num != 0.0, output['time_machining']))
        machining_plans.at[6,'Time (min)'] = list(filter(lambda num: num != 0.0, output['total_machining_time']))
        machining_plans.at[7,'Operation Description'] = list(filter(lambda num: num != 0.0, output['machining_cost']))











