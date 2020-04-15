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
temp_data.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)

temp_data.reset_index(level=0, inplace=True)

#converting string to numbers
#process str to int 
#temp_data.loc[(temp_data['Process'] == '3D-Printing', 'Process_Type')] = 0 #3D-printing 
##temp_data.loc[(temp_data['Process'] == 'Welding', 'Process_Type')] = 2 #Welding
#temp_data.loc[(temp_data['Process'] == 'Casting', 'Process_Type')] = 3 #3D-printing 


#will need to be changed for our database
#material for each process 
#temp_data.loc[(temp_data['Process'] == '3D-Printing', 'Material_Type')] = 0 #ABS
#temp_data.loc[(temp_data['Process'] == 'Machining', 'Material_Type')] = 1 #Aluminum
#temp_data.loc[(temp_data['Process'] == 'Welding', 'Material_Type')] = 1
#temp_data.loc[(temp_data['Process'] == 'Casting', 'Material_Type')] = 1 

#changing from float to int
#temp_data['Process_Type'] = temp_data['Process_Type'].apply(np.int64)
#emp_data['Material_Type'] = temp_data['Material_Type'].apply(np.int64)   

 
temp_data.drop(columns=['Material']

#xgboost classifier 
#re-do 

X = temp_data.drop(axis=0, columns=['Name', 'Process', 'closed_shell', 'Material', 'Cost', 'Time_(hr)', 'Number_of_Steps'])
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
output.reset_index(level=0, inplace=True)
output = output.merge(temp_data,on='index')
output = output.drop(columns=['Expected Output'])
output.head()


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

#cross validation with xgboost 
https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/

#helpful info for this type of classifier 
#https://towardsdatascience.com/a-step-by-step-guide-to-building-a-multiclass-classifier-for-breast-tissue-classification-5b685d765e97


################################################################
#code for retrieving manufacturing plans


#https://stackoverflow.com/questions/42102674/how-can-i-see-the-formulas-of-an-excel-spreadsheet-in-pandas-python
#https://stackoverflow.com/questions/57595318/pandas-retrieve-values-from-one-dataframe-and-do-calculation-on-another-datafram


###Manufacturing plans
three_d_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/3D_printing.csv")
machining_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/cnc_milling.csv")
welding_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/welding_plan.csv")
casting_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/casting_plan.csv")

####miller code csv

m_3 = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/3D_printing_results.csv")
m_casting = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/casting_results.csv")
m_machining = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/machining_results.csv")
m_welding = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/Welding_results.csv")


#data cleaning 

m_3.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_3["filename"] = m_3["filename"].str.replace("./step-files/", "")
m_3['Process']='3D-Printing'

##

m_casting.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

m_casting.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_casting["filename"] = m_casting["filename"].str.replace("./step-files/", "")

m_casting['Process']='Casting'
##

m_machining.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_machining["filename"] = m_machining["filename"].str.replace("./step-files/", "")

m_machining['Process']='Machining'

##
m_welding.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_welding["filename"] = m_welding["filename"].str.replace("./step-files/", "")

m_welding['Process']='Welding'

#appending into one with each process
x_feat = m_welding.append(m_machining, ignore_index=False, verify_integrity=False, sort=None)
x_feat = fin.append(m_casting, ignore_index=False, verify_integrity=False, sort=None)
x_feat = fin.append(m_3, ignore_index=False, verify_integrity=False, sort=None)


###3-D printing
#[4,5] formula 
def time_3D(row):
    for i in row:
        return row['volume_properties'] + (0.2 * 0.2 * 3.14 * 60)

x_feat['time_3D'] = x_feat.apply(time_3D, axis=1)

###Welding

def time_welding(row):
    for i in row:
        return row['length'] * 3

x_feat['time_welding'] = x_feat.apply(time_welding, axis=1)

###Casting

def time_casting20(row):
    c=2
    for i in row:
        return c * (row['volume_properties']/row['surface_properties'])**2

x_feat['time_casting20'] = x_feat.apply(time_casting20, axis=1)

def time_casting23(row):
    c=2
    for i in row:
        return c * (row['volume_properties']/row['surface_properties'])**2

x_feat['time_casting23'] = x_feat.apply(time_casting23, axis=1)


def time_casting24(row):
    for i in row:
        return row['surface_properties'] * 0.5

x_feat['time_casting24'] = x_feat.apply(time_casting24, axis=1)








for column in temp_data[['Name', 'Process']]:
    if 

for column in temp_data[['Name', 'Process']]:
    if temp_data['Process'].str.contains('3D-Printing'):
        temp_data['d'] = temp_data.apply(time_row4, axis=1)
    else:
        print('no')




plans = pd.DataFrame(output, columns = ['index', 'Predicted_Output', 'Name']) ###here i would also add the columns needed for formulas
plans.to_csv(r"~/Desktop/output.csv")

Casting = pd.DataFrame(plans[plans.Predicted_Output.str.contains('Casting',case=False)])
printing = pd.DataFrame(plans[plans.Predicted_Output.str.contains('3D-Printing',case=False)])
Welding = pd.DataFrame(plans[plans.Predicted_Output.str.contains('Welding',case=False)])
Machining = pd.DataFrame(plans[plans.Predicted_Output.str.contains('Machining',case=False)])

Casting_plan = pd.read_csv(r"~/Desktop/3D_plan.csv")
printing_plan = pd.read_csv(r"~/Desktop/3D_plan.csv")
Welding_plan = pd.read_csv(r"~/Desktop/3D_plan.csv")
Machining_plan = pd.read_csv(r"~/Desktop/3D_plan.csv")
