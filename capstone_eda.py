###EDA

counts = temp_data['Process'].value_counts()
#--last year dataset is not evenly balanced with the processes


#visual heat-map
#1 2 3 4 5 7 8 9 13 26 27 28 29 30 
heatmap_data = temp_data.drop(axis=0, columns=['Name', 'Process','Material', 'Cost', 'Time_(hr)', 'Number_of_Steps'])
plt.figure(figsize=(50,20))
cor = temp_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()

### FEATURE IMPORTANCE 

print(xgb.feature_importances_)
for col in temp_data.columns: 
    print(col) 

#visualization for feature important 
from matplotlib import pyplot
from xgboost import plot_importance
pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()
#--more detailed chart
plot_importance(xgb)
pyplot.show()

#decision tree visual 
#conda install graphviz python-graphviz
from xgboost import plot_tree
model = XGBClassifier()
model.fit(X, Y)
plot_tree(model, num_trees=2)
plot_tree(model)
##
fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(model, num_trees=2, ax=ax)
plt.show()