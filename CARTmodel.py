''' CART Model to predict a number of complaints in a month across a borough'''
 
# =============================================================================
# `heat/hot water` complaints 
# =============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


hhw = pd.read_csv('./results/nyc_hhw_complaints.csv', 
                  parse_dates = ['created_date'])

hhw['month'] = hhw.created_date.dt.month
hhw['year'] = hhw.created_date.dt.year

# =============================================================================
# Train, tets data sets to predict an average number of complaints 
# in a month across borough
# =============================================================================

from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeRegressor


Train = hhw.loc[hhw.created_date<'2019-01-01']
Test = hhw.loc[hhw.created_date>='2019-01-01']

Train = Train.groupby(['borough', 'year',
                       'month'])['complaints'].sum().reset_index()

Train = Train.groupby(['borough', 'month'])['complaints'].mean().reset_index()


Test = Test.groupby(['borough','year',
                     'month'])['complaints'].sum().reset_index()

Test = Test.groupby(['borough', 'month'])['complaints'].mean().reset_index()


X_train = Train[['borough', 'month']]
y_train = Train['complaints']

X_test = Test[['borough','month']]
y_test = Test['complaints']

# Features 
# convert the categorical varibale to binary variables
mapper = DataFrameMapper([
    ('month', None),
    ('borough', LabelBinarizer())
], df_out=True)

# preprocessing features data sets
Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

CARTmodel = DecisionTreeRegressor(max_depth=5)
CARTmodel.fit(Z_train,y_train)
CARTmodel.score(Z_train, y_train)

y_pred = CARTmodel.predict(Z_test)

RSS = ((y_test-y_pred)**2).sum()
TSS = ((y_train.mean()-y_test)**2).sum()

R2 = 1.0 - RSS/TSS

print("Model performance on test data R^2 = {}".format(R2))
    
print("Baseline model prediction {}".format(y_train.mean()))
print("RSS = {}".format(RSS))
print("TSS = {}".format(TSS))

width = 12
height = 5

fig = plt.figure(figsize=(width, height))
ax1 = fig.add_subplot(121)
sns.residplot(y_test, y_pred)
plt.ylabel("Residuals")

ax2 = fig.add_subplot(122)
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" )
sns.distplot(y_test, hist=False, color="r", label="Actual Value")
plt.title("Distribution Plot")

plt.savefig('./figs/y_pred_tree_distribition.pdf')
plt.show()
plt.close()

# =============================================================================
# Save a model as pickle object
# =============================================================================
import pickle

# save model
with open('./results/CARTmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# =============================================================================
# load a model    
# with open('.results/CARTmodel.pkl', 'rb') as f:
#     pipe = pickle.load(f)
# =============================================================================

# =============================================================================
# Plot the tree
# =============================================================================

# =============================================================================
# import graphviz
# import pydotplus
# from sklearn.tree import export_graphviz, plot_tree
# from IPython.display import Image  
# import os
# 
# #Tell Python where the graphviz package is load; then load it.
# os.environ["PATH"] += os.pathsep + 'C:\\Users\\Olena\\Miniconda3\\pkgs\\graphviz-2.38-hfd603c8_2\\Library\\bin\\graphviz'
# 
# dot_data = export_graphviz(CARTmodel, out_file=None, 
#                                 feature_names=Z_train.columns,  
#                                 class_names=y_train.unique()) #, rotate=True)
# # Draw graph
# graph = pydotplus.graph_from_dot_data(dot_data)  
# 
# # Show graph
# Image(graph.create_png())
# 
# =============================================================================
