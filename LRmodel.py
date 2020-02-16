''' Regression model to predict a number of complaints in a month across borough'''
 
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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

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

model = LinearRegression(normalize=True)
model.fit(Z_train, y_train)
model.score(Z_train, y_train)

y_pred = model.predict(Z_test)

RSS = ((y_test-y_pred)**2).sum()
TSS = ((y_train.mean()-y_test)**2).sum()

R2 = 1.0 - RSS/TSS

print("Model performance R^2 = {}".format(R2))
    
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

plt.savefig('./figs/y_pred_LR_distribition.pdf')
plt.show()
plt.close()

# =============================================================================
# Save a model as pickle object
# =============================================================================
import pickle

# save model
with open('./results/LRmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# =============================================================================
# load a model    
# with open('.results/LRmodel.pkl', 'rb') as f:
#     pipe = pickle.load(f)
# =============================================================================

