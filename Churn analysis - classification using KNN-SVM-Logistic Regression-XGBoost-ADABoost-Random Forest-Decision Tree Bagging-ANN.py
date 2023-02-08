#This analysis is based on the following article:
#https://github.com/rmpbastos/data_science/blob/master/Churn_Prediction.ipynb


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras

# set default matplotlib parameters
COLOR = '#ababab'
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['grid.color'] = COLOR
mpl.rcParams['grid.alpha'] = 0.1

DATA_PATH = "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

df.head()

print(f"Rows: ", df.shape[0])
print(f"Columns: ", df.shape[1])

df.customerID.unique().shape
sum(df['customerID'].value_counts(dropna = False))

df.info()

df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = df['TotalCharges'].astype(float)

len(df[df['TotalCharges'] == ' ']) #the previous command gave error because there are empty spaces. Let's see how many there are, 11

df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = np.nan #we first put nan values in the empry spaces

len(df[df['TotalCharges'] == ' '])

# checking null value
df.isnull().sum()

# replace null values by the median of 'TotalCharges'
TotalCharges_median = df['TotalCharges'].median()
df['TotalCharges'].fillna(TotalCharges_median, inplace=True)

#Now it is possible to convert TotalCharges into float
df['TotalCharges'] = df['TotalCharges'].astype(float)

df.describe()#Some descriptive statistics

#Let's see the distributions of the contiuous variables and outliers.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'Monthly Charges' and 'Total Charges'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(df['MonthlyCharges'], orient='v', color='#488ab5', ax=ax[0], 
            boxprops=boxprops, 
            whiskerprops=whiskerprops, 
            capprops=capprops, 
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([20, 70, 120])
sns.boxplot(df['TotalCharges'], orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops, 
            whiskerprops=whiskerprops, 
            capprops=capprops, 
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 4000, 8000])
plt.tight_layout(pad=4.0);

sns.distplot(df['MonthlyCharges'], hist=True, kde=True)

sns.distplot(df['TotalCharges'], hist=True, kde=True)

print(df['Churn'].value_counts())
print('\nTotal Churn Rate: {:.2%}'.format(df[df['Churn'] == 'Yes'].shape[0] / df.shape[0]))

plt.hist(df['Churn'], color='#488ab5')

# unique values for each column containing a categorical feature
def unique_values():
  cat_columns = np.unique(df.select_dtypes('object').columns)
  for i in cat_columns:
    print(i, df[i].unique())

unique_values()
#Some of the columns with 3 unique values could be treated as binary. To illustrate, the columns StreamingTV and
#TechSupport have values "No", "Yes", and "No internet service". In these cases, "No internet service" could be 
#considered as "No".

# switch 'No inernet service to 'No'
to_binary = ['DeviceProtection', 'OnlineBackup', 'OnlineSecurity', 'StreamingMovies', 'StreamingTV', 'TechSupport']

for i in to_binary:
  df.loc[df[i].isin(['No internet service']), i] = 'No'
  
unique_values()

#First, we need to convert Churn values to numerical.
df.loc[df['Churn'] == 'No','Churn'] = 0 
df.loc[df['Churn'] == 'Yes','Churn'] = 1
df['Churn'] = df['Churn'].astype(int)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
sns.barplot(x = df['gender'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[0][0])
ax[0][0].set_facecolor('#f5f5f5')
ax[0][0].set_ylim(0,1)
ax[0][0].set_xlabel(None)
ax[0][0].set_title('Gender')
sns.barplot(x = df['Dependents'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[0][1])
ax[0][1].set_facecolor('#f5f5f5')
ax[0][1].tick_params(labelleft=False)
ax[0][1].set_ylim(0,1)
ax[0][1].set_ylabel(None)
ax[0][1].set_xlabel(None)
ax[0][1].set_title('Dependents')
sns.barplot(x = df['InternetService'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[0][2])
ax[0][2].set_facecolor('#f5f5f5')
ax[0][2].tick_params(labelleft=False)
ax[0][2].set_ylim(0,1)
ax[0][2].set_ylabel(None)
ax[0][2].set_xlabel(None)
ax[0][2].set_title('Internet Service')
sns.barplot(x = df['DeviceProtection'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[1][0])
ax[1][0].set_facecolor('#f5f5f5')
ax[1][0].set_ylim(0,1)
ax[1][0].set_xlabel(None)
ax[1][0].set_title('Device Protection')
sns.barplot(x = df['OnlineSecurity'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[1][1])
ax[1][1].set_facecolor('#f5f5f5')
ax[1][1].tick_params(labelleft=False)
ax[1][1].set_ylim(0,1)
ax[1][1].set_ylabel(None)
ax[1][1].set_xlabel(None)
ax[1][1].set_title('Online Security')
sns.barplot(x = df['Contract'], y = df['Churn'], color='#488ab5', errorbar=None, ax=ax[1][2])
ax[1][2].set_facecolor('#f5f5f5')
ax[1][2].tick_params(labelleft=False)
ax[1][2].set_ylim(0,1)
ax[1][2].set_ylabel(None)
ax[1][2].set_xlabel(None)
ax[1][2].set_title('Contract')
plt.tight_layout(pad=4.0)

#Considering that most Machine Learning algorithms work better with numerical inputs, we'll preprocess our data 
#using the following techniques to convert categorical features into numerical values
# list of binary variables, except 'Churn'
bin_var = [col for col in df.columns if len(df[col].unique()) == 2 and col != 'Churn']
# list of categorical variables
cat_var = [col for col in df.select_dtypes(['object']).columns.tolist() if col not in bin_var]
# apply Label Encoding for binaries
le = LabelEncoder()
for col in bin_var:
  df[col] = le.fit_transform(df[col])
# apply get_dummies for categorical
df = pd.get_dummies(df, columns=cat_var)

df.head()

#Creating training and test sets
# feature matrix
X = df.drop('Churn', axis=1)
# target vector
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 25)

#Remember that we are dealing with an unbalanced dataset, as we determined a few steps above
#To manage the situation, we'll standardize the features of the training set using StardardScaler and then apply
#RandomUnderSampler, which is a "way to balance the data by randomly selecting a subset of data for the targeted
#classes"
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()

#----------------------------------------------------------------------------------------------------------------
model = []
cross_val = []
recall = []
for i in (svc, lr, xgb):
  model.append(i.__class__.__name__)
  cross_val.append(cross_validate(i, X_train_rus, y_train_rus, scoring='recall'))
  
for d in range(len(cross_val)):
  recall.append(cross_val[d]['test_score'].mean())
  
model_recall = pd.DataFrame
pd.DataFrame(data = recall, index = model, columns = ['Recall'])
#-----------------------------------------------------------------------------------------------------------------

# Tuning KNN hyperparameter
param_grid = {'n_neighbors':range(1, 100)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')
KNN_best_result = grid_result.best_score_
KNN_best_param = grid_result.best_params_

KNN_best_param



# Tuning SVM hyperparameters
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(svc, param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')
SVM_best_result = grid_result.best_score_
SVM_best_param = grid_result.best_params_

SVM_best_param



# Tuning Logistic Regression hyperparameters
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(lr, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')
LR_best_result = grid_result.best_score_
LR_best_param = grid_result.best_params_

LR_best_param



#determine the optimal number of trees in the XGBoost model, searching over values for the n_estimators argument
param_grid = {'n_estimators': range(0, 1000, 25)}
grid_search = GridSearchCV(xgb, param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

#earch over other two relevant parameters, max_depth, and min_child_weight
xgb = XGBClassifier(n_estimators = 25)
param_grid = {'max_depth': range(1,8,1),
              'min_child_weight': np.arange(0.0001, 0.5, 0.001)}
grid_search = GridSearchCV(xgb, param_grid, scoring = 'recall', n_jobs = -1)
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

#determine the best value for gamma, an important parameter used to control the model's tendency to overfit.
xgb = XGBClassifier(n_estimators = 25, max_depth = 1, min_child_weight = 0.0001)
param_grid = {'gamma': np.arange(0.0, 20.0, 0.05)}
grid_search = GridSearchCV(xgb, param_grid, scoring = 'recall', n_jobs = -1)
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_search.best_params_}')

#Finally, search for the optimal learning_rate value
xgb = XGBClassifier(n_estimators=25, max_depth = 1, min_child_weight = 0.0001, gamma = 11.95)
param_grid = {'learning_rate': [0.0001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(xgb, param_grid, scoring='recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_search.best_score_} for {grid_search.best_params_}')
XGB_best_result = grid_search.best_score_
XGB_best_param = grid_search.best_params_

XGB_best_param




#ADABoost
param_grid = {'max_depth':range(1, 5), 'max_leaf_nodes':[500,2000,8000,99999]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_search.best_score_} for {grid_search.best_params_}')

param_grid = {'n_estimators': range(0, 1000, 25)}
grid_search = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, max_leaf_nodes = 500)),
                           param_grid,
                           scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')
ADA_best_result = grid_search.best_score_
ADA_best_param = grid_search.best_params_

ADA_best_param




#Random Forest
param_grid = {'n_estimators':range(0, 1000, 25)}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

RF = RandomForestClassifier(n_estimators = 175)
param_grid = {'max_leaf_nodes':[500,2000,8000,99999]}
grid_search = GridSearchCV(RF, param_grid, scoring = 'recall', n_jobs = -1)#n_jobs = -1 use all CPU cores
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

RF_best_result = grid_result.best_score_
RF_best_param = grid_result.best_params_

RF_best_param



#Decision tree Bagging
#Setting values for the parameters
DT_bag = BaggingClassifier(DecisionTreeClassifier(), bootstrap = True, n_jobs = -1, random_state = 25)
param_grid = {'n_estimators': range(0, 1000, 25)}
grid_search = GridSearchCV(DT_bag, param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

DT_bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 325, bootstrap = True, n_jobs = -1, random_state = 25)
param_grid = {'max_samples': [5, 10, 25, 50, 100],
              'max_features': [5, 10, 25, 50, 100]}
grid_search = GridSearchCV(DT_bag, param_grid, scoring = 'recall')
grid_result = grid_search.fit(X_train_rus, y_train_rus)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

DT_bag_best_result = grid_result.best_score_
DT_bag_best_param = grid_result.best_params_

DT_bag_best_param



pd.DataFrame({'Model':['KNN', 'SVM', 'LR', 'XGB', 'ADABoost', 'Random Forest', 'Decision Tree Bagging'],
              'Recall':[KNN_best_result, SVM_best_result, LR_best_result, XGB_best_result, ADA_best_result, RF_best_result, DT_bag_best_result]})




#Let's visualise their performance on the test set
#final KNN model
KNN = KNeighborsClassifier(n_neighbors = 59)
KNN.fit(X_train_rus, y_train_rus)

# prediction
X_test_KNN = scaler.transform(X_test)
y_pred_KNN = KNN.predict(X_test_KNN)

# classification report
print(classification_report(y_test, y_pred_KNN))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_KNN, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# final SVC model
svc = SVC(kernel = 'poly', C = 0.0001)
svc.fit(X_train_rus, y_train_rus)

# prediction
X_test_svc = scaler.transform(X_test)
y_pred_svc = svc.predict(X_test_svc)

# classification report
print(classification_report(y_test, y_pred_svc))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_svc, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# final Logistic Regression model
lr = LogisticRegression(solver = 'liblinear', C = 0.0001)
lr.fit(X_train_rus, y_train_rus)

# prediction
X_test_lr = scaler.transform(X_test)
y_pred_lr = lr.predict(X_test_lr)

# classification report
print(classification_report(y_test, y_pred_lr))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_lr, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# final XGBoost model
xgb = XGBClassifier(learning_rate = 0.0001, n_estimators = 25, max_depth = 1, min_child_weight = 0.0001, gamma = 11.95)
xgb.fit(X_train_rus, y_train_rus)

# prediction
X_test_xgb = scaler.transform(X_test)
y_pred_xgb = xgb.predict(X_test_xgb)

# classification report
print(classification_report(y_test, y_pred_xgb))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_xgb, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# Final ADABoost model
ADA = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, max_leaf_nodes = 500), n_estimators = 50)
ADA.fit(X_train_rus, y_train_rus)

# prediction
X_test_ADA = scaler.transform(X_test)
y_pred_ADA = ADA.predict(X_test_ADA)

# classification report
print(classification_report(y_test, y_pred_ADA))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_ADA, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# Final Random Forest model
RF = RandomForestClassifier(n_estimators = 750, max_leaf_nodes = 500)
RF.fit(X_train_rus, y_train_rus)

# prediction
X_test_RF = scaler.transform(X_test)
y_pred_RF = RF.predict(X_test_RF)

# classification report
print(classification_report(y_test, y_pred_RF))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_RF, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()



# final DTBagging model
DT_bag = BaggingClassifier(DecisionTreeClassifier(), 
                           n_estimators = 325, 
                           max_features = 10, 
                           max_samples = 5, 
                           bootstrap = True, 
                           n_jobs = -1, 
                           random_state = 25)

DT_bag.fit(X_train_rus, y_train_rus)

# prediction
X_test_DT_bag = scaler.transform(X_test)
y_pred_DT_bag = DT_bag.predict(X_test_DT_bag)

# classification report
print(classification_report(y_test, y_pred_DT_bag))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_DT_bag, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()






#ANN the results are not prefect because the model needs to be tuned, and that requires time.
X_test = scaler.transform(X_test)

model = keras.Sequential([
    # input layer
    keras.layers.Dense(40, input_shape = (28,), activation = 'relu'),
    keras.layers.Dense(20, activation = 'relu'),
    keras.layers.Dense(10,activation = 'relu'),
    # we use sigmoid for binary output
    # output layer
    keras.layers.Dense(1, activation = 'sigmoid')])

optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

model.compile(optimizer = optimizer,
              loss = keras.losses.BinaryCrossentropy(from_logits = False),#from_logit = false, use probabilities [0,1]
              metrics = [keras.metrics.Recall()])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience  = 3)]

history = model.fit(X_train, y_train, epochs = 100, callbacks = callbacks, validation_split = 0.2)

plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.title('Training and Validation loss - Adam, lr = 0.001')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='validation Recall')
plt.title('Training and Validation Recall - Adam, lr = 0.001')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()






model.evaluate(X_test, y_test)

y_pred_ANN = model.predict(X_test).round()
print(y_pred_ANN)

# classification report
print(classification_report(y_test, y_pred_ANN))

# confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_ANN, normalize = 'true'), annot = True, ax = ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('Real Value')
ax.set_xlabel('Predicted Value')
plt.show()





recall_KNN = recall_score(y_test, y_pred_KNN)
recall_svc = recall_score(y_test, y_pred_svc)
recall_lr = recall_score(y_test, y_pred_lr)
recall_xgb = recall_score(y_test, y_pred_xgb)
recall_ADA = recall_score(y_test, y_pred_ADA)
recall_RF = recall_score(y_test, y_pred_RF)
recall_DT_bag = recall_score(y_test, y_pred_DT_bag)
df = pd.DataFrame({'Model':['KNN', 'SVM', 'LR', 'XGB', 'ADABoost', 'Random Forest', 'Decision Tree Bagging'],
              'Recall':[recall_KNN, recall_svc, recall_lr, recall_xgb, recall_ADA, recall_RF, recall_DT_bag]})
df = df.sort_values('Recall', ascending = False)
df













