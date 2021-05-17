import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
#train test and validation seperation
train_A_file=pd.read_csv("A_train.csv",index_col=['no'])
train_B_file=pd.read_csv("B_train.csv",index_col=['no'])
test_B_file=pd.read_csv("B_test.csv",index_col=["no"])
test_B_file=test_B_file.reset_index()
test_B_file=test_B_file.drop(["no"],axis=1)
test_B_file=test_B_file.rename(columns={str(x):y for x,y in zip(test_B_file.columns,range(0,len(test_B_file.columns)))})
train_A_target=train_A_file["flag"]
train_B_target=train_B_file["flag"]
target=np.concatenate((train_A_target,train_B_target),axis=0)
X_A=train_A_file.drop(["flag"],axis=1)
X_B=train_B_file.drop(["flag"],axis=1)
newDataSet=np.concatenate((X_A,X_B),axis=0)
#file pre-processing
def var_summary(x):
    return pd.Series([x.count(),x.isnull(),x.sum(),x.mean(),x.std(),x.dropna().quantile(0.25),
                      x.dropna().qunatile(0.5),x.dropna().quantile(0.75)])
train_A_file.apply(lambda x:var_summary(x))
#drop line that are almost blank
blank_col=[]
for i in newDataSet.columns:
    if newDataSet[i].isnull().sum()>0.90*len(newDataSet[i]):
        blank_col.append(i)
newDataSet_dn=newDataSet.drop(blank_col,axis=1)
#drop the highly correlated variable
relation = newDataSet_fn.corr()
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols =[]
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i+1,length):
            if (relation.iloc[i,j] > 0.98) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])
newDataSet_hc=newDataSet_fn.drop(del_cols,axis=1)
for i in del_cols:
    if i in test_B_file_fn.columns:
        test_B_file_fn=test_B_file_fn.drop([i],axis=1)
#use heapmap to depict the correlation of data
sns.heapmap(newDataSet_hc)
#use boxplot to visualization the impact of each independent variable on dependent variable
figplot=PdfPages("Boxplot with independent variable.pdf")
for variable in newDataSet_hc:
    fig,axes=plt.subplots(figsize=(10,4))
    sns.boxplot(x='default',y=variable,data=train_A_file)
    figplot.savefig(fig)
figplot.close()
#rebalance the sample, as most people are not default, to improve the model efficiency
train_A_file_1=train_A_file[train_A_file["flag"]==1]
train_A_file_0=train_A_file[train_A_file["flag"]==0]
train_B_file_1=train_B_file[train_B_file["flag"]==1]
train_B_file_0=train_B_file[train_B_file["flag"]==0]
df_sample_a0=train_A_file_0.sample(train_A_file_1.shape[0])
train_A=train_A_file_1.append(df_sample_a0)
train_A=train_A.sample(frac=1)
df_sample_B0=train_B_file_0.sample(train_B_file_1.shape[0])
train_B=train_B_file_1.append(df_sample_B0)
train_B=train_B.sample(frac=1)
#use GridSearch for parameter selection
#Xgboost
param_test0 = {'learning_rate':[0.02,0.04, 0.06, 0.08, 0.1,0.2]}
gsearch_0 = GridSearchCV(estimator = xgb.XGBRegressor(random_state = 1),
                        param_grid = param_test0, cv = 3)
gsearch_0.fit(newDataSet_hc1,target1)
#lgBoost model
param_test1= {'learning_rate':[0.02,0.04, 0.06,0.08,0.1,0.2]}
gsearch = GridSearchCV(estimator = lgb.LGBMRegressor(random_state = 1),
                        param_grid = param_test1, cv = 3)
gsearch.fit(newDataSet_hc1,target1)
#random forest model
from sklearn.ensemble import RandomForestRegressor
param_test2= {"max_depth":[40,60,80,100,120,200]}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(random_state = 1),
                        param_grid = param_test2, cv = 3)
gsearch2.fit(newDataSet_hc1,target1)
#svm model
from sklearn import svm
classifier=svm.SVC(kernel="rbf")
classifier.fit(newDataSet_hc1,target1)
#model fitting
model_xbg=xgb.XGBRegressor(learning_rate=0.2)
model_xbg.fit(newDataSet_hc1,target1)
model_lgb=lgb.LGBMRegressor(learning_rate=0.04)
model_lgb.fit(newDataSet_hc1,target1)
#model stacking
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            X = X.reset_index(drop=True)
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models=StackingAveragedModels(base_models =(model_xbg,classifier),meta_model =model_lgb)
stacked_averaged_models.fit(newDataSet_hc1,target1)
prediction=stacked_averaged_models.predict(test_B_file)
#file loading
train_A_file=pd.read_csv("A_train.csv",index_col=['no'])
train_B_file=pd.read_csv("B_train.csv",index_col=['no'])
test_B_file=pd.read_csv("B_test.csv",index_col=["no"])
test_B_file=test_B_file.reset_index()
test_B_file=test_B_file.drop(["no"],axis=1)
test_B_file=test_B_file.rename(columns={str(x):y for x,y in zip(test_B_file.columns,range(0,len(test_B_file.columns)))})
train_A_target=train_A_file["flag"]
train_B_target=train_B_file["flag"]
target=np.concatenate((train_A_target,train_B_target),axis=0)
X_A=train_A_file.drop(["flag"],axis=1)
X_B=train_B_file.drop(["flag"],axis=1)
newDataSet=np.concatenate((X_A,X_B),axis=0)
#drop line that are almost blank
blank_col=[]
for i in newDataSet.columns:
    if newDataSet[i].isnull().sum()>0.90*len(newDataSet[i]):
        blank_col.append(i)
newDataSet_dn=newDataSet.drop(blank_col,axis=1)
#drop the highly correlated variable
relation = newDataSet_fn.corr()
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols =[]
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i+1,length):
            if (relation.iloc[i,j] > 0.98) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])
newDataSet_hc=newDataSet_fn.drop(del_cols,axis=1)
for i in del_cols:
    if i in test_B_file_fn.columns:
        test_B_file_fn=test_B_file_fn.drop([i],axis=1)
#rebalance the sample, as most people are not default, to improve the model efficiency
train_A_file_1=train_A_file[train_A_file["flag"]==1]
train_A_file_0=train_A_file[train_A_file["flag"]==0]
train_B_file_1=train_B_file[train_B_file["flag"]==1]
train_B_file_0=train_B_file[train_B_file["flag"]==0]
df_sample_a0=train_A_file_0.sample(train_A_file_1.shape[0])
train_A=train_A_file_1.append(df_sample_a0)
train_A=train_A.sample(frac=1)
df_sample_B0=train_B_file_0.sample(train_B_file_1.shape[0])
train_B=train_B_file_1.append(df_sample_B0)
train_B=train_B.sample(frac=1)
#use GridSearch for parameter selection
#Xgboost
param_test0 = {'learning_rate':[0.02,0.04, 0.06, 0.08, 0.1,0.2]}
gsearch_0 = GridSearchCV(estimator = xgb.XGBRegressor(random_state = 1),
                        param_grid = param_test0, cv = 3)
gsearch_0.fit(newDataSet_hc1,target1)
#lgBoost model
param_test1= {'learning_rate':[0.02,0.04, 0.06,0.08,0.1,0.2]}
gsearch = GridSearchCV(estimator = lgb.LGBMRegressor(random_state = 1),
                        param_grid = param_test1, cv = 3)
gsearch.fit(newDataSet_hc1,target1)
#random forest model
from sklearn.ensemble import RandomForestRegressor
param_test2= {"max_depth":[40,60,80,100,120,200]}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(random_state = 1),
                        param_grid = param_test2, cv = 3)
gsearch2.fit(newDataSet_hc1,target1)
#svm model
from sklearn import svm
classifier=svm.SVC(kernel="rbf")
classifier.fit(newDataSet_hc1,target1)
#model fitting
model_xbg=xgb.XGBRegressor(learning_rate=0.2)
model_xbg.fit(newDataSet_hc1,target1)
model_lgb=lgb.LGBMRegressor(learning_rate=0.04)
model_lgb.fit(newDataSet_hc1,target1)
#model stacking
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            X = X.reset_index(drop=True)
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models=StackingAveragedModels(base_models =(model_xbg,classifier),meta_model =model_lgb)
stacked_averaged_models.fit(newDataSet_hc1,target1)

prediction=stacked_averaged_models.predict(test_B_file_data)
#test the accuracy of model result with a confusion matrix
from sklearn import metrics
cm_test=metrics.confusion_matrix(test_B_file["flag"],test_B_file_data,[1,0])
sns.heapmap(cm_test,annot=True,fmt=".0f")
plt.title("Train data confusion matrix")
plt.show()