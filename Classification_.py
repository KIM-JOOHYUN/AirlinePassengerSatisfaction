import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings( 'ignore' )

#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

#ROC curve
def rocvis(true , prob , label ) :
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker='.', label = label  )


# Load the dataset from csv file
df = pd.read_csv('train.csv')
#Preprocessing
x_column = df.columns.drop('satisfaction')
x_column = x_column.drop('id')
y_column = ['satisfaction']
X = np.array(pd.DataFrame(df, columns = x_column))
y = np.array(pd.DataFrame(df, columns = y_column))

#convert categorical attributes into numeric ones
encoder = LabelEncoder()
for i in range(len(x_column)) :
    X[:,i] = encoder.fit_transform(X[:,i])
y = encoder.fit_transform(y)

#nomalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y)

#make model
d_model = DecisionTreeClassifier()
predict = d_model.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, predict)
print('Decision Tree Accuracy(Default) :',accuracy)
print()
l_model = LogisticRegression()
predict = l_model.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, predict)
print('Logistic Regression Accuracy(Default) :',accuracy)
print()
s_model = LinearSVC()
predict = s_model.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, predict)
print('LinearSVC Accuracy(Default):',accuracy)
print()


#parameters for GridSearch
d_params = {
    'max_depth' : [None,6, 8, 10, 12, 16, 20, 24],
    'min_samples_split' : [2,20,50,100,200],
    'criterion':['entropy','gini']
}
l_params = {
    'C' : [0.1,1.0,10.0],
   'solver' : ['liblinear','lbfgs','sag'],
    'max_iter' : [50,100,200]
}
s_params = {
    'C' : [0.1,1.0,10.0],
    'loss':['hinge', 'squared_hinge'],
    'penalty' : ['l1', 'l2'],
    'multi_class':['ovr', 'crammer_singer']
    
}

#K-Fold
cv=KFold(n_splits=5, random_state=1, shuffle=False)

#GridSearch
print()
print('--------------Decision Tree--------------')
d_grid_cv = GridSearchCV(d_model, param_grid=d_params, scoring='accuracy', cv=cv, verbose=1)
d_grid_cv.fit(X_train, y_train)
d_scores = d_grid_cv.cv_results_['mean_test_score']
print("Cross-verification Scores : ", d_scores)

d_best_model = d_grid_cv.best_estimator_
print('Best parameter :', d_grid_cv.best_params_)
print('Best score :',d_grid_cv.best_score_)

d_predict = d_best_model.predict(X_test)
d_accuracy = accuracy_score(y_test, d_predict)
print('Test Accuracy :',d_accuracy)

#confusion matrix
plt.figure(figsize=[10, 6])
cm = pd.DataFrame(confusion_matrix(y_test, d_predict))
sns.heatmap(cm, annot=True)
plt.title('Decision Tree ')
plt.show()

print()
print('--------------Logistic Regression--------------')
l_grid_cv = GridSearchCV(l_model, param_grid=l_params, scoring='accuracy', cv=cv, verbose=1)
l_grid_cv.fit(X_train, y_train)
l_scores = l_grid_cv.cv_results_['mean_test_score']
print("Cross-verification Scores : ", l_scores)

l_best_model = l_grid_cv.best_estimator_
print('Best parameter :', l_grid_cv.best_params_)
print('Best score :',l_grid_cv.best_score_)

l_predict = l_best_model.predict(X_test)
l_accuracy = accuracy_score(y_test, l_predict)
print('Test Accuracy :',l_accuracy)

#confusion matrix
plt.figure(figsize=[10, 6])
cm = pd.DataFrame(confusion_matrix(y_test, l_predict))
sns.heatmap(cm, annot=True)
plt.title('Logistic Regression ')
plt.show()

print()
print('--------------LinearSVM--------------')
s_grid_cv = GridSearchCV(s_model, param_grid=s_params, scoring='accuracy', cv=cv, verbose=1)
s_grid_cv.fit(X_train, y_train)
s_scores =s_grid_cv.cv_results_['mean_test_score']
print("Cross-verification Scores : ", s_scores)

s_best_model = s_grid_cv.best_estimator_
print('Best parameter :', s_grid_cv.best_params_)
print('Best score :',s_grid_cv.best_score_)

s_predict = s_best_model.predict(X_test)
s_accuracy = accuracy_score(y_test, s_predict)
print('Test Accuracy :',s_accuracy)

#confusion matrix
plt.figure(figsize=[10, 6])
cm = pd.DataFrame(confusion_matrix(y_test, s_predict))
sns.heatmap(cm, annot=True)
plt.title('LinearSVM')
plt.show()


#Decision Tree Feature importance
#show importance of each feature
importance_feature = d_best_model.feature_importances_
feature_importances = pd.Series(importance_feature, index=x_column)
# sort the Series by importance
feature_top = feature_importances.sort_values(ascending=False)[:]

#Feature importance
plt.figure(figsize=[10, 6])
plt.title('Feature Importances(Decision Tree)')
sns.barplot(x=feature_top, y=feature_top.index)
plt.show()


#ROC curve
fig , ax = plt.subplots(figsize= (10,6))
plt.plot([0, 1], [0, 1], linestyle='--')
rocvis(y_test, d_predict,"Decision Tree")
rocvis(y_test, l_predict,"Logistic Regression")
rocvis(y_test, s_predict," LinearSVM")
plt.legend(fontsize = 18)
plt.title("Models Roc Curve" , fontsize= 25)
plt.show()
