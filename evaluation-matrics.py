#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now. 
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to numeric data
X = X._get_numeric_data()
# Separate the labels
y = X['Survived']
# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
# Split the data
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.4,random_state=0)
# 1Q Evaluate the accuracy
# The decision tree classifier
clf1 = DecisionTreeClassifier()
clf1.fit(x_train,y_train)
print "Decision Tree has accuracy: ",accuracy_score(y_test, clf1.predict(x_test))
# The naive Bayes classifier

clf2 = GaussianNB()
clf2.fit(x_train,y_train)
print "GaussianNB has accuracy: ",accuracy_score(y_test, clf2.predict(x_test))

answer = { 
 "Naive Bayes Score": accuracy_score(y_test, clf1.predict(x_test)), 
 "Decision Tree Score": accuracy_score(y_test, clf2.predict(x_test))
}
# 2Q Build a confusion matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
clf1 = DecisionTreeClassifier()
clf1.fit(x_train,y_train)
print "Confusion matrix for this Decision Tree:\n",confusion_matrix(y_test,clf1.predict(x_test))

clf2 = GaussianNB()
clf2.fit(x_train,y_train)
print "GaussianNB confusion matrix:\n",confusion_matrix(y_test,clf2.predict(x_test))


confusions = {
 "Naive Bayes": confusion_matrix(y_test,clf1.predict(x_test)),
 "Decision Tree": confusion_matrix(y_test,clf2.predict(x_test))
}

# 3Q Precision Vs Recall
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf1.predict(x_test)),precision(y_test,clf1.predict(x_test)))

clf2 = GaussianNB()
clf2.fit(x_train, y_train)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf2.predict(x_test)),precision(y_test,clf2.predict(x_test)))

results = {
  "Naive Bayes Recall": recall(y_test,clf2.predict(x_test)),
  "Naive Bayes Precision": precision(y_test,clf2.predict(x_test)),
  "Decision Tree Recall": recall(y_test,clf1.predict(x_test)),
  "Decision Tree Precision": precision(y_test,clf1.predict(x_test))
}

# 4Q Compute F1 Score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)
print "Decision Tree F1 score: {:.2f}".format(f1_score(y_test, clf1.predict(x_test)))

clf2 = GaussianNB()
clf2.fit(x_train, y_train)
print "GaussianNB F1 score: {:.2f}".format(f1_score(y_test, clf2.predict(x_test)))

F1_scores = {
 "Naive Bayes": f1_score(y_test, clf2.predict(x_test)),
 "Decision Tree": f1_score(y_test, clf1.predict(x_test))
}

# 5Q Mean Absolute Error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
reg1 = DecisionTreeRegressor()
reg1.fit(x_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(y_test,reg1.predict(x_test)))

reg2 = LinearRegression()
reg2.fit(x_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(y_test,reg2.predict(x_test)))

results = {
 "Linear Regression": mae(y_test,reg2.predict(x_test)),
 "Decision Tree": mae(y_test,reg1.predict(x_test))
}

# 6Q Mean Squared Error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
reg1 = DecisionTreeRegressor()
reg1.fit(x_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mse(y_test, reg1.predict(x_test)))

reg2 = LinearRegression()
reg2.fit(x_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mse(y_test, reg2.predict(x_test)))

results = {
 "Linear Regression": mse(y_test, reg2.predict(x_test)),
 "Decision Tree": mse(y_test, reg1.predict(x_test))
}



