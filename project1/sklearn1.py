# import-->using model-->fit
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(train,train)

# train and test split
from sklearn import cross_validation
from sklearn import datasets
from sklearn.svm import SVC
iris=datasets.load_iris()
features=iris.data
labels=iris.target
features_train,features_test,labels_train,labels_test=
cross_validation.train_test_split(features,labels,test_size=0.4,random_state=0)
clf=SVC(kernel='linear',C=1)
clf.fit(features_train,labels_train)


