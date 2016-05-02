import sklearn
import numpy as np

import lib.utils
import lib.scikit_model_optimization as opt
from sklearn import feature_selection
from sklearn import tree

##read file using the buildint util
#train_x,train_y,validate_set_X, validate_set_Y = utils.readtrain_validate('./data/train/adult.data')

features_train,labels_train = lib.utils.read_X_Y('./data/train/adult.data')
test_X,test_Y = lib.utils.read_X_Y('./data/test/adult.test')

#"""

best_model = opt.select_best_model(features_train,labels_train, data_title="Divorce ")
prediction = opt.run_model(best_model[0],features_train,labels_train,test_X,alpha=best_model[1],beta=best_model[2])
#score test x
counter = 0
for y in test_Y:
	if y ==prediction[y]:
		counter +=1
print "score: ", counter/ float(len(test_Y))

#"""


#baseline
clf = tree.DecisionTreeClassifier(max_depth = 6).fit(features_train,labels_train)
score = clf.score(test_X, test_Y)	
print "decision tree baseline ", score
print

## try again with feature selections
##kbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

anova_filter = SelectKBest(f_regression, k=5)
clf = tree.DecisionTreeClassifier(max_depth = 6)
anova_clf = Pipeline([('anova', anova_filter), ('svc', clf)])

for x in range(2,len(features_train[0])+1,1):
	anova_filter.k = x
	pipeline = anova_clf.fit(features_train, labels_train)
	print "feat kbest :",x, " score: ", pipeline.score(test_X, test_Y)	
print

##
## variance threshold
from sklearn.feature_selection import VarianceThreshold
anova_filter = VarianceThreshold()
clf = tree.DecisionTreeClassifier(max_depth = 6)
anova_clf = Pipeline([('anova', anova_filter), ('svc', clf)])

for x in range(0,len(features_train[0])+1,1):
	variance = x/float(len(features_train[0])*5 )
	anova_filter.threshold = variance
	pipeline = anova_clf.fit(features_train, labels_train)
	print "feat var threshold :",variance, " score: ", pipeline.score(test_X, test_Y)	
print


##L1
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
for x in range(-5,0,1):
	c = 10**x
	lsvc = LinearSVC(C=c, penalty="l1", dual=False).fit(features_train, labels_train)
	model = SelectFromModel(lsvc, prefit=True)
	feat = model.transform(features_train)
	clf = tree.DecisionTreeClassifier(max_depth = 6).fit(feat, labels_train)
	feat_test = model.transform(test_X)
	print "feat L1 SVC: ", c ," feat score: ", clf.score(feat_test, test_Y)	
print

##extratrees
for x in range(2,len(features_train[0])+1,1):
	clf = tree.ExtraTreeClassifier(max_depth = x).fit(features_train, labels_train)
	model = SelectFromModel(clf, prefit=True)
	feat = model.transform(features_train)
	clf = tree.DecisionTreeClassifier(max_depth = 6).fit(feat, labels_train)
	feat_test = model.transform(test_X)
	print "feat extra-tree depth: ", x," score: ",clf.score(feat_test, test_Y)	
print

##extra trees to see which features are important 
clf = tree.ExtraTreeClassifier(max_depth = 6).fit(features_train, labels_train)
print "feature weights: "
print clf.feature_importances_  

