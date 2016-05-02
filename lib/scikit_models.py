
###################################################################################################
#models


def scikit_dc_tree(X,y, X_test,y_test=None,alpha=None):
    from sklearn import tree
    predictions = tree.DecisionTreeClassifier(random_state=0,max_depth=alpha,criterion='gini').fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions


def scikit_knn_model(X,y, X_test,y_test=None,alpha=None):
	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=alpha)
	neigh.fit(X, y) 
	y_test_predict =neigh.predict(X_test)


	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(y_test_predict):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy = float(correctcount)/totalcount
	    return accuracy

	return y_test_predict

def scikit_log_reg(X,y, X_test,y_test=None,alpha=None):
    from sklearn import linear_model
    predictions = linear_model.LogisticRegression(C=alpha, penalty='l2').fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions



def scikit_onevsrest(X,y, X_test,y_test=None,alpha=None):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    predictions = OneVsRestClassifier(LinearSVC(random_state=0,C=alpha)).fit(X, y).predict(X_test)

    if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

    return predictions


def scikit_ran_forest(X,y, X_test,y_test=None,alpha=None,beta =None):
	from sklearn.ensemble import RandomForestClassifier
	predictions = RandomForestClassifier(n_estimators=alpha,max_depth=beta,random_state=0,n_jobs=-1).fit(X, y).predict(X_test)

	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

	return predictions


def scikit_ada_boost(X,y, X_test,y_test=None,alpha=None,beta =None):
	from sklearn.ensemble import AdaBoostClassifier
	predictions = AdaBoostClassifier(n_estimators=alpha,random_state=0).fit(X, y).predict(X_test)

	if y_test is not None:
	    correctcount = 0
	    totalcount = 0
	    for index, each in enumerate(predictions):
	        if y_test[index] == each:
	            correctcount +=1
	        totalcount+=1

	    accuracy =float(correctcount)/totalcount
	    return accuracy

	return predictions


