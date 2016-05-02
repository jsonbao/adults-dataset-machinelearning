import sklearn
import numpy as np

import matplotlib.pyplot as plt
import lib.utils


from sklearn import cluster
from sklearn import metrics

##read file using the buildint util
#train_x,train_y,validate_set_X, validate_set_Y = utils.readtrain_validate('./data/train/adult.data')
features_train,labels_train = lib.utils.read_X_Y('./data/train/adult.data')
test_X,test_Y = lib.utils.read_X_Y('./data/test/adult.test')
random_state=100

print "Minibatch K-means  - scoring model has to start at K=2"
bestk = {}
bestkchange = {}
prevscore = 0
for k in range(2,12,1):
	predict = cluster.MiniBatchKMeans(n_clusters=k, random_state=random_state).fit_predict(features_train)
	score = metrics.adjusted_mutual_info_score(labels_train, predict)  
	if k==2:
		prevscore = score
	bestkchange[k]=score - prevscore
	bestk[k] = score
	prevscore = score
	print "K clusters:",k," score: ", score

##normalize and find the bestk
normalizek = {}
number1 = sorted(bestk.iteritems(), key=lambda x:-x[1])[:1][0]
print number1
for key in bestk.keys():
	if key <len(bestk.keys()):
		normalizek[key] =(1+bestkchange[key]- bestkchange[key+1]) /(1+ number1[1]-bestk[key])
		print  str(key) + " normalize score: ", (1+bestkchange[key]- bestkchange[key+1]) /(1+ number1[1]-bestk[key])


#print normalizek
normalizek1 = sorted(normalizek.iteritems(), key=lambda x:-x[1])[:1][0][0]
K = normalizek1
print "best number of K clusters: ",K

Zs = cluster.MiniBatchKMeans(n_clusters=K, random_state=random_state).fit(features_train).predict(features_train)
score = metrics.adjusted_mutual_info_score(labels_train, Zs)  
print "fulltrain score: ",score
## output prediction
lib.utils.kaggleize(Zs,"./predictions/train_predict.csv")


### rerun for best optimal
Zs = cluster.MiniBatchKMeans(n_clusters=K, random_state=random_state).fit(features_train).predict(test_X)
score = metrics.adjusted_mutual_info_score(test_Y, Zs)  
print "test score: ",score
## output prediction
lib.utils.kaggleize(Zs,"./predictions/test_predict.csv")

## print number count of each cluster
for k in np.unique(Zs):
	print str(k) +": count: ", len([ks for ks in Zs if ks==k])

print 


## othe clustering models too slow
##Zs = cluster.Birch(n_clusters=4).fit_predict(test_X)


clusters = 4


##try PCA denoising
from sklearn.decomposition import PCA
pca        = PCA(copy=False,n_components=6, whiten=False).fit(features_train)
data_denoised =pca.transform(features_train)
feat_test =pca.transform(test_X)
print data_denoised.shape

Zs = cluster.MiniBatchKMeans(n_clusters=clusters, random_state=random_state).fit(data_denoised).predict(feat_test)
score = metrics.adjusted_mutual_info_score(test_Y, Zs)  
print "PCA test score denoised: ",score
## print number count of each cluster
for k in np.unique(Zs):
	print str(k) +": count: ", len([ks for ks in Zs if ks==k])
print


##try FastICA denoising
from sklearn.decomposition import FastICA
ica        = FastICA(n_components=2).fit(features_train)
data_denoised =ica.transform(features_train)
feat_test =ica.transform(test_X)
print data_denoised.shape


Zs = cluster.MiniBatchKMeans(n_clusters=clusters, random_state=random_state).fit(data_denoised).predict(feat_test)
score = metrics.adjusted_mutual_info_score(test_Y, Zs)  
print "fastICA test score denoised: ",score
## print number count of each cluster
for k in np.unique(Zs):
	print str(k) +": count: ", len([ks for ks in Zs if ks==k])



Zs = cluster.MiniBatchKMeans(n_clusters=clusters, random_state=random_state).fit(data_denoised).predict(data_denoised)
## output prediction
lib.utils.kaggleize(Zs,"./predictions/train_predict.csv")
