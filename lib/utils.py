
import numpy as np
import pandas as pd

"""
# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://goo.gl/j0Rvxq"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]
"""

"""
import pandas as pd
# comma delimited is the default
df = pd.read_csv(input_file, header = 0)
## label encoding!!!!
from sklearn import preprocessing
le_encode = preprocessing.LabelEncoder()
#to convert into numbers
df.McCain = le_encode.fit_transform(df.McCain)


enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])   /// data

enc.transform([[0, 1, 3]]).toarray() // turn all features to one array
"""
# np.recfromcsv(input_file, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')


def readtraining_validate(name):
	input_file = name

	df=pd.read_csv(input_file, sep=',',header=None)

	print(df.shape)
	print(df[:2])


	#to convert into numbers
	from sklearn import preprocessing
	le_encode = preprocessing.LabelEncoder()
	for i in range(0,df.shape[1],1):
		le_encode.fit(df[i])
		df[i] = le_encode.transform(df[i])

	training_Y = np.array((df[5]))
	training_X = np.array(df.drop([2,5,7], axis=1))


	print training_Y.shape
	print training_X.shape
	print training_Y[:1]
	print training_X[:1]

	list1x = training_X[0::10]
	list2x = training_X[1::10]
	list3x = training_X[2::10]
	list4x = training_X[3::10]
	list5x = training_X[4::10]	
	list6x = training_X[5::10]
	list7x = training_X[6::10]
	list8x = training_X[7::10]
	list9x = training_X[8::10]
	list10x = training_X[9::10]

	list1y = training_Y[0::10]
	list2y = training_Y[1::10]
	list3y = training_Y[2::10]
	list4y = training_Y[3::10]
	list5y = training_Y[4::10]
	list6y = training_Y[5::10]
	list7y = training_Y[6::10]
	list8y = training_Y[7::10]
	list9y = training_Y[8::10]
	list10y = training_Y[9::10]

	##split = training_X.shape[0] * .90
	seed = 1
	train_set_X = np.concatenate((list2x,list3x,list4x,list5x,list6x,list7x))
	train_set_Y = np.concatenate((list2y,list3y,list4y,list5y,list6y,list7y))
	validate_set_X = list8x
	validate_set_Y = list8y

	return train_set_X,train_set_Y,validate_set_X, validate_set_Y


def read_X_Y(name):
	input_file = name

	df=pd.read_csv(input_file, sep=',',header=None)

	#to convert into numbers
	from sklearn import preprocessing
	le_encode = preprocessing.LabelEncoder()
	for i in range(0,df.shape[1],1):
		le_encode.fit(df[i])
		df[i] = le_encode.transform(df[i])

	label_Y = np.array((df[5]))
	label_X = np.array(df.drop([2,5,7], axis=1))

	return label_X,label_Y

def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
      predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions))
    np.savetxt(file,kaggle_predictions,fmt=["%d","%f"], delimiter=",",
               header="ID,Target", comments='')
    