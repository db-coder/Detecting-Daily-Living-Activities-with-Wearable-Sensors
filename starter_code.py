from helpers import *
from opportunity_dataset import OpportunityDataset
from scipy import stats
from sklearn.preprocessing import Imputer


def create_io_pairs(inputs, labels):
	#Compute your windowed features here and labels. Right now
	#it just returns the inputs and labels without changing anything.
	#X = inputs
	#Y = labels

	window_size = 1
	stride = 1
	#...

	X = [list(np.average(inputs[i:i+window_size,:],axis=0)) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	Y = [list((stats.mode(labels[i:i+window_size,:], axis = None)[0]).astype(int)) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	#print X
	#print Y
	#exit(0)
	#....
	return X,Y


def impute_data(arr):
	#Data imputation code goes here!
	#...
	#...
	return Imputer(verbose = 9).fit_transform(arr)


def test_imputation(dataset):
	#Get the input array on which to perform imputation
	training_data, testing_data = dataset.leave_subject_out(left_out = ["S2", "S3", "S4"])
	X_train, Y_train = create_dataset(training_data, dataset.data_map["AccelWristSensors"], dataset.locomotion_labels["idx"])
	arr = X_train
	out = impute_data(arr)		
	baseline = np.load("imputed_data.npy")
	return np.sum( (out - baseline)**2 )


def train(X, Y):
	#This is where you train your classifier, right now a dummy 
	#classifier which uniformly guesses a label is "trained"
	#....
	#....
	model = {"clf": DummyClassifier( len(set(Y.flatten())) ) }
	#....
	#....
	return model


def test(X, model):
	#This is where you compute predictions using your trained classifier
	#...
	Y = model["clf"].predict(X)
	return Y


def cv_train_test(dataset, sensors, labels):
	"""
	Template code for performing leave on subject out cross-validation
	"""
	subjects = dataset.subject_data.keys()
	Y_pred_total, Y_test_total = [], []

	#Leave one subject out cross validation
	for subj in subjects:
		training_data, testing_data = dataset.leave_subject_out(left_out = subj)
		
		X_train, Y_train = create_dataset(training_data, sensors, labels["idx"])
		X_test, Y_test = create_dataset(testing_data, sensors, labels["idx"])

		#Impute missing inputs data
		X_train = impute_data(X_train)
		X_test = impute_data(X_test)

		#Compute features and labels for train and test set
		X_train, Y_train = create_io_pairs(X_train, Y_train)
		X_test, Y_test = create_io_pairs(X_test, Y_test)

		#Fit your classifier
		model = train(X_train, Y_train)

		#Make predictions on the test set here
		Y_pred = test(X_test, model)

		#Append prediction and current labels to cv dataset
		Y_pred_total.append( Y_pred.reshape((Y_pred.size, 1)) )
		Y_test_total.append( Y_test.reshape((Y_test.size, 1)) )

	#Perform evaluations
	eval_preds(Y_pred_total, Y_test_total, labels["classes"])


if __name__ == "__main__":
	#Example inputs to cv_train_test function, you would use
	#these inputs for  problem 2
	dataset = OpportunityDataset()
	sensors = dataset.data_map["AccelWristSensors"]
	
	#Locomotion labels
	cv_train_test(dataset, sensors, dataset.locomotion_labels)

	#Activity labels
	cv_train_test(dataset, sensors, dataset.activity_labels)