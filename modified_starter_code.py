from helpers import *
from opportunity_dataset import OpportunityDataset
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.fftpack import fftfreq, rfft

def create_io_pairs_with_initial(inputs, labels):

	window_size = 15
	stride = 5

	# initial
	X = [list(np.average(inputs[i:i+window_size,1:],axis=0)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	Y = [list((stats.mode(labels[i:i+window_size,:], axis = None)[0]).astype(int)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	return np.array(X), np.array(Y)

def create_io_pairs_with_RMS(inputs, labels):

	window_size = 15
	stride = 5

	# RMS
	X = [list(np.sqrt(np.average(inputs[i:i+window_size,1:]**2,axis=0))) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	Y = [list((stats.mode(labels[i:i+window_size,:], axis = None)[0]).astype(int)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	return np.array(X), np.array(Y)

def create_io_pairs_with_sd(inputs, labels):

	window_size = 15
	stride = 5

	# sd
	X = [list(np.std(inputs[i:i+window_size,1:],axis=0)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	Y = [list((stats.mode(labels[i:i+window_size,:], axis = None)[0]).astype(int)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	return np.array(X), np.array(Y)

def nan_helper(y):
	print y.shape
	return np.isnan(y)

def impute_data(arr):
	#Data imputation code goes here!

	for x in xrange(arr.shape[1]):
		col = arr[:,x]
		not_nan = np.logical_not(np.isnan(col))
		ind = np.arange(len(col))
		new_col = np.interp(ind,ind[not_nan],col[not_nan])
		arr[:,x] = new_col
	return arr

def test_imputation(dataset):

	#Get the input array on which to perform imputation
	training_data, testing_data = dataset.leave_subject_out(left_out = ["S2", "S3", "S4"])
	X_train, Y_train = create_dataset(training_data, dataset.data_map["AccelWristSensors"], dataset.locomotion_labels["idx"])

	arr = X_train
	print arr.shape
	out = impute_data(arr)		
	baseline = np.load("imputed_data.npy")

	count = 1
	while(X_train[count, 0] > 0):
		count += 1
	#Only compute the sum for the first ADL run
	return np.sum( (out[:count, :] - baseline[:count, :])**2 )
	# return np.sum( (out - baseline)**2 )

def train(X, Y):
	#This is where you train your classifier, right now a dummy 
	#classifier which uniformly guesses a label is "trained"

	model = RandomForestClassifier(n_estimators = 50, oob_score = True)

	model.fit(X, Y.flatten())
	return model

def test(X, model):
	#This is where you compute predictions using your trained classifier
	#...
	Y = model.predict(X)
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
		X_train_1, Y_train_1 = create_io_pairs_with_initial(X_train, Y_train)

		X_train_2, Y_train_2 = create_io_pairs_with_RMS(X_train, Y_train)

		X_train_3, Y_train_3 = create_io_pairs_with_sd(X_train, Y_train)

		X_test_1, Y_test_1 = create_io_pairs_with_initial(X_test, Y_test)

		X_test_2, Y_test_2 = create_io_pairs_with_RMS(X_test, Y_test)

		X_test_3, Y_test_3 = create_io_pairs_with_sd(X_test, Y_test)

		#Fit your classifier
		model_1 = train(X_train_1, Y_train_1)

		model_2 = train(X_train_2, Y_train_2)

		model_3 = train(X_train_3, Y_train_3)

		#Make predictions on the test set here
		Y_pred_1 = test(X_test_1, model_1)

		Y_pred_2 = test(X_test_2, model_2)

		Y_pred_3 = test(X_test_3, model_3)

		Y_pred = np.empty(Y_pred_1.shape,dtype=int)


		for i in range(len(Y_pred_1)):
			list = []
			list.append(Y_pred_1[i])
			list.append(Y_pred_2[i])
			list.append(Y_pred_3[i])
			Y_pred[i] = max(set(list), key=list.count)


		#Append prediction and current labels to cv dataset
		Y_pred_total.append( Y_pred.reshape((Y_pred.size, 1)) )
		Y_test_total.append( Y_test_1.reshape((Y_test_1.size, 1)) )

	#Perform evaluations
	eval_preds(Y_pred_total, Y_test_total, labels["classes"])

if __name__ == "__main__":

	dataset = OpportunityDataset()

	sensors = dataset.data_map["FullBodySensors"]
	
	#Locomotion labels
	cv_train_test(dataset, sensors, dataset.locomotion_labels)

	#Activity labels
	cv_train_test(dataset, sensors, dataset.activity_labels)
