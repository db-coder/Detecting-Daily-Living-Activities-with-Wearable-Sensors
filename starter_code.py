from helpers import *
from opportunity_dataset import OpportunityDataset
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from scipy.fftpack import fftfreq, rfft


def create_io_pairs(inputs, labels):
	#Compute your windowed features here and labels. Right now
	#it just returns the inputs and labels without changing anything.

	window_size = 5
	stride = 3
	# initial
	# X = [list(np.average(inputs[i:i+window_size,:],axis=0)) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	
	# # RMS
	# X = [list(np.sqrt(np.average(inputs[i:i+window_size,:]**2,axis=0))) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	
	# # SD
	# X = [list(np.std(inputs[i:i+window_size,:],axis=0)) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	
	# # MAS1
	# X = [list( np.abs(np.subtract( inputs[i:i+window_size,:],np.average(inputs[i:i+window_size,:],axis=0))) ) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	
	# # MAS2
	# X = [list( np.sqrt(np.abs(np.subtract( inputs[i:i+window_size,:],np.average(inputs[i:i+window_size,:],axis=0)))) ) for i in range(0,inputs.shape[0]-window_size+1,stride)]
	
	def RP(inp):
		x_fft = rfft(inp,axis = 0)
		#print x_fft
		x_fft_sum_square = np.sum(np.abs(x_fft)**2)
		z1,z2 = np.histogram(x_fft,10)
		z2[-1] += 0.1
		op = []
		for j in range(z1.size):
			if(z1[j] == 0):
				op.append(0)
			else:
				z_temp = [np.abs(k)**2 if (k>= z2[j] and k < z2[j+1]) else 0 for k in x_fft.flatten() ]
				#print z_temp
				op.append(float(np.sum(z_temp))/x_fft_sum_square)
		return op
		#print op
		#exit(0)



	# RP
	X = [RP(inputs[i:i+window_size,:])for i in range(0,inputs.shape[0]-window_size+1,stride)]
	

	Y = [list((stats.mode(labels[i:i+window_size,:], axis = None)[0]).astype(int)) for i in range(0,inputs.shape[0]-window_size+1,stride)]

	return np.array(X),np.array(Y)

def nan_helper(y):
	print y.shape
	return np.isnan(y)

def impute_data(arr):
	#Data imputation code goes here!
	#...
	#...
	# n = arr.shape[0]
	# c1 = range(n)
	#c1_ori = arr[:,0]
	# print c1_ori.shape
	# arr[:,0] = c1
	# print arr
	#imputed = Imputer(strategy = 'median').fit_transform(arr)
	# n1 = arr.shape[0]
	# n2 = arr.shape[1]
	# n1_r = np.array(range(n1))
	# op = np.reshape(arr[:,0],(-1,1))
	# for i in range(1,n2):
	# 	y = arr[:,i]
	# 	nans= nan_helper(y)

	# 	y[nans]= np.interp(n1_r[nans], n1_r[~nans], y[~nans])
	# 	y = np.reshape(y,(-1,1))
	# 	op = np.hstack((op,y))

	# return op

	for x in xrange(arr.shape[1]):
		col = arr[:,x]
		not_nan = np.logical_not(np.isnan(col))
		ind = np.arange(len(col))
		new_col = np.interp(ind,ind[not_nan],col[not_nan])
		arr[:,x] = new_col
	return arr

def test_imputation(dataset):
	# #Get the input array on which to perform imputation
	# training_data, testing_data = dataset.leave_subject_out(left_out = ["S2", "S3", "S4"])
	# X_train, Y_train = create_dataset(training_data, dataset.data_map["AccelWristSensors"], dataset.locomotion_labels["idx"])
	# arr = X_train
	# #out = impute_data(arr)
	# out = impute_data(dataset.subject_data["S3"][2])	
	# baseline = np.load("imputed_data.npy")
	# return np.sum( (out - baseline)**2 )

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
	# model = {"clf": DummyClassifier( len(set(Y.flatten())) ) }

	# model = AdaBoostClassifier(n_estimators=100)
	model = RandomForestClassifier(n_estimators = 50, oob_score = True)
	model.fit(X,Y.flatten())
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

	# print test_imputation(dataset)

	# sensors = dataset.data_map["ImuWristSensors"]
	# sensors = dataset.data_map["FullBodySensors"]
	
	#Locomotion labels
	cv_train_test(dataset, sensors, dataset.locomotion_labels)

	#Activity labels
	cv_train_test(dataset, sensors, dataset.activity_labels)

	# part1 = impute_data(dataset.subject_data["S3"][2])
	# np.save("part1.npy",part1)
	# part1 = np.load("part1.npy")
	# print part1
	# print test_imputation(dataset)