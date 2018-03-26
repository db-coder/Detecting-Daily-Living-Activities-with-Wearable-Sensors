import scipy.io as sio
import numpy as np

class OpportunityDataset:

	def __init__(self, dataset_root = "OpportunityUCIDataset/dataset/"):
		self.subject_data = {"S1": [ sio.loadmat(dataset_root + "S1-ADL1.mat")['data'],
				 		sio.loadmat(dataset_root + "S1-ADL2.mat")['data'],
				 		sio.loadmat(dataset_root + "S1-ADL3.mat")['data'],
				 		sio.loadmat(dataset_root + "S1-ADL4.mat")['data'],
				 		sio.loadmat(dataset_root + "S1-ADL5.mat")['data'],
				 		sio.loadmat(dataset_root + "S1-Drill.mat")['data'] ],
						"S2": [ sio.loadmat(dataset_root + "S2-ADL1.mat")['data'],
				 		sio.loadmat(dataset_root + "S2-ADL2.mat")['data'],
				 		sio.loadmat(dataset_root + "S2-ADL3.mat")['data'],
				 		sio.loadmat(dataset_root + "S2-ADL4.mat")['data'],
				 		sio.loadmat(dataset_root + "S2-ADL5.mat")['data'],
				 		sio.loadmat(dataset_root + "S2-Drill.mat")['data'] ],
						"S3": [ sio.loadmat(dataset_root + "S3-ADL1.mat")['data'],
				 		sio.loadmat(dataset_root + "S3-ADL2.mat")['data'],
				 		sio.loadmat(dataset_root + "S3-ADL3.mat")['data'],
				 		sio.loadmat(dataset_root + "S3-ADL4.mat")['data'],
				 		sio.loadmat(dataset_root + "S3-ADL5.mat")['data'],
				 		sio.loadmat(dataset_root + "S3-Drill.mat")['data'] ],
						"S4": [ sio.loadmat(dataset_root + "S4-ADL1.mat")['data'],
				 			sio.loadmat(dataset_root + "S4-ADL2.mat")['data'],
				 			sio.loadmat(dataset_root + "S4-ADL3.mat")['data'],
				 			sio.loadmat(dataset_root + "S4-ADL4.mat")['data'],
				 			sio.loadmat(dataset_root + "S4-ADL5.mat")['data'],
				 			sio.loadmat(dataset_root + "S4-Drill.mat")['data'] ]}
		#Mapping from sensor names to matrix columns
		self.data_map = {"T": [0], "RWR": [1,2,3], "LWR":[4,5,6], 
					"BACK": range(7,20), "RUA":range(20,33), "LUA":range(33,46),
					"RLA":range(46,59), "LLA":range(59, 72)} 
		self.data_map["AccelWristSensors"] = self.data_map["T"] + self.data_map["RWR"] + self.data_map["LWR"] 
		self.data_map["ImuWristSensors"] = self.data_map["AccelWristSensors"] + self.data_map["RLA"] + self.data_map["LLA"]
		self.data_map["FullBodySensors"] = self.data_map["ImuWristSensors"] + self.data_map["BACK"] + self.data_map["RUA"] + self.data_map["LUA"] 
		self.locomotion_labels ={"idx": [72], "classes":["Null", "Stand", "Sit", "Walk", "Lie"]}
		self.activity_labels ={"idx": [73], "classes":["Null", "Open Door", "Close Door", "Open Dishwasher", 
														"Close Dishwasher", "Open Drawer", "Close Drawer", 
														"Clean Table", "Drink Coffee", "Toggle Switch"]} 

	def leave_subject_out(self, left_out):
		training_data = []
		testing_data = []
		for key in self.subject_data:
			if (key not in left_out):
				training_data.extend(self.subject_data[key])
			else:
				testing_data.extend(self.subject_data[key])
		return training_data, testing_data

	def small(self):
		training_data = [self.subject_data["S1"][0]]
		testing_data = [self.subject_data["S1"][1]]
		return training_data, testing_data

	def get_data(self, subject_idxs, train_idxs = [0,1,2,5], test_idxs = [3,4]):
		training_data = []
		testing_data = []
		for key in subject_idxs:
			training_data.extend([self.subject_data[key][idx] for idx in train_idxs])
			testing_data.extend([self.subject_data[key][idx] for idx in test_idxs])
		return training_data, testing_data