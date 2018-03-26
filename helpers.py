import numpy as np
import scipy.signal
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class DummyClassifier:

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def predict(self, X):
        return np.random.randint(0, self.output_dim,  X.shape[0])

#Make new array consisting of the given list of columns
def reduce_arr(arr, cols):
	out = (np.concatenate([arr[:,c] for c in cols]).reshape(len(cols), arr.shape[0])).T
	return out

def create_dataset(data, sensors, labels):
	X, Y = [], []
	for arr in data:
		X.extend(reduce_arr(arr, sensors))
		Y.extend(reduce_arr(arr, labels))
	return np.asarray(X), np.asarray(Y)

def eval_preds(Y_pred, Y_test, classes):
    Y_pred = np.concatenate(Y_pred)
    Y_test = np.concatenate(Y_test)
    cm = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(cm, classes)
    score = f1_score(Y_test, Y_pred, average = 'weighted')
    print "Weighted F-Score:"
    print score

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion Matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()