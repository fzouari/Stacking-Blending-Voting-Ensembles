import os
import pickle

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=23)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.3, random_state=23)

    def load_waveform_freq_half_data(self):
        path = 'E:/generated_results/waveforms_for_training'
        waveform_data_file = os.path.join(path, 'annotated_waveforms.pkl')
        with open(waveform_data_file, 'rb') as handle:
            waveform_data = pickle.load(handle)
        x, y = waveform_data['x_waveform_freq2'], waveform_data['y_data']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=23)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.3, random_state=23)

    def load_waveform_freq_data(self):
        path = 'E:/generated_results/waveforms_for_training'
        waveform_data_file = os.path.join(path, 'annotated_waveforms.pkl')
        with open(waveform_data_file, 'rb') as handle:
            waveform_data = pickle.load(handle)
        x, y = waveform_data['x_waveform_freq'], waveform_data['y_data']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=23)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.3, random_state=23)

    def load_waveform_data(self):
        path = 'E:/generated_results/waveforms_for_training'
        waveform_data_file = os.path.join(path, 'annotated_waveforms.pkl')
        with open(waveform_data_file, 'rb') as handle:
            waveform_data = pickle.load(handle)
        x, y = waveform_data['x_waveform'], waveform_data['y_data']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=23)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.3, random_state=23)

    def load_waveform_features_data(self):
        path = 'E:/generated_results/waveforms_for_training'
        waveform_data_file = os.path.join(path, 'annotated_waveforms.pkl')
        with open(waveform_data_file, 'rb') as handle:
            waveform_data = pickle.load(handle)
        x, y = waveform_data['x_data'], waveform_data['y_data']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15, random_state=23)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.3, random_state=23)

    def BlendingClassifier(self):

        # Define weak learners
        weak_learners = [('dt', DecisionTreeClassifier()),
                         ('knn', KNeighborsClassifier()),
                         ('rf', RandomForestClassifier()),
                         ('lsvc', LinearSVC()),
                         ('gn', GaussianNB())]

        # ('svc', SVC(gamma=2, C=1)), ('svc', SVC(kernel="linear", C=0.025)), ('lsvc', LinearSVC), ('nys', Nystroem), ('sgd', SGDClassifier),
        #
        # Finaler learner or meta model
        final_learner = LogisticRegression()

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            
            # Predictions for each classifier based on k-fold
            print('---------{}----------'.format(clf_id))
            train_predictions, val_predictions, test_predictions = self.train_level_0(clf)
            print(f"Train accuracy: {clf.score(self.x_train, self.y_train)}")
            print(f"Validation accuracy: {clf.score(self.x_val, self.y_val)}")
            print(f"Test accuracy: {clf.score(self.x_test, self.y_test)}")

            # Stack predictions which will form 
            # the inputa data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, val_predictions))
            else:
                train_meta_model = val_predictions

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, test_predictions))
            else:
                test_meta_model = test_predictions
        
        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T
        
        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def train_level_0(self, clf):
        # Train with base x_train
        clf.fit(self.x_train, self.y_train)

        train_predictions = clf.predict(self.x_train)

        # Generate predictions for the holdout set (validation)
        # These predictions will build the input for the meta model
        val_predictions = clf.predict(self.x_val)
        
        # Generate predictions for original test set
        # These predictions will be used to test the meta model
        test_predictions = clf.predict(self.x_test)

        return train_predictions, val_predictions, test_predictions

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.y_val)
        print('==================={}================='.format('final learner'))
        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model,  self.y_val)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.y_test)}")
        print('*******************{}*****************'.format('Confusion Matrix'))
        y_test_prediction = final_learner.predict(test_meta_model)
        fp = np.sum(np.logical_and(y_test_prediction == 1, self.y_test == 0))
        tp = np.sum(np.logical_and(y_test_prediction == 1, self.y_test == 1))
        tn = np.sum(np.logical_and(y_test_prediction == 0, self.y_test == 0))
        fn = np.sum(np.logical_and(y_test_prediction == 0, self.y_test == 1))
        print("tn = {}, fp = {}, fn = {}, tp = {}".format(tn, fp, fn, tp))
        tn, fp, fn, tp = confusion_matrix(y_test_prediction, self.y_test, labels=[1, 0]).ravel()
        print("tn = {}, fp = {}, fn = {}, tp = {}".format(tn, fp, fn, tp))
        print("list of fp:".format(np.where(np.logical_and(y_test_prediction == 1, self.y_test == 0))))
        print('*******************{}*****************'.format('Precision - Recall'))
        print('Precision = {} - Recall = {}'.format(tp/(tp+fp), tp/(tp+fn)))
        

if __name__ == "__main__":
    ensemble = Ensemble()
    x, y = load_breast_cancer(return_X_y=True)
    ensemble.split_data(x, y)
    # ensemble.load_waveform_features_data()
    # ensemble.load_waveform_data()
    # ensemble.load_waveform_freq_data()
    # ensemble.load_waveform_freq_half_data()
    print('size of the data: Training {} - Validation {} - Test {}'.format(ensemble.x_train.shape[0], ensemble.x_val.shape[0], ensemble.x_test.shape[0]))
    ensemble.BlendingClassifier()
