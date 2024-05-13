import pandas as pd
import matplotlib.pyplot as plt
import risk_assessment
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os


class MLModule:
    def __init__(self, model='default'):
        self.fileops = risk_assessment.file_ops.FileSystem()
        self.training_df = self.fileops.import_training_dataset()
        self.X = self.fileops.load_cleaned_dataset()
        self.model_name = model
        self.clf_model = DecisionTreeClassifier()
        self.load_model(model)
        self.model_summary = self.fileops.model_summary_file_ops(model_name=self.model_name, operation='load')

    def load_model(self, model_name):
        folder_path = os.path.join(self.fileops.main_dir, 'risk_assessment/models')
        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)
        # print(folder_path)
        if model_name:
            name = self.model_name + '.pkl'
            for filename in file_list:
                # print(filename)
                if filename == name:
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'rb') as file:
                        self.clf_model = pickle.load(file)
        else:
            file_path = os.path.join(folder_path, 'default.pkl')
            with open(file_path, 'rb') as file:
                self.clf_model = pickle.load(file)

    def preprocess_data(self):
        # remove duplicates
        x = pd.DataFrame()
        duplicate_rows = self.training_df[self.training_df.duplicated()]
        print('duplicate entries:\n{}'.format(duplicate_rows))
        x = self.training_df.drop_duplicates()
        # remove NaN rows
        x.dropna(inplace=True)
        print('Removed duplicates and NaN rows, new shape: {}'.format(x.shape))
        self.fileops.create_csv(x, 'x', 'risk_assessment/cleaneddata')
        self.X = x

    def analyse_dataset(self):
        corr_matrix = self.X.corr(numeric_only=True)
        self.X.plot(kind="scatter", x="lastmonth_activity", y="lastyear_activity", alpha=0.4,
                    s=self.X['number_of_employees'], label="number of employees",
                    figsize=(10, 7), c="exited", cmap=plt.get_cmap("jet"), colorbar=True)
        plt.legend()
        plt.show()
        timestamp = int(time.time())
        plt.savefig('risk_assessment/inferencedata/' + str(timestamp) + '_scatterplot.png')
        self.fileops.create_csv(corr_matrix, 'correlations', 'risk_assessment/inferencedata')

    def train_decision_tree_model(self, splitter='best', max_depth=None, min_samples_split=2):
        clf_model = DecisionTreeClassifier(splitter=splitter, max_depth=int(max_depth),
                                           min_samples_split=int(min_samples_split))
        y = self.X['exited']
        x = self.X.drop(columns=['corporation', 'exited'])
        print('training model {} with splitter: {}, max_depth: {}, min_samples_split: {} hyperparameter settings'
              .format(self.model_name, splitter, max_depth, min_samples_split))
        start_time = time.time()
        clf_model.fit(x, y)
        end_time = time.time()
        session_time = end_time - start_time
        print('time elapsed {}'.format(session_time))
        self.clf_model = clf_model
        model_pkl_file = self.model_name + '.pkl'
        file_path = os.path.join(self.fileops.main_dir, ('risk_assessment/models/' + model_pkl_file))

        with open(file_path, 'wb') as file:
            pickle.dump(self.clf_model, file)

        self.test_model()

    def test_model(self):
        y = self.X['exited']
        x = self.X.drop(columns=['corporation', 'exited'])
        # print(x.columns)
        start_time = time.time()
        cross_val_scores = cross_val_score(self.clf_model, x, y, cv=3, scoring='accuracy')
        predictions = cross_val_predict(self.clf_model, x, y)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        end_time = time.time()
        session_time = end_time - start_time

        test_stats = {
            'hyperparameters': self.clf_model.get_params,
            'cross_val_score': cross_val_scores,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time_taken': session_time,
            'comments': 'testing'
        }
        model_stats = pd.DataFrame(test_stats, columns=['hyperparameters', 'cross_val_score', 'precision', 'recall',
                                                        'f1_score', 'time_taken', 'comments'])

        if self.model_summary:
            self.model_summary.append(model_stats)
        else:
            self.model_summary = model_stats

        self.fileops.model_summary_file_ops(self.model_name, self.model_summary, 'save')

    def predict(self, name=None):
        test_set = self.fileops.import_test_dataset(name)
        x = test_set.drop(['corporation'], axis=1)
        exited_prediction = self.clf_model.predict(x)
        test_set['exited_prediction'] = exited_prediction
        print(test_set)
        if not name:
            name = 'alltestdata'
        self.fileops.export_predictions(dataframe=test_set, filename=name)

