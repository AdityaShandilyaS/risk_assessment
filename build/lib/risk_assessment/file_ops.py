import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path
import random
import shutil
import pickle


class FileSystem:
    def __init__(self):
        self.curr_dir = os.path.dirname(__file__)
        self.main_dir = Path(self.curr_dir).parents[0]

    def import_training_dataset(self, file_name=None):
        folder_path = os.path.join(self.main_dir, 'risk_assessment/inputdata')
        session_start = time.time()
        # Get a list of all files in the folder
        file_list = []
        if file_name:
            file_list.append(file_name)
        else:
            file_list = os.listdir(folder_path)
        combined_df = pd.DataFrame()
        # Iterate through each file in the folder
        for filename in file_list:
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(file_path)
                # Process the DataFrame here
                if "corporation" and "exited" in df.columns:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    print('file {} with shape {} included into dataframe'.format(filename, df.shape))
                else:
                    print('file {} is inappropriate for risk assessment model training: primary key columns missing'
                          .format(filename))

        session_end = time.time()
        print('training dataframe imported with shape {}.\nSession time {}'.format(combined_df.shape,
                                                                               (session_end - session_start)))

        return combined_df

    def import_test_dataset(self, file_name=None):
        folder_path = os.path.join(self.main_dir, 'risk_assessment/testdata')
        session_start = time.time()
        # Get a list of all files in the folder
        file_list = []
        if file_name:
            file_list.append(file_name)
        else:
            file_list = os.listdir(folder_path)
        combined_df = pd.DataFrame()
        # Iterate through each file in the folder
        for filename in file_list:
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(file_path)
                # Process the DataFrame here
                if "corporation" in df.columns:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    print('file {} with shape {} included into dataframe'.format(filename, df.shape))
                else:
                    print('file {} is inappropriate for risk assessment predictions: primary key column missing'
                          .format(filename))

        session_end = time.time()
        print('test dataframe imported with shape {}.\nSession time {}'.format(combined_df.shape,
                                                                                   (session_end - session_start)))
        return combined_df

    def load_cleaned_dataset(self):
        folder_path = os.path.join(self.main_dir, 'risk_assessment/cleaneddata')
        file_list = os.listdir(folder_path)
        df = pd.DataFrame()
        for filename in file_list:
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(folder_path, filename)
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(file_path)

        return df

    def create_csv(self, dataframe, filename, dest_folder):
        timestamp = int(time.time())
        file_path = os.path.join(self.main_dir, dest_folder)
        index = True
        if dest_folder == 'risk_assessment/cleaneddata':
            file = os.path.join(file_path, (filename + '.csv'))
            index = False
        else:
            file = os.path.join(file_path, (str(timestamp) + '_' + filename + '.csv'))
        dataframe.to_csv(file, header=True, index=index)
        print('Created {} in {}'.format(filename, dest_folder))

    def copy_file(self, filename, filetype):
        if filetype == 'training':
            folder_name = 'risk_assessment/inputdata'
        elif filetype == 'prediction':
            folder_name = 'risk_assessment/testdata'
        else:
            folder_name = 'risk_assessment/evaluatorsdata'
        dest_path = os.path.join(self.main_dir, folder_name)
        shutil.copy(filename, dest_path)
        print('Copied {} to {} directory'.format(filename, folder_name))

    def model_summary_file_ops(self, model_name='default', dataframe=None, operation='load'):
        if operation == 'load':
            folder_path = os.path.join(self.main_dir, 'risk_assessment/models')
            file_list = os.listdir(folder_path)
            for filename in file_list:
                if filename == model_name:  # Check if the file is of the model
                    file_path = os.path.join(folder_path, filename)
                    # Read the CSV file into a pandas DataFrame
                    dataframe = pd.read_csv(file_path)

        if operation == 'save':
            folder_path = os.path.join(self.main_dir, 'risk_assessment/models')
            file = os.path.join(folder_path, (model_name + '.csv'))
            dataframe.to_csv(file, mode='a', header=True, index=True)

        return dataframe

    def export_predictions(self, dataframe, filename):
        timestamp = int(time.time())
        file_path = os.path.join(self.main_dir, 'risk_assessment/predictions')
        file = os.path.join(file_path, (str(timestamp) + '_' + filename))
        dataframe.to_csv(file, header=True, index=True)
        print('Done. Find predictions in the predictions folder')


if __name__ == '__main__':
    print(os.getcwd())
    file = 'file_ops.py'
    print(Path.cwd())
    print(os.path.dirname(__file__))
    print(Path(os.path.dirname(__file__)).parents[0])
