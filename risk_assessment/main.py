import file_ops
import data_ops
import argparse
import os

fs = file_ops.FileSystem()


# filename must include .csv but model must NOT include .pkl
def predict_risk(filename=None, modelname=None):
    file_path = os.path.join(fs.main_dir, filename)
    if os.path.exists(file_path):
        fs.copy_file(filename, filetype='prediction')
    ml = data_ops.MLModule(modelname)
    ml.load_model(modelname)
    ml.predict(filename)


def train_model(filename=None, modelname=None, splitter=None, max_depth=None, min_samples_split=None):
    if filename:
        file_path = os.path.join(fs.main_dir, filename)
        if os.path.exists(file_path):
            fs.copy_file(filename, filetype='training')
    ml = data_ops.MLModule(modelname)
    ml.preprocess_data()
    ml.analyse_dataset()
    ml.train_decision_tree_model(splitter, max_depth, min_samples_split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attrition risk predictor')
    # parser.add_argument('--predict', '-p', help='Predict attrition risk with model')
    subparsers = parser.add_subparsers(dest='function')
    predict_parser = subparsers.add_parser('predict', help='predict attrition risk')
    predict_parser.add_argument('-d', '--dataset', help='enter csv file name in directory', default='dataset3.csv')
    predict_parser.add_argument('-m', '--model', help='enter model to use for predictions', default=None)

    train_parser = subparsers.add_parser('train', help='train a model for attrition risk prediction')
    train_parser.add_argument('-d', '--dataset', help='enter csv file name in directory')
    train_parser.add_argument('-m', '--model', help='enter model to train', default='')
    train_parser.add_argument('-s', '--splitter', help='choose type of splitting', choices=['best', 'random'],
                              default='best')
    train_parser.add_argument('-md', '--max_depth', help='enter max depth of the tree branching', default=None)
    train_parser.add_argument('-mss', '--min_samples_split', help='enter min samples that shall cause split',
                              default=2)
    args = parser.parse_args()

    if args.function == 'predict':
        predict_risk(args.dataset)
    if args.function == 'train':
        train_model(args.dataset, args.model, args.splitter, args.max_depth, args.min_samples_split)
