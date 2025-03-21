import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

''' Data class '''
class Data:
    # change verbose and plot, or add other helpful attributes, as necessary
    def __init__(self, datafile, xcolumns, ycolumns, features=None, verbose=True, plot=False, scaler=MinMaxScaler()):
        self.datafile = datafile # training AND testing data
        self.xcolumns = xcolumns # columns with X 
        self.ycolumns = ycolumns # columns with Y
        if features == None:
            self.features = np.arange(columns.shape) # generic 0, 1, 2, ...
        else:
            self.features = features
        self.scaler = scaler
        self.verbose = verbose
        self.make_plots = plot

    # Preprocess data
    def preprocess(self):
        print("Preprocessing data...")
        # Load train and test data, store self.Y
        self.data = pd.read_csv(self.datafile).values
        self.data = self.data[:, self.xcolumns + self.ycolumns]
        self.Y = self.data[:, self.ycolumns] # unscaled

        # Scale data, store into self.x, self.y
        data_scaled = self.scaler.fit_transform(self.data)
        self.x = data_scaled[:, self.xcolumns]
        self.y = data_scaled[:, self.ycolumns]
    
    '''
    There is some poor design beyond here, for stratified sampling:
        I process the data to obtain all five of self.Y and self.[x|y][train|test] 
            (as well as the unused self.test_indices) as attributes of the data class,
            then I run the training, prediction, and evaluation in sampler.py
        The functions should have (nearly) identical names as their sampler.py counterparts
    '''

    # Train-test split based on random_state
    def traintest_split(self, test_ratio=0.2, random_state=None):
        self.rng = random_state
        indices = np.arange(self.x.shape[0])
        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, random_state=self.rng)
        self.Ytest = self.Y[self.test_indices]

    def stratify(self, target=0, test_ratio=0.2, n_splits=5, random_state=None):
        # if n_splits is too large, some bins may only contain one value, which raises an error
        self.rng = random_state
        indices = np.arange(self.x.shape[0])

        xdf = pd.DataFrame(self.x)
        xdf["binned"] = pd.qcut(xdf[target], q=n_splits, labels=False)
        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, stratify=xdf["binned"])
        self.Ytest = self.Y[self.test_indices]
    
    def stratify_target(self, test_ratio=0.2, n_splits=10):
        # again, if n_splits is too large, some bins may only contain one value, which raises an error
        indices = np.arange(self.x.shape[0])

        ydf = pd.DataFrame(self.y)
        ydf["binned"] = pd.qcut(ydf[0], q=n_splits, labels=False)
        
        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, stratify=ydf["binned"])
        self.Ytest = self.Y[self.test_indices]

    def stratify_on_two(self, target=[0, 1], test_ratio=0.2, n_splits=5):
        # if n_splits is too large, some bins may only contain one value, which raises an error

        indices = np.arange(self.x.shape[0])

        xdf = pd.DataFrame(self.x)
        xdf["bin0"] = pd.qcut(xdf[target[0]], q=n_splits, labels=False)
        xdf["bin1"] = pd.qcut(xdf[target[1]], q=n_splits, labels=False)
        xdf["binned"] = xdf["bin0"] + n_splits * xdf["bin1"]

        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, stratify=xdf["binned"])
        self.Ytest = self.Y[self.test_indices]
    
    def stratify_on_all(self, test_ratio=0.2, n_splits=2):
        # if n_splits is too large, some bins may only contain one value, which raises an error

        indices = np.arange(self.x.shape[0])

        xdf = pd.DataFrame(self.x)
        for i in range(5):
            xdf[f"bin{i}"] = pd.qcut(xdf[i], q=n_splits, labels=False)
        xdf["binned"] = xdf["bin0"] + n_splits * (xdf["bin1"] +  n_splits * (xdf["bin2"] + 
                                    n_splits * (xdf["bin3"] +  n_splits * xdf["bin4"])))

        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, stratify=xdf["binned"])
        self.Ytest = self.Y[self.test_indices]

    def stratify_by_cluster(self, test_ratio=0.2, n_splits=10):
        indices = np.arange(self.x.shape[0])

        kmeans = KMeans(n_clusters=n_splits, random_state=42)

        xdf = pd.DataFrame(self.x)
        xdf["clustered"] = kmeans.fit_predict(xdf)
        self.xtrain, self.xtest, self.ytrain, self.ytest, _, self.test_indices = train_test_split(
            self.x, self.y, indices, test_size=test_ratio, stratify=xdf["clustered"])
        self.Ytest = self.Y[self.test_indices]

    def extrapolate(self, target=0, test_ratio=0.2, high=False, random_state=None):
        # if n_splits is too large, some bins may only contain one value, which raises an error

        self.rng = random_state
        #print("random state:", self.rng)
        xlen = self.x.shape[0]
        test_size = (int)(xlen*test_ratio)
        indices = np.arange(xlen)

        # Obtain feature to extrapolate on
        if target == -1:
            target_slice = self.y
        else:
            target_slice = self.x[:, target]
        
        # Get highest or lowest values
        sorted_slice = np.argsort(target_slice)
        if high:
            self.target_indices = sorted_slice[-test_size:]
        else:
            self.target_indices = sorted_slice[:test_size]
        complement_indices = np.setdiff1d(indices, self.target_indices)

        self.xtrain = self.x[complement_indices]
        self.xtest = self.x[self.target_indices]
        self.ytrain = self.y[complement_indices]
        self.ytest = self.y[self.target_indices]
        self.Ytest = self.Y[self.target_indices]
        #print(self.Ytest.shape)
        #print(self.xtrain[:, target].max(), self.xtrain[:, target].min())

# An actual helper: RMSPE
def root_mean_square_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

#################################### TESTING ################################
# print( 'START HERE')
# datafile = 'chf_train_synth.'
# df = pd.read_excel(datafile).values
# print(df)
# sample = Data('datasets/mini_1000_chf_full.xlsx', [1,2,3,4,8,7], ["D", "L", "P", "G", "T"])
# sample.preprocess()