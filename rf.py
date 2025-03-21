import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from helpers import *

''' Random forest class '''
class RandomForest:
    def __init__(self, Data):
        self.data = Data

    # Train RF
    def train_rf(self, num_trees=200, load=True):
        print("Training random forest...")
        self.rf = RandomForestRegressor(n_estimators=num_trees)

        # Load from saved model or train from scratch
        if not load:
            self.rf.fit(self.data.xtrain, self.data.ytrain)
            # with open('random_forest_2.pkl', 'wb') as file:
            #     pickle.dump(self.rf, file)
        else:
            self.rf.fit(self.data.xtrain, self.data.ytrain)
            # with open('random_forest.pkl', 'rb') as file:
            #     self.rf = pickle.load(file)

    '''
    There is code duplication for DNN and RF prediction and evaluation
    '''
    # Predict
    def predict_rf(self):
        print("Predicting using random forest...")
        # Predict and scale back
        yrf = self.rf.predict(self.data.xtest).reshape(-1, 1)
        generated_data = np.concatenate((self.data.xtest, yrf), axis=1)
        generated_data = self.data.scaler.inverse_transform(generated_data)
        self.rf_pred = generated_data[:, -1]
    
    # Evaluate
    def evaluate_rf(self):
        print("Evaluating random forest...")
        mape = mean_absolute_percentage_error(self.data.Ytest, self.rf_pred)
        rmspe = root_mean_square_percentage_error(self.data.Ytest, self.rf_pred)
        mae = mean_absolute_error(self.data.Ytest, self.rf_pred)
        rmse = np.sqrt(mean_squared_error(self.data.Ytest, self.rf_pred))
        r2 = r2_score(self.data.Ytest, self.rf_pred)
        if self.data.verbose:
            print('MAPE:', mape)
            print('RMSPE:', rmspe)
            print('MAE:', mae)
            print('RMSE:', rmse)
            print('R2:', r2)
        return [mape, rmspe, mae, rmse, r2]

    # Get feature and permutation importances
    def explain_rf(self, num_repeats=20, num_jobs=4):
        print("Explaining random forest...")
        feats = self.data.features

        # FI
        feature_importances = self.rf.feature_importances_
        fi_std = np.std([tree.feature_importances_ for tree in self.rf.estimators_], axis=0)

        # PI
        perm_importances = permutation_importance(self.rf, self.data.xtest, self.data.ytest, 
                                                  n_repeats=num_repeats, n_jobs=num_jobs)
        pi_std = perm_importances.importances_std

        # Plot importances
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        fi = pd.Series(feature_importances, index=feats, name="FI")
        # if self.data.make_plots:
        #     fi.plot.bar(yerr=fi_std, ax=ax1)
        #     ax1.set_title("Feature importances using MDI")
        #     ax1.set_ylabel("Mean decrease in impurity")

        pi = pd.Series(perm_importances.importances_mean, index=feats, name="PI")
        # if self.data.make_plots:
        #     pi.plot.bar(yerr=pi_std, ax=ax2)
        #     ax2.set_title("Feature importances using permutation on full model")
        #     ax2.set_ylabel("Mean accuracy decrease")
        #     fig.tight_layout()
        #     plt.savefig("explain/rf_explain.png")
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        x = np.arange(len(feats))
        x2 = x + 0.4*np.ones(len(x))
        ax.bar(x, feature_importances, 0.4, color='blue', yerr=fi_std, align='center', alpha=1, ecolor='black', capsize=10, label='MDI')
        ax2.bar(x2, perm_importances.importances_mean, 0.4, color='orange', yerr=pi_std, align='center', alpha=1, ecolor='black', capsize=10, label='Permutation')
        ax.set_ylabel('Mean Decrease in Impurity, []', color='blue')
        ax2.set_ylabel('Mean Accuracy Decrease, []', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(feats)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.tight_layout()
        plt.savefig("figures/rf_explain.png", dpi=300)

        results = pd.concat([fi, pd.Series(fi_std, index=feats, name="FI std"),
                           pi, pd.Series(pi_std, index=feats, name = "PI std")], axis=1)
        return results
    
    # Run all sequentially
    def run_all(self):
        self.train_rf()
        self.predict_rf()
        self.evaluate_rf()
        self.explain_rf()

if __name__=="__main__":
    datafile = 'chf_train_synth.csv'
    xcolumns = [0,1,2,3,4,5]
    ycolumns = [6]
    features = ['D' ,'L' ,'P' ,'G' ,'Tin' ,'Xe']
    chf_data = Data(datafile, xcolumns, ycolumns, features, verbose=True, plot=False, scaler=MinMaxScaler())
    chf_data.preprocess()
    chf_data.traintest_split(test_ratio=0.2)
    chf_synth_rf = RandomForest(chf_data)
    chf_synth_rf.run_all()