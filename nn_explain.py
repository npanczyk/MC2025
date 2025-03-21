import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from functools import partial
import pandas as pd
import numpy as np
import shap
import datetime as dt

def rmspe(ytest, ypred):
    """Generates root mean square percentage error.

    Args:
        ytest (np array): true values
        ypred (np array): predicted values

    Returns:
        np array: array of rmspe scores
    """
    inside = np.square((ytest - ypred)/ytest)
    return np.sqrt(np.mean(inside))*100

def mape(ytest, ypred):
    """Generate mean absolute percentage error.

    Args:
        ytest (np array): true values
        ypred (np array): predicted values

    Returns:
        np array: array of MAPE scores
    """
    return np.mean(np.abs((ytest - ypred)/ytest))*100

def get_chf(cuda=False):
    """
    Gets data for CHF prediction.

    Features:
    - ``D (m)``: Diameter of the test section (:math:`0.002 - 0.016~m`),
    - ``L (m)``: Heated length (:math:`0.07 - 15.0~m`),
    - ``P (kPa)``: Pressure (:math:`100-20000~kPa`),
    - ``G (kg m-2s-1)``: Mass flux (:math:`17.7-7712.0~\\frac{kg}{m^2\\cdot s}`),
    - ``Tin (C)``: Inlet temperature length (:math:`9.0-353.62^\\circ C`),
    - ``Xe (-)``: Outlet equilibrium quality (:math:`-0.445-0.986`),

    Output:
    - ``CHF (kW m-2)``: Critical heat flux (:math:`130.0-13345.0~\\frac{kW}{m^2}`).
    Args:
        cuda (bool, optional): Whether to use gpu or cpu. Defaults to False (cpu).

    Returns:
        dict: a dictionary containing four PyTorch tensors (train_input, train_output, test_input, test_output), y scaler, and feature/output labels
    """

    train_df = pd.read_csv('chf_train_synth.csv')
    test_df = pd.read_csv('chf_test_synth.csv')
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    x_train = train_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  # Input columns (1-6) D, L, P, G, T, Xe
    y_train = train_df.iloc[:, [6]].values  # CHF
    x_test = test_df.iloc[:, [0, 1, 2, 3, 4, 5]].values  
    y_test = test_df.iloc[:, [6]].values

    # Define the Min-Max Scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)

    # Convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)

    # Creating the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': ['D', 'L', 'P', 'G', 'T_in', 'Xe'],
        'output_labels': ['CHF'],
        'y_scaler': scaler_Y
    }
    return dataset

class FNN(nn.Module):
    def __init__(self, input_size, hidden_nodes, output_size, use_dropout=False, dropout_prob=0.5):
        super(FNN, self).__init__()
        layers = []
        # define input layer
        layers.append(nn.Linear(input_size, hidden_nodes[0]))
        layers.append(nn.ReLU())
        # loop through layers in hidden nodes
        for i in range(1, len(hidden_nodes)):
            print(f'i: {i}, hidden_nodes[i-1]: {hidden_nodes[i-1]}, hidden_nodes[i]: {hidden_nodes[i]}')
            layers.append(nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            layers.append(nn.ReLU())
        # add a dropout layer if pymaise does for each model
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob)) 
        # define output layer
        layers.append(nn.Linear(hidden_nodes[-1], output_size))
        # stick all the layers in the model
        self.model = nn.Sequential(*layers)
        self.float()

    def forward(self, x):
        return self.model(x)

def fit_fnn(params, plot=False, save_as=None):
    # define hyperparams
    dataset = params['dataset'](cuda=True)
    input_size = dataset['train_input'].shape[1]
    print(f'Input Size: {input_size}')
    hidden_nodes = params['hidden_nodes']
    output_size = dataset['train_output'].shape[1]
    print(f'Output Size: {output_size}')
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    use_dropout = params['use_dropout']
    dropout_prob = params['dropout_prob']

    # get train and test data from dataset
    train_data = TensorDataset(dataset['train_input'], dataset['train_output'])
    test_data = TensorDataset(dataset['test_input'], dataset['test_output'])

    # write dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # define the model
    model = FNN(input_size, hidden_nodes, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    all_losses = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device).float(), y_train.to(device).float()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        if epoch%10 == 0:
            print(loss)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(all_losses)
        ax.set_ylabel('Training Loss')
        ax.set_xlabel('Epoch')
        plt.savefig(f'figures/fnn_loss_{save_as}.png', dpi=300)

    # # save model
    path = f'models/{save_as}.pt'
    # torch.save(model.state_dict(), path)
    # evaluate model performance
    y_preds, y_tests = get_metrics(model, test_loader, dataset['y_scaler'], dataset, save_as=save_as)
    return model.cpu(), path

def get_metrics(model, test_loader, scaler, dataset, save_as, p=5):
    """This function generates metrics on the original model training call, not with a loaded model.

    Args:
        model (pytorch model object): fnn model
        test_loader (pytorch dataloader object): defined in fit_fnn()
        scaler (sklearn scaler object): contained in dataset dictionary from preprocessing.py
        save_as (str): dataset/model name
        p (int, optional): Precision to save decimals to. Defaults to 5.

    Returns:
        tuple: (predicted y values, test y values)
    """
    model.eval()
    y_preds = []
    y_tests = []
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device).float(), y_test.to(device).float()
            # get prediction
            y_pred = model(x_test)
            # unscale y_test and y_pred
            y_test_unscaled = scaler.inverse_transform(y_test.cpu().detach().numpy())
            y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().detach().numpy())
            # append the tests and predictions to lists
            y_tests.append(y_test_unscaled)
            y_preds.append(y_pred_unscaled)
        y_tests = np.concatenate(y_tests, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
    
    metrics = {
            'OUTPUT': dataset['output_labels'],
            'MAE':[],
            'MAPE':[],
            'MSE':[],
            'RMSE':[],
            'RMSPE':[],
            'R2':[]
        }
    for i in range(len(dataset['output_labels'])):
        # get metrics for each output
        yi_test = y_tests[:,i]
        yi_pred = y_preds[:,i]
        print(f'yi_test: {yi_test.shape}')
        print(f'yi_pred: {yi_pred.shape}')
        metrics['MAE'].append(round(mean_absolute_error(yi_test, yi_pred), p))
        metrics['MAPE'].append(round(mape(yi_test, yi_pred), p))
        metrics['MSE'].append(round(mean_squared_error(yi_test, yi_pred), p))
        metrics['RMSE'].append(round(np.sqrt(mean_squared_error(yi_test, yi_pred)), p))
        metrics['RMSPE'].append(round(rmspe(yi_test, yi_pred), p))
        metrics['R2'].append(round(r2_score(yi_test, yi_pred),p))
    metrics_df = pd.DataFrame.from_dict(metrics)
    # check to see if there 
    if not os.path.exists('results'):
        os.makedirs('results')
    metrics_df.to_csv(f'results/{save_as}_FNN.csv', index=False)

    return y_preds, y_tests

def get_fnn_models(params_dict):
    """Gets metrics and saves trained fnn model objects.

    Args:
        params_dict (dict): keys = model name
                            values = params dictionary
    Returns:
        dict: dictionary where keys are model names and values are a list containing get_dataset functions and paths to pytorch model objects (state_dict) for loading. You can feed this directly into get_fnn_shap()
    """
    for model, params in params_dict.items():
        dataset = params['dataset'](cuda=True)
        X_test = dataset['test_input'].cpu().detach().numpy()
        Y_test = dataset['test_output'].cpu().detach().numpy()
        input_names = dataset['feature_labels']
        output_names = dataset['output_labels']
        save_as = f"{model.upper()}_{str(dt.date.today())}"
        return fit_fnn(params, plot=False, save_as=save_as)[0]

def fnn_shap(model, X_train, X_test, input_names, output_names, save_as, type, n_samples=10):
    """gets feature importances using kernel shap for an fnn

    Args:
        model (pytorch model obj): _description_
        X_train (numpy array): 
        X_test (numpy array): _description_
        input_names (_type_): _description_
        output_names (_type_): _description_
        save_as (_type_): _description_
        k (int, optional): Number of means used for shap.kmeans approximation of training data. Defaults to 50.

    Returns:
        _type_: _description_
    """
    #model_pred = lambda inputs: model(torch.tensor(inputs, dtype=torch.float32)).cpu().detach().numpy()
    X_train_summary = X_train[0:n_samples]
    # pick the type of explainer here
    if type == 'ig':
        explainer = shap.GradientExplainer(model, X_train_summary)
        shap_values = explainer.shap_values(X_test[0:])
    elif type == 'deep':
        explainer = shap.DeepExplainer(model, X_train_summary)
        shap_values = explainer.shap_values(X_test[0:])
    else:
        explainer = shap.ExactExplainer(model, X_train_summary)
        shap_values = explainer(X_test[0:].detach().numpy()).values
    shap_mean = pd.DataFrame(np.abs(shap_values).mean(axis=0),columns=[output_names],index=input_names)
    if not os.path.exists('shap-values'):
        os.makedirs('shap-values')
    path = f'shap-values/{save_as}_.pkl'
    shap_mean.to_pickle(path)
    return path

def get_fnn_shap(model_dict, device):
    """Loads dataset and model and calculates shap values. 

    Args:
        models_dict (dict): key = model name (chf, htgr, etc.),
                            values[0] = get_dataset
                            values[1] = model object path
    """
    shap_paths = {}
    for key, values in model_dict.items():
        dataset = values[0](cuda=False)
        X_train = dataset['train_input'].float()
        X_test = dataset['test_input'].float()
        input_names = dataset['feature_labels']
        output_names = dataset['output_labels']
        save_as =  f"{key.upper()}"
        # feed model args
        model = values[1]
        model.eval()
        for type in ['ig', 'deep']:
            path = fnn_shap(model, X_train, X_test, input_names, output_names, save_as=save_as, type=type)
            shap_paths[f'{key}_{type}'] = path
    return shap_paths

def plot_shap(path, save_as, type='fnn', width=0.2):
    """_summary_

    Args:
        path (str): path to shap values file
        save_as (str): name of run
        type (str, optional): Model type, either fnn or kan. Defaults to 'kan'.
        width (float, optional): Width of plotted bars. Defaults to 0.2.

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    shap_mean = pd.read_pickle(path)
    fig, ax = plt.subplots(figsize=(10,6))
    x_positions = np.arange(len(shap_mean.index))
    output_names = list(shap_mean.columns)
    input_names = list(shap_mean.index)
    for i, col in enumerate(shap_mean.columns):
        # very stupid label thing but it has to do with a list of tuples for fnn
        if type.upper()=='FNN':
            label = output_names[i][0]
        else:
            label = output_names[i]
        ax.bar(x_positions + i*width, shap_mean[col], capsize=4, width=width, label=label)
    ax.set_ylabel("Mean of |SHAP Values|")
    ax.set_yscale('log')
    ax.legend(title='Output')
    n = len(output_names)
    ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    ax.set_xticks(x_positions + (n-1)*width/2)
    ax.set_xticklabels(input_names, rotation=45)
    plt.tight_layout()
    if not os.path.exists('figures/shap'):
        os.makedirs('figures/shap')
    plt.savefig(f'figures/shap/{save_as}.png', dpi=300)
    return fig, ax

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyperparams = {
        'chf': {
            'hidden_nodes' : [231, 138, 267],
            'num_epochs' : 200,
            'batch_size' : 64,
            'learning_rate' : 0.0009311391232267503,
            'use_dropout': True,
            'dropout_prob': 0.4995897609454529,
            'dataset': get_chf
        }       
    }

    # #  train FNN models and get metrics
    # model = get_fnn_models(hyperparams)

    # model_dict = {'chf': [get_chf, model]}

    # # get shap values from FNN models
    # shap_paths = get_fnn_shap(model_dict, device)
    # print(shap_paths)

    shap_path_dict = {'chf_ig': 'shap-values/CHF_.pkl', 'chf_deep': 'shap-values/CHF_.pkl'}
    # # uncomment to plot shap values
    for model, path in shap_path_dict.items():
        plot_shap(path, save_as=f'{model}_fnn', type='fnn', width=0.2)

    # ## uncomment to print shap values
    # for model, path in shap_path_dict.items():
    #     print_shap(path, save_as=f'{model}', type='fnn')
 