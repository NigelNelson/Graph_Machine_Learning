import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#@title Imports 
import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject, Extract_subject_from_BDF, load_events
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time,  Filter_by_condition


mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning )

from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn import metrics
import tensorflow as tf
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import torch
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, Linear, GATv2Conv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.metrics import precision_recall_fscore_support


# ## Hyperparameters

experiment_name = '8layer_1head_GAT'

save_path = f'./saved_models/{experiment_name}.pth'
training_fig_path = f'./Experiment_outputs/{experiment_name}/training_fig.png'
encoding_fig_path = f'./Experiment_outputs/{experiment_name}/encoding_fig.png'

os.mkdir(f'./Experiment_outputs/{experiment_name}')

num_epochs = 200
hidden_dim = 256

feature_dim = 512

heads = 1


# The root dir have to point to the folder that cointains the database
root_dir = "/data/datasets/inner_speech/ds003626/"
# Data Type
datatype = "EEG"
# Sampling rate
fs = 256
# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5
# Subject number
NS = 1
BN = 1

X, Y = Extract_subject_from_BDF(root_dir, NS, BN)

channels = X.ch_names
chan_idxs = list(zip(channels, range(len(channels))))[:128]
chan_map = {x[0]: x[1] for x in chan_idxs}

edge_index = [
    [chan_map['C17'], chan_map['C16']],
    [chan_map['C17'], chan_map['C15']],
    [chan_map['C17'], chan_map['C18']],
    [chan_map['C17'], chan_map['C28']],
    [chan_map['C17'], chan_map['C29']],
    
    [chan_map['C16'], chan_map['C18']],
    [chan_map['C16'], chan_map['C15']],
    [chan_map['C16'], chan_map['C9']],
    [chan_map['C16'], chan_map['C8']],
    
    [chan_map['C8'], chan_map['C15']],
    [chan_map['C8'], chan_map['C9']],
    [chan_map['C8'], chan_map['C6']],
    [chan_map['C8'], chan_map['C7']],
    
    [chan_map['C7'], chan_map['C9']],
    [chan_map['C7'], chan_map['C6']],
    [chan_map['C7'], chan_map['B28']],
    [chan_map['C7'], chan_map['B27']],
    
    [chan_map['B27'], chan_map['C6']],
    [chan_map['B27'], chan_map['B28']],
    [chan_map['B27'], chan_map['B25']],
    [chan_map['B27'], chan_map['B26']],
    
    [chan_map['B26'], chan_map['B28']],
    [chan_map['B26'], chan_map['B25']],
    [chan_map['B26'], chan_map['B15']],
    [chan_map['B26'], chan_map['B14']],
    
    [chan_map['B14'], chan_map['B25']],
    [chan_map['B14'], chan_map['B15']],
    [chan_map['B14'], chan_map['B12']],
    [chan_map['B14'], chan_map['B11']],
    
    [chan_map['B11'], chan_map['B15']],
    [chan_map['B11'], chan_map['B12']],
    [chan_map['B11'], chan_map['B6']],
    [chan_map['B11'], chan_map['B7']],
    [chan_map['B11'], chan_map['B10']],
    [chan_map['B11'], chan_map['B8']],
    
    [chan_map['B7'], chan_map['B12']],
    [chan_map['B7'], chan_map['B6']],
    [chan_map['B7'], chan_map['A29']],
    [chan_map['B7'], chan_map['A28']],
    [chan_map['B7'], chan_map['A27']],
    [chan_map['B7'], chan_map['B8']],
    [chan_map['B7'], chan_map['B10']],
    
    [chan_map['A28'], chan_map['B6']],
    [chan_map['A28'], chan_map['A29']],
    [chan_map['A28'], chan_map['A22']],
    [chan_map['A28'], chan_map['A23']],
    [chan_map['A28'], chan_map['A24']],
    [chan_map['A28'], chan_map['A27']],
    [chan_map['A28'], chan_map['B8']],
    
    [chan_map['A23'], chan_map['A29']],
    [chan_map['A23'], chan_map['A22']],
    [chan_map['A23'], chan_map['A16']],
    [chan_map['A23'], chan_map['A15']],
    [chan_map['A23'], chan_map['A14']],
    [chan_map['A23'], chan_map['A24']],
    [chan_map['A23'], chan_map['A27']],
    
    [chan_map['A10'], chan_map['A16']],
    [chan_map['A10'], chan_map['A9']],
    [chan_map['A10'], chan_map['D30']],
    [chan_map['A10'], chan_map['D31']],
    [chan_map['A10'], chan_map['D32']],
    [chan_map['A10'], chan_map['A11']],
    [chan_map['A10'], chan_map['A14']],
    
    [chan_map['D31'], chan_map['A9']],
    [chan_map['D31'], chan_map['D30']],
    [chan_map['D31'], chan_map['D25']],
    [chan_map['D31'], chan_map['D24']],
    [chan_map['D31'], chan_map['D32']],
    [chan_map['D31'], chan_map['A11']],
    
    [chan_map['D24'], chan_map['D30']],
    [chan_map['D24'], chan_map['D25']],
    [chan_map['D24'], chan_map['D22']],
    [chan_map['D24'], chan_map['D23']],
    
    [chan_map['D23'], chan_map['D25']],
    [chan_map['D23'], chan_map['D22']],
    [chan_map['D23'], chan_map['D9']],
    [chan_map['D23'], chan_map['D8']],
    
    [chan_map['D8'], chan_map['D22']],
    [chan_map['D8'], chan_map['D9']],
    [chan_map['D8'], chan_map['D6']],
    [chan_map['D8'], chan_map['D7']],
    
    [chan_map['D7'], chan_map['D9']],
    [chan_map['D7'], chan_map['D6']],
    [chan_map['D7'], chan_map['C31']],
    [chan_map['D7'], chan_map['C30']],
    
    [chan_map['C30'], chan_map['D6']],
    [chan_map['C30'], chan_map['C31']],
    [chan_map['C30'], chan_map['C28']],
    [chan_map['C30'], chan_map['C29']],
    
    [chan_map['C29'], chan_map['C31']],
    [chan_map['C29'], chan_map['C28']],
    [chan_map['C29'], chan_map['C18']],
    
    [chan_map['C19'], chan_map['C18']],
    [chan_map['C19'], chan_map['C15']],
    [chan_map['C19'], chan_map['C14']],
    [chan_map['C19'], chan_map['C13']],
    [chan_map['C19'], chan_map['C20']],
    [chan_map['C19'], chan_map['C26']],
    [chan_map['C19'], chan_map['C27']],
    [chan_map['C19'], chan_map['C28']],
    
    [chan_map['C14'], chan_map['C15']],
    [chan_map['C14'], chan_map['C9']],
    [chan_map['C14'], chan_map['C10']],
    [chan_map['C14'], chan_map['C13']],
    [chan_map['C14'], chan_map['C20']],
    
    [chan_map['C10'], chan_map['C15']],
    [chan_map['C10'], chan_map['C9']],
    [chan_map['C10'], chan_map['C6']],
    [chan_map['C10'], chan_map['C5']],
    [chan_map['C10'], chan_map['C4']],
    [chan_map['C10'], chan_map['C13']],
    
    [chan_map['C5'], chan_map['C6']],
    [chan_map['C5'], chan_map['B28']],
    [chan_map['C5'], chan_map['B29']],
    [chan_map['C5'], chan_map['B30']],
    [chan_map['C5'], chan_map['C4']],
    
    [chan_map['B29'], chan_map['C6']],
    [chan_map['B29'], chan_map['B28']],
    [chan_map['B29'], chan_map['B25']],
    [chan_map['B29'], chan_map['B24']],
    [chan_map['B29'], chan_map['B23']],
    [chan_map['B29'], chan_map['B30']],
    
    [chan_map['B24'], chan_map['B28']],
    [chan_map['B24'], chan_map['B25']],
    [chan_map['B24'], chan_map['B15']],
    [chan_map['B24'], chan_map['B16']],
    [chan_map['B24'], chan_map['B17']],
    [chan_map['B24'], chan_map['B23']],
    [chan_map['B24'], chan_map['B30']],
    
    [chan_map['B16'], chan_map['B25']],
    [chan_map['B16'], chan_map['B15']],
    [chan_map['B16'], chan_map['B12']],
    [chan_map['B16'], chan_map['B13']],
    [chan_map['B16'], chan_map['B17']],
    [chan_map['B16'], chan_map['B23']],
    
    [chan_map['B13'], chan_map['B15']],
    [chan_map['B13'], chan_map['B12']],
    [chan_map['B13'], chan_map['B6']],
    [chan_map['B13'], chan_map['B5']],
    [chan_map['B13'], chan_map['B4']],
    [chan_map['B13'], chan_map['B17']],
    
    [chan_map['B5'], chan_map['B12']],
    [chan_map['B5'], chan_map['B6']],
    [chan_map['B5'], chan_map['A29']],
    [chan_map['B5'], chan_map['A30']],
    [chan_map['B5'], chan_map['A31']],
    [chan_map['B5'], chan_map['B4']],
    
    [chan_map['A30'], chan_map['B6']],
    [chan_map['A30'], chan_map['A29']],
    [chan_map['A30'], chan_map['A22']],
    [chan_map['A30'], chan_map['A21']],
    [chan_map['A30'], chan_map['A20']],
    [chan_map['A30'], chan_map['A31']],
    
    [chan_map['A21'], chan_map['A29']],
    [chan_map['A21'], chan_map['A22']],
    [chan_map['A21'], chan_map['A16']],
    [chan_map['A21'], chan_map['A17']],
    [chan_map['A21'], chan_map['A18']],
    [chan_map['A21'], chan_map['A20']],
    [chan_map['A21'], chan_map['A31']],
    
    [chan_map['A17'], chan_map['A22']],
    [chan_map['A17'], chan_map['A16']],
    [chan_map['A17'], chan_map['A9']],
    [chan_map['A17'], chan_map['A8']],
    [chan_map['A17'], chan_map['A18']],
    [chan_map['A17'], chan_map['A20']],
    
    [chan_map['A8'], chan_map['A16']],
    [chan_map['A8'], chan_map['A9']],
    [chan_map['A8'], chan_map['D30']],
    [chan_map['A8'], chan_map['D29']],
    [chan_map['A8'], chan_map['A7']],
    [chan_map['A8'], chan_map['A18']],
    
    [chan_map['D29'], chan_map['A9']],
    [chan_map['D29'], chan_map['D30']],
    [chan_map['D29'], chan_map['D25']],
    [chan_map['D29'], chan_map['D26']],
    [chan_map['D29'], chan_map['D27']],
    [chan_map['D29'], chan_map['A7']],
    
    [chan_map['D26'], chan_map['D30']],
    [chan_map['D26'], chan_map['D25']],
    [chan_map['D26'], chan_map['D22']],
    [chan_map['D26'], chan_map['D21']],
    [chan_map['D26'], chan_map['D20']],
    [chan_map['D26'], chan_map['D27']],
    
    [chan_map['D21'], chan_map['D25']],
    [chan_map['D21'], chan_map['D22']],
    [chan_map['D21'], chan_map['D9']],
    [chan_map['D21'], chan_map['D10']],
    [chan_map['D21'], chan_map['D11']],
    [chan_map['D21'], chan_map['D20']],
    [chan_map['D21'], chan_map['D27']],
    
    [chan_map['D10'], chan_map['D22']],
    [chan_map['D10'], chan_map['D9']],
    [chan_map['D10'], chan_map['D6']],
    [chan_map['D10'], chan_map['D5']],
    [chan_map['D10'], chan_map['D11']],
    [chan_map['D10'], chan_map['D20']],
    
    [chan_map['D5'], chan_map['D9']],
    [chan_map['D5'], chan_map['D6']],
    [chan_map['D5'], chan_map['C31']],
    [chan_map['D5'], chan_map['C32']],
    [chan_map['D5'], chan_map['D4']],
    [chan_map['D5'], chan_map['D11']],
    
    [chan_map['C32'], chan_map['D6']],
    [chan_map['C32'], chan_map['C31']],
    [chan_map['C32'], chan_map['C28']],
    [chan_map['C32'], chan_map['C27']],
    [chan_map['C32'], chan_map['C26']],
    [chan_map['C32'], chan_map['D4']],
    
    [chan_map['C27'], chan_map['C31']],
    [chan_map['C27'], chan_map['C28']],
    [chan_map['C27'], chan_map['C18']],
    [chan_map['C27'], chan_map['C20']],
    [chan_map['C27'], chan_map['C26']],
    
    [chan_map['C21'], chan_map['C20']],
    [chan_map['C21'], chan_map['C13']],
    [chan_map['C21'], chan_map['C12']],
    [chan_map['C21'], chan_map['C11']],
    [chan_map['C21'], chan_map['C22']],
    [chan_map['C21'], chan_map['C24']],
    [chan_map['C21'], chan_map['C25']],
    [chan_map['C21'], chan_map['C26']],
    
    [chan_map['C12'], chan_map['C20']],
    [chan_map['C12'], chan_map['C13']],
    [chan_map['C12'], chan_map['C4']],
    [chan_map['C12'], chan_map['C3']],
    [chan_map['C12'], chan_map['C11']],
    [chan_map['C12'], chan_map['C22']],
    
    [chan_map['C3'], chan_map['C13']],
    [chan_map['C3'], chan_map['C4']],
    [chan_map['C3'], chan_map['B30']],
    [chan_map['C3'], chan_map['B31']],
    [chan_map['C3'], chan_map['B32']],
    [chan_map['C3'], chan_map['C11']],
    
    [chan_map['B31'], chan_map['C4']],
    [chan_map['B31'], chan_map['B30']],
    [chan_map['B31'], chan_map['B23']],
    [chan_map['B31'], chan_map['B22']],
    [chan_map['B31'], chan_map['B21']],
    [chan_map['B31'], chan_map['B32']],
    
    [chan_map['B22'], chan_map['B30']],
    [chan_map['B22'], chan_map['B23']],
    [chan_map['B22'], chan_map['B17']],
    [chan_map['B22'], chan_map['B18']],
    [chan_map['B22'], chan_map['B19']],
    [chan_map['B22'], chan_map['B21']],
    [chan_map['B22'], chan_map['B32']],
    
    [chan_map['B18'], chan_map['B23']],
    [chan_map['B18'], chan_map['B17']],
    [chan_map['B18'], chan_map['B4']],
    [chan_map['B18'], chan_map['B3']],
    [chan_map['B18'], chan_map['B19']],
    [chan_map['B18'], chan_map['B21']],
    
    [chan_map['B3'], chan_map['B19']],
    [chan_map['B3'], chan_map['B17']],
    [chan_map['B3'], chan_map['B4']],
    [chan_map['B3'], chan_map['A31']],
    [chan_map['B3'], chan_map['A32']],
    [chan_map['B3'], chan_map['B19']],
    
    [chan_map['A32'], chan_map['B4']],
    [chan_map['A32'], chan_map['A31']],
    [chan_map['A32'], chan_map['A20']],
    [chan_map['A32'], chan_map['A19']],
    [chan_map['A32'], chan_map['A4']],
    [chan_map['A32'], chan_map['B19']],
    
    [chan_map['A19'], chan_map['A31']],
    [chan_map['A19'], chan_map['A20']],
    [chan_map['A19'], chan_map['A18']],
    [chan_map['A19'], chan_map['A5']],
    [chan_map['A19'], chan_map['A4']],
    
    [chan_map['A5'], chan_map['A20']],
    [chan_map['A5'], chan_map['A18']],
    [chan_map['A5'], chan_map['A7']],
    [chan_map['A5'], chan_map['A6']],
    [chan_map['A5'], chan_map['A4']],
    
    [chan_map['A6'], chan_map['A18']],
    [chan_map['A6'], chan_map['A7']],
    [chan_map['A6'], chan_map['D27']],
    [chan_map['A6'], chan_map['D28']],
    [chan_map['A6'], chan_map['D17']],
    
    [chan_map['D28'], chan_map['A7']],
    [chan_map['D28'], chan_map['D27']],
    [chan_map['D28'], chan_map['D20']],
    [chan_map['D28'], chan_map['D19']],
    [chan_map['D28'], chan_map['D18']],
    [chan_map['D28'], chan_map['D17']],
    
    [chan_map['D19'], chan_map['D27']],
    [chan_map['D19'], chan_map['D20']],
    [chan_map['D19'], chan_map['D11']],
    [chan_map['D19'], chan_map['D12']],
    [chan_map['D19'], chan_map['D13']],
    [chan_map['D19'], chan_map['D18']],
    [chan_map['D19'], chan_map['D17']],
    
    [chan_map['D12'], chan_map['D20']],
    [chan_map['D12'], chan_map['D11']],
    [chan_map['D12'], chan_map['D4']],
    [chan_map['D12'], chan_map['D3']],
    [chan_map['D12'], chan_map['D13']],
    [chan_map['D12'], chan_map['D18']],
    
    [chan_map['D3'], chan_map['D11']],
    [chan_map['D3'], chan_map['D4']],
    [chan_map['D3'], chan_map['C26']],
    [chan_map['D3'], chan_map['C25']],
    [chan_map['D3'], chan_map['C24']],
    [chan_map['D3'], chan_map['D13']],
    
    [chan_map['C25'], chan_map['D4']],
    [chan_map['C25'], chan_map['C26']],
    [chan_map['C25'], chan_map['C20']],
    [chan_map['C25'], chan_map['C22']],
    [chan_map['C25'], chan_map['C24']],
    [chan_map['C25'], chan_map['D13']],
    
    [chan_map['C23'], chan_map['C22']],
    [chan_map['C23'], chan_map['C11']],
    [chan_map['C23'], chan_map['C2']],
    [chan_map['C23'], chan_map['C1']],
    [chan_map['C23'], chan_map['D1']],
    [chan_map['C23'], chan_map['D2']],
    [chan_map['C23'], chan_map['D24']],
    
    [chan_map['C2'], chan_map['C22']],
    [chan_map['C2'], chan_map['C11']],
    [chan_map['C2'], chan_map['B32']],
    [chan_map['C2'], chan_map['B21']],
    [chan_map['C2'], chan_map['B20']],
    [chan_map['C2'], chan_map['B1']],
    [chan_map['C2'], chan_map['C1']],
    
    [chan_map['B20'], chan_map['B32']],
    [chan_map['B20'], chan_map['B21']],
    [chan_map['B20'], chan_map['B19']],
    [chan_map['B20'], chan_map['B2']],
    [chan_map['B20'], chan_map['B1']],
    [chan_map['B20'], chan_map['C1']],
    
    [chan_map['B2'], chan_map['B1']],
    [chan_map['B2'], chan_map['B21']],
    [chan_map['B2'], chan_map['B19']],
    [chan_map['B2'], chan_map['A4']],
    [chan_map['B2'], chan_map['A2']],
    [chan_map['B2'], chan_map['C1']],
    [chan_map['B2'], chan_map['B3']],
    [chan_map['B2'], chan_map['A32']],
    
    [chan_map['A3'], chan_map['A2']],
    [chan_map['A3'], chan_map['A4']],
    [chan_map['A3'], chan_map['D16']],
    [chan_map['A3'], chan_map['B3']],
    [chan_map['A3'], chan_map['A32']],
    [chan_map['A3'], chan_map['A5']],
    [chan_map['A3'], chan_map['A6']],
    [chan_map['A3'], chan_map['D15']],
    [chan_map['A3'], chan_map['B1']],
    
    [chan_map['D16'], chan_map['A2']],
    [chan_map['D16'], chan_map['A4']],
    [chan_map['D16'], chan_map['A6']],
    [chan_map['D16'], chan_map['A5']],
    [chan_map['D16'], chan_map['D17']],
    [chan_map['D16'], chan_map['D14']],
    [chan_map['D16'], chan_map['D15']],
    
    [chan_map['D14'], chan_map['D15']],
    [chan_map['D14'], chan_map['D17']],
    [chan_map['D14'], chan_map['D18']],
    [chan_map['D14'], chan_map['D13']],
    [chan_map['D14'], chan_map['D2']],
    [chan_map['D14'], chan_map['D1']],
    [chan_map['D14'], chan_map['D15']],
    
    [chan_map['D2'], chan_map['D18']],
    [chan_map['D2'], chan_map['D13']],
    [chan_map['D2'], chan_map['D24']],
    [chan_map['D2'], chan_map['D1']],
    [chan_map['D2'], chan_map['D15']],
    [chan_map['D2'], chan_map['D22']],
    
    [chan_map['D1'], chan_map['C1']],
    [chan_map['C1'], chan_map['B1']],
    [chan_map['B1'], chan_map['A2']],
    [chan_map['A2'], chan_map['D15']],
    [chan_map['D15'], chan_map['D1']],
    
    [chan_map['A1'], chan_map['D1']],
    [chan_map['A1'], chan_map['C1']],
    [chan_map['A1'], chan_map['B1']],
    [chan_map['A1'], chan_map['A2']],
    [chan_map['A1'], chan_map['D15']],
    
    [chan_map['B9'], chan_map['B10']],
    [chan_map['B9'], chan_map['B8']],
    [chan_map['B9'], chan_map['A27']],
    [chan_map['B9'], chan_map['A26']],
    
    [chan_map['A26'], chan_map['B8']],
    [chan_map['A26'], chan_map['A27']],
    [chan_map['A26'], chan_map['A24']],
    [chan_map['A26'], chan_map['A25']],
    
    [chan_map['A25'], chan_map['A27']],
    [chan_map['A25'], chan_map['A24']],
    [chan_map['A25'], chan_map['A14']],
    [chan_map['A25'], chan_map['A13']],
    
    [chan_map['A13'], chan_map['A24']],
    [chan_map['A13'], chan_map['A14']],
    [chan_map['A13'], chan_map['A11']],
    [chan_map['A13'], chan_map['A12']],
    
    [chan_map['A12'], chan_map['A14']],
    [chan_map['A12'], chan_map['A11']],
    [chan_map['A12'], chan_map['D32']]
]

reversed_index = list(map(lambda x: [x[1], x[0]], edge_index))
edge_index = edge_index + reversed_index

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load all trials for a sigle subject
X, Y = Extract_data_from_subject(root_dir, NS, datatype)

# Cut usefull time. i.e action interval
X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)

X, Y = Filter_by_condition(X, Y, "INNER")

y_labels = Y[:,1]

class GAT(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # GAT layers
        self.initial_gat = GATv2Conv(feature_dim, hidden_dim, heads)
        self.gat1 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat3 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat4 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat5 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat6 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)
        self.gat7 = GATv2Conv(hidden_dim*heads, hidden_dim, heads)

    
        self.out = Linear(feature_dim, num_classes)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_gat(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.gat1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat3(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat4(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat5(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat6(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.gat7(hidden, edge_index)
        hidden = torch.tanh(hidden)

        hidden = global_max_pool(x, batch_index) # add pooling layer here

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden

model = GAT(feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=4)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print()

def train(data):
    # Enumerate over the data
    correct = 0
    for batch in data:
        # Use GPU
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 

        pred_cls = pred.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred_cls == batch.y).sum())
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)     
        loss.backward()  
        # Update using the gradients
        optimizer.step()   
        
    acc = correct / len(data.dataset)
    return loss, embedding, acc

def test(data_loader):

    correct = 0
    for batch in data_loader:  # Iterate in batches over the training/test dataset.
      # Passing the node features and the connection info
        batch.to(device)
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
        pred_cls = pred.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred_cls == batch.y).sum())  # Check against ground-truth labels.
        loss = loss_fn(pred, batch.y) 
        acc = correct / len(data_loader.dataset)
    return loss, acc  # Derive ratio of correct predictions.

def Average(lst):
    return sum(lst) / len(lst)

def eval(model, data_loader):
    model.eval()
    precisions =[]
    f1s = []
    recalls = []

    correct = 0
    for data in data_loader:  # Iterate in batches over the training/test dataset.
        X, y = data.x.to(device), data.y.to(device)
        edges = data.edge_index.to(device)
        batch = data.batch.to(device)
        out, _ = model(X, edges, batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == y).sum())  # Check against ground-truth labels.
        precision, recall, F1, _ = precision_recall_fscore_support(y.cpu(), pred.cpu(), average='macro')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(F1)
    acc = correct / len(data_loader.dataset)
    precision = Average(precisions)
    f1 = Average(f1s)
    recall = Average(recalls)
    return acc, precision, recall, f1  # Derive ratio of correct predictions.

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once


accuracies = []
precisions = []
recalls = []
F1s = []

split = 0
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X, y_labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_labels[train_index], y_labels[test_index]
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_val= torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()
    
    train_data = []
    for i in range(X_train.shape[0]):
        data = Data(x=X_train[i], edge_index=torch.Tensor(edge_index).long().t(), y=y_train[i])
        train_data.append(data)

    val_data = []
    for i in range(X_val.shape[0]):
        data = Data(x=X_val[i], edge_index=torch.Tensor(edge_index).long().t(), y=y_val[i])
        val_data.append(data)

    test_data = []
    for i in range(X_test.shape[0]):
        data = Data(x=X_test[i], edge_index=torch.Tensor(edge_index).long().t(), y=y_test[i])
        test_data.append(data)
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    # Root mean squared error 
    
    model = GAT(feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=4)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    # model = GAT(X_train.shape[2], hidden_dim)
    
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, h, train_acc = train(train_loader)

        val_loss, val_acc = test(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (val_acc + train_acc) > best_val_acc:
            best_val_acc = val_acc + train_acc
            torch.save(model.state_dict(), save_path)

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
    model = GAT(feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=4)
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    acc, precision, recall, f1 = eval(model, test_loader)
    
    accuracies.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    F1s.append(f1)
    
    print(f'Split {split} evaluatation: Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 : {f1:.4f}\n')
    
    split+=1


print(f'Average Evaluation Metrics: Acc {sum(accuracies)/4:.4f}, Precision: {sum(precisions)/4:.4f}, Recall: {sum(recalls)/4:.4f}, F1: {sum(F1s)/4:.4f}')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5
                                                   ))
epochs = np.linspace(1, num_epochs, num_epochs).astype(int)
axes[0].plot(epochs, train_losses, color='#1E90FF', label='training loss')
axes[0].plot(epochs, val_losses, c='#FFA500', label='validation loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross Entropy Loss')
axes[0].set_title('Loss Plot')
axes[0].legend()

axes[1].plot(epochs, train_accs, color='#1E90FF', label='training accuracy')
axes[1].plot(epochs, val_accs, c='#FFA500', label='validation accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Plot')
axes[1].legend()
fig.tight_layout()
fig.savefig(training_fig_path, bbox_inches='tight')

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(encoding_fig_path, bbox_inches='tight')


# untrained_gat = GAT(dataset.num_features, 8, dataset.num_classes)

# Get embeddings
hs = torch.tensor([])
ys = torch.tensor([])
for data in train_loader:
    tmp, _ = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
    hs = torch.cat((hs, tmp.cpu()))
    ys = torch.cat((ys, data.y))

visualize(hs, ys)
