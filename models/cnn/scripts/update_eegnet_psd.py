#%%
######################################################################################################################
# Update EEGNet PSD model with new data, typically collected on the day of CLES
#
#
# Karim Mithani
# August 2024
######################################################################################################################


# General imports
import os
import numpy as np
import pandas as pd

# DSP libraries
import mne
from scipy.signal import resample, decimate, bessel, sosfiltfilt, butter
from scipy.stats import zscore
from skimage.measure import regionprops, label
import scipy # To make some historic functions work

# Stats libraries
from scipy.integrate import simpson
from sklearn.linear_model import LogisticRegression

# ML libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Lambda, Embedding, Multiply, Add
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
# from EEGModels import EEGNet_PSD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score
from imblearn.over_sampling import SMOTE

# Neuroimaging libraries
import nibabel as nib
from nilearn import plotting, datasets
from nilearn.surface import load_surf_data, vol_to_surf

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Misc
from itertools import compress
from math import floor, ceil
import h5py
from tqdm import tqdm
import glob
from seegloc import fuzzyquery_aal
from pathlib import Path
from time import strftime
import random
import time
from collections import Counter

# User-defined variables

data_dict = {'SEEG-SK-70': {'day7': ['GoNogo_py']}}
processed_dir = '/d/gmi/1/karimmithani/seeg/processed'
model_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_40Hz/online/using_smote/using_rfe/LogisticRegression/20_channels/SEEG-SK-70/tp_weight_4'
interested_events = ['Nogo Correct', 'Nogo Incorrect']
interested_timeperiod = (-0.8, 0)
montage = 'bipolar'
target_sfreq = 250
fmax = int([x for x in model_dir.split('/') if 'psd' in x][0].split('_')[-1].split('Hz')[0])

keras.utils.set_random_seed(42)

######################################################################################################################
# Functions
######################################################################################################################

def gonogo_dataloader(subjects, target_sfreq, montage=montage, processed_dir=processed_dir, drop_bad_channels=True):
    
    '''
    Load SEEG data for GoNogo task from epochs
    
    Parameters
    ----------
    subjects : dict
        A dictionary containing the subjects, days, and tasks.
    target_sfreq : int
        The target sampling frequency.
        
    Returns
    -------
    subjects_epochs : mne.Epochs
        The epochs object containing the data.
    
    '''
    
    subjects_epochs = {}
    day_vector = []
    
    for subj in subjects:
        epoch_files = []
        for day in subjects[subj]:
            for task in subjects[subj][day]:
                epoch_files.append(glob.glob(os.path.join(processed_dir, subj, day, task, f'*{montage}*.fif')))
                # specific_bad_channels = pd.read_csv(os.path.join(processed_dir, subj, day, task, f'{subj}_bad_channels.csv'), header=None)
                # bad_channels.append(specific_bad_channels[0].values)
                epoch_length = mne.read_epochs(epoch_files[0][0]).get_data().shape[2]
                day_vector.append(np.repeat(day.strip('day'), epoch_length))
        epoch_files = [item for sublist in epoch_files for item in sublist]
        epochs = mne.concatenate_epochs([mne.read_epochs(f) for f in epoch_files])
        
        # bad_channels = np.unique(np.concatenate(bad_channels))
        
        # if drop_bad_channels:
        #     print(f'Dropping {len(bad_channels)} bad channels:\n{bad_channels}')
        #     epochs.drop_channels(bad_channels)
        
        # Obtain and drop bad channels
        bad_channels_path = os.path.join(processed_dir, subj, f'{subj}_bad_channels.csv')
        if os.path.exists(bad_channels_path):
            bad_channels = pd.read_csv(bad_channels_path, header=None)[0].values
            # bad_channels.append(specific_bad_channels[0].values)
            # bad_channels.columns = ['bad_channels']
            # print(f'Dropping {len(bad_channels)} bad channels:\n{bad_channels["bad_channels"].values}')
            # epochs.drop_channels(bad_channels['bad_channels'].values)
        
        # # Decimate epochs
        # decim_factor = int(epochs.info['sfreq'] / target_sfreq)
        # print(f'Resampling epochs to {epochs.info["sfreq"] / decim_factor} Hz')
        epochs.resample(target_sfreq)
        
        # Store epochs
        subjects_epochs[f'{subj}'] = epochs
        
        # Clear memory
        del epochs
        
    return subjects_epochs, bad_channels, day_vector


def EEGNet_PSD_custom(Chans, Samples, dropoutRate = 0.50, 
                        F1 = 4, D = 2, mode = 'multi_channel',
                        num_days = 2):
    
    '''
    Custom-built CNN, based on the EEGNet architecture, using PSDs as input features
    
    '''
    
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    
    input1   = Input(shape = (Chans, Samples, 1))
    
    # Add a layer to account for inter-day non-stationarity
    day_input = Input(shape=(1,))
    
    # Day-specific linear transformation layer
    day_embedding = Embedding(input_dim=num_days, output_dim=Chans*Samples, input_length=1, embeddings_initializer=initializer)(day_input)
    day_embedding = tf.reshape(day_embedding, (-1, Chans, Samples, 1))
    
    x = Multiply()([input1, day_embedding])
    
    ##################################################################

    block1       = Conv2D(F1, (1, 12), use_bias = False, padding = 'same', kernel_initializer=initializer)(x) # Original kernel size = (1, 10)
    block1       = BatchNormalization()(block1)

    if mode == 'multi_channel':
        
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.),
                                    kernel_initializer=initializer)(block1)
        block1       = BatchNormalization()(block1)

    block1       = Activation('relu')(block1)
    block1       = AveragePooling2D((1, 2), padding = 'valid')(block1) # 8 is also good
    block1       = Dropout(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F1*D, (1, 4), use_bias = False, padding = 'same', kernel_initializer=initializer)(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)
    block2       = AveragePooling2D((1, 2), padding = 'valid')(block2) # Can be used
    block2       = Dropout(dropoutRate)(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(1, name = 'dense', kernel_initializer=initializer)(flatten)
    out      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=[input1, day_input], outputs=out)

def get_run_logdir(root_logdir):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

#%%
######################################################################################################################
# Main
######################################################################################################################

#%% Load data
subj = list(data_dict.keys())[0]
subj_outdir = os.path.join(model_dir, 'checkpoints', f'{subj}_gonogo_model_updated')
if not os.path.exists(subj_outdir):
    os.makedirs(subj_outdir)
else:
    print()
    print('*'*50)
    print(f'WARNING: Updated model for {subj} already exists.')
    print('*'*50)

subj_epochs, bad_channels, day_vector = gonogo_dataloader(data_dict, target_sfreq=250)
print(f'Dropping {len(bad_channels)} bad channels:\n{bad_channels}')
subj_epochs = subj_epochs[subj][interested_events]
subj_epochs.drop_channels(bad_channels)
event_ids = subj_epochs.event_id

#%% If important channels have been identified, use them
if os.path.exists(os.path.join(model_dir, f'{subj}_top_channels_aal.csv')):
    top_channels = pd.read_csv(os.path.join(model_dir, f'{subj}_top_channels_aal.csv'))['channel'].values
    subj_epochs.pick_channels(top_channels)
elif os.path.exists(os.path.join(model_dir, f'{subj}_rfe_channels.csv')):
    top_channels = pd.read_csv(os.path.join(model_dir, f'{subj}_rfe_channels.csv'))['label'].values
    subj_epochs.pick_channels(top_channels)
    # Identify repeat top channels
    repeat_channels = [k for k, v in Counter(top_channels).items() if v > 1]
    if len(repeat_channels) > 0:
        print()
        print('*'*50)
        print(f'WARNING: Repeat channels found:\n{repeat_channels}')
        print('*'*50)
        print()
        repeat_channel_index = [i for i, x in enumerate(top_channels) if x in repeat_channels][0]

#%% Extract features
 # Crop data to the time period of interest
subj_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])

# Normalize data
subj_epochs_data = subj_epochs.get_data()
subj_epochs_data = zscore(subj_epochs_data, axis=2)

btr_sos = butter(4, [0.5, 40], btype='bandpass', fs=target_sfreq, output='sos')
filtered_data = sosfiltfilt(btr_sos, subj_epochs_data, axis=2)
# psds, freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=0.5, fmax=40, n_fft=int(target_sfreq/4), n_overlap=int(target_sfreq/8), n_jobs=10)
signal_length = subj_epochs_data.shape[2]
psds, freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=1, fmax=fmax, n_fft=signal_length, n_overlap=int(signal_length/2), n_jobs=10)

# Normalize psds by the Simpson integral
psds_normalized = np.zeros(psds.shape)
for trial in range(psds.shape[0]):
    psds_normalized[trial, :, :] = 100 * (psds[trial, :, :] / simpson(psds[trial, :, :], freqs)[:,None])
    
#%% Extract events
events_orig = subj_epochs.events[:,-1]
inverse_event_dict = {v: k for k, v in event_ids.items()}
events = [inverse_event_dict[event] for event in events_orig]
events = [0 if event == interested_events[0] else 1 for event in events]
events = np.array(events)

#%% Split the data
X_train, X_test, y_train, y_test = train_test_split(psds_normalized, events, test_size=0.2, random_state=42, stratify=events)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# If repeat channels are found, duplicate them
# TODO: Fix after adjusting RFE code in original EEGNet PSD training script

if len(repeat_channels) > 0:
    print()
    print('*'*50)
    print(f'Duplicating repeat channels:\n{repeat_channels}')
    print('*'*50)
    print()
    X_train = np.insert(X_train, repeat_channel_index, X_train[:, repeat_channel_index], axis=1)
    X_test = np.insert(X_test, repeat_channel_index, X_test[:, repeat_channel_index], axis=1)

#%% Load model
model_path = os.path.join(model_dir, 'checkpoints', f'{subj}_gonogo_model')
model = keras.models.load_model(model_path)

#%% Replace day vector with '1'
day_train = np.ones(y_train.shape)
day_test = np.ones(y_test.shape)

#%% Get baseline model performance
X_all = np.concatenate([X_train, X_test])
day_all = np.concatenate([day_train, day_test])
y_all = np.concatenate([y_train, y_test])
with open(os.path.join(model_dir, f'{subj}_optimal_threshold.txt'), 'r') as f:
    optimal_threshold = float(f.read())
y_pred = model.predict([X_all, day_all])

baseline_metrics = pd.DataFrame({'accuracy': accuracy_score(y_all, y_pred > optimal_threshold),
                                'precision': precision_score(y_all, y_pred > optimal_threshold),
                                'recall': recall_score(y_all, y_pred > optimal_threshold),
                                'f1': f1_score(y_all, y_pred > optimal_threshold),
                                'roc_auc': roc_auc_score(y_all, y_pred)}, index=[0])

baseline_metrics.to_csv(os.path.join(subj_outdir, f'{subj}_baseline_metrics.csv'))

confusion_matrix = metrics.confusion_matrix(y_all, y_pred > optimal_threshold)            
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot(cmap='Blues')
plt.savefig(os.path.join(subj_outdir, f'{subj}_baseline_confmat.png'))
plt.close()

#%% Define callbacks and class weights
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(os.path.join(subj_outdir, f'{subj}_model'), monitor='val_loss', verbose=1, save_best_only=True)
run_logdir = get_run_logdir(os.path.join(subj_outdir, 'logs'))
tensorboard_cb = TensorBoard(log_dir=run_logdir)
print('*'*50)
print(f'\nPoint TensorBoard to:\n{run_logdir}')
print('*'*50)

if 'tp_weight' in model_dir:
    tp_weight = int(model_dir.split('tp_weight_')[-1])
    ml_class_weights = {0: 1, 1: tp_weight}

#%% Update model
fitted_model = model.fit([X_train, day_train], y_train, epochs=100000, batch_size=16, validation_data=([X_test, day_test], y_test), callbacks=[early_stopping, model_checkpoint, tensorboard_cb], class_weight=ml_class_weights)

#%% Save updated model
model.save(os.path.join(subj_outdir, f'{subj}_gonogo_model'))

#%% Get updated model performance
y_pred = model.predict([X_test, day_test])
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
f1_scores = 2 * (precision * recall) / (precision + recall)
# Replace NaNs with 0
f1_scores = np.nan_to_num(f1_scores)
optimal_threshold = thresholds[np.argmax(f1_scores)]
with open(os.path.join(subj_outdir, f'{subj}_optimal_threshold.txt'), 'w') as f:
    f.write(str(optimal_threshold))

y_pred = model.predict([X_test, y_test])
model_metrics = pd.DataFrame({'accuracy': accuracy_score(y_test, y_pred > optimal_threshold),
                                'precision': precision_score(y_test, y_pred > optimal_threshold),
                                'recall': recall_score(y_test, y_pred > optimal_threshold),
                                'f1': f1_score(y_test, y_pred > optimal_threshold),
                                'roc_auc': roc_auc_score(y_test, y_pred)}, index=[0])
model_metrics.to_csv(os.path.join(subj_outdir, f'{subj}_model_metrics.csv'))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred > optimal_threshold)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot(cmap='Blues')
plt.savefig(os.path.join(subj_outdir, f'{subj}_updated_confmat.png'))
plt.close()
