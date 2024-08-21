####################################################################################################
# A script to train and test the EEGNet model using Power Spectral Density (PSD) features
#
#
# Karim Mithani
# June 2024
####################################################################################################

import argparse
parser = argparse.ArgumentParser(description='Train and test the EEGNet model using PSD estimates.')
parser.add_argument('--use_rfe', action='store_true', help='Whether or not to use recursive feature elimination to select the best channels based on broadband power')
parser.add_argument('--rfe_method', type=str, choices=["SVC", "LogisticRegression", "RandomForest"], help='The type of RFE to use. Options are "SVC" or "LogisticRegression"', nargs='?')
parser.add_argument('--online', action='store_true', help='Whether or not to use data from a different day to improve model performance')
parser.add_argument('--fmax', type=int, help='The maximum frequency to use for the PSD estimates', nargs='?')
args = parser.parse_args()

# # For debugging, assign the arguments manually
class Args:
    def __init__(self):
        self.use_rfe = False
        self.rfe_method = 'LogisticRegression'
        self.online = True
        self.fmax = 40
args = Args()

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

# User-defined variables
processed_dir = '/d/gmi/1/karimmithani/seeg/processed'
fmax = args.fmax
if not fmax:
    raise ValueError('Please specify the maximum frequency to use for the PSD estimates')
outdir = f'/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_{fmax}Hz'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'
cles_array_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/cles'
n_chans = [10, 15, 20, 25] # Hyperparameter for number of channels when using RFE

if args.online:
    print()
    print('*'*50)
    print('Using data from a different day to improve model performance')
    print('*'*50)
    outdir = os.path.join(outdir, 'online')
    use_online = True
else:
    outdir = os.path.join(outdir, 'offline')
    use_online = False

if args.use_rfe:
    print()
    print('*'*50)
    print('Using recursive feature elimination to select the best channels based on broadband power')
    print('*'*50)
    outdir = os.path.join(outdir, 'using_rfe')
    if not args.rfe_method:
        raise ValueError('Please specify the type of RFE to use')
    if args.rfe_method == 'SVC':
        print('\nUsing SVC for RFE')
        rfe_method = 'SVC'
        outdir = os.path.join(outdir, 'SVC')
    elif args.rfe_method == 'LogisticRegression':
        print('\nUsing Logistic Regression for RFE')
        rfe_method = 'LogisticRegression'
        outdir = os.path.join(outdir, 'LogisticRegression')
    elif args.rfe_method == 'RandomForest':
        print('\nUsing Random Forest for RFE')
        rfe_method = 'RandomForest'
        outdir = os.path.join(outdir, 'RandomForest')
    use_rfe = True
else:
    use_rfe = False
    
if use_rfe:
    n_chans = n_chans
else:
    n_chans = ['all']

# use_rfe = True
# rfe_method = 'LogisticRegression' # Options are 'SVC' or 'LogisticRegression'
# if use_rfe:
#     outdir = os.path.join(outdir, 'rfe', f'{rfe_method}')
# use_online = False # If true, will inject data from a different day to improve model performance

# figures_dir = os.path.join(outdir, 'figures')
montage = 'bipolar'
target_sfreq = 250 # Resampling frequency

subjects = {
    # 'SEEG-SK-53': {'day3': ['GoNogo']},
    # 'SEEG-SK-54': {'day2': ['GoNogo_py']},
    # 'SEEG-SK-55': {'day2': ['GoNogo_py']},
    # 'SEEG-SK-62': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-63': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-64': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-66': {'day1': ['GoNogo_py']},
    # # 'SEEG-SK-67': {'day1': ['GoNogo_py']}, # Not enough NoGo trials
    # 'SEEG-SK-68': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-69': {'day1': ['GoNogo_py']}
    'SEEG-SK-70': {'day1': ['GoNogo_py']}
}

validation_data = {
    'SEEG-SK-54': {'day4': ['GoNogo_py']},
    'SEEG-SK-55': {'day3': ['GoNogo_py']},
    'SEEG-SK-62': {'day2': ['GoNogo_py']},
    'SEEG-SK-63': {'day2': ['GoNogo_py']},
    'SEEG-SK-64': {'day2': ['GoNogo_py']},
    'SEEG-SK-66': {'day2': ['GoNogo_py']},
    # 'SEEG-SK-67': {'day2': ['GoNogo_py']},
    'SEEG-SK-68': {'day2': ['GoNogo_py']},
    'SEEG-SK-69': {'day2': ['GoNogo_py']},
    'SEEG-SK-70': {'day2': ['GoNogo_py']}
}

interested_events = ['Nogo Correct', 'Nogo Incorrect']

interested_timeperiod = (-0.8, 0)

frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'low gamma': (30, 50),
    'high gamma': (50, 100)
}

if not os.path.exists(outdir):
    os.makedirs(outdir)

keras.utils.set_random_seed(42)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#######################################################################################################################################
# Functions
#######################################################################################################################################


def gonogo_dataloader(subjects, target_sfreq, montage=montage, processed_dir=processed_dir):
    
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
    
    for subj in subjects:
        epoch_files = []
        for day in subjects[subj]:
            for task in subjects[subj][day]:
                epoch_files.append(glob.glob(os.path.join(processed_dir, subj, day, task, f'*{montage}*.fif')))
        epoch_files = [item for sublist in epoch_files for item in sublist]
        epochs = mne.concatenate_epochs([mne.read_epochs(f) for f in epoch_files])
        
        # # Decimate epochs
        # decim_factor = int(epochs.info['sfreq'] / target_sfreq)
        # print(f'Resampling epochs to {epochs.info["sfreq"] / decim_factor} Hz')
        epochs.resample(target_sfreq)
        
        # Store epochs
        subjects_epochs[f'{subj}'] = epochs
        
        # Clear memory
        del epochs
        
    return subjects_epochs


def detect_spikes(timeseries, fs, filt=(25, 80), thresh=5, mindur=0.05, maxdur=0.2):
    '''
    A function to detect spikes in a time-series
    Works better with referential (i.e. not bipolar) montages
    
    Inputs:
        timeseries: a numpy array of shape (n_times,) containing time-series data
        fs: sampling frequency
        filt: a tuple containing the lower and upper bounds of the bandpass filter to apply
        thresh: threshold for spike detection
        mindur: minimum duration of a spike (in seconds)
        maxdur: maximum duration of a spike (in seconds)
    Outputs:
        spks: a numpy array of shape (n_spikes,) containing the indices of each spike
        rp: a list of regionprops objects containing information about each spike
        isspike: a numpy array of shape (n_spikes,) containing booleans indicating whether each spike meets the duration criteria
    
    See Dahal et al. (doi:10.1093/brain/awz269)
    '''

    timeseries = timeseries.squeeze()
    if timeseries.ndim > 1:
        raise ("Input timeseries must be a single vector")

    mindur_sample = np.ceil(mindur * fs)
    maxdur_sample = np.ceil(maxdur * fs)

    # filter narrowband
    btr_sos = scipy.signal.butter(4, filt, btype="bandpass", output="sos", fs=fs)
    timeseries_bandlimited = scipy.signal.sosfiltfilt(btr_sos, timeseries)

    # compute envelopes
    ts_bandlimited = np.abs(scipy.signal.hilbert(timeseries_bandlimited))
    ts_unfiltered = np.abs(scipy.signal.hilbert(timeseries))

    # zscore
    ts_bandlimited /= np.mean(ts_bandlimited)
    ts_unfiltered /= np.mean(ts_unfiltered)

    # get segments of suprathreshold filtered signal
    lb = label(ts_bandlimited >= thresh)
    rp = regionprops(lb[:, None])

    nelem = len(rp)
    isspike = np.ones(nelem, dtype=bool)
    for idx, region in enumerate(rp):
        # does putative spike meet duration criteria
        if ((region.area < mindur_sample) | (region.area > maxdur_sample)):
            isspike[idx] = False
            continue

        # is filtered amplitude accompanied by increase in broadband envelope
        if np.mean(ts_unfiltered[region.bbox[0]:region.bbox[2]]) < thresh:
            isspike[idx] = False
            continue

    if nelem == 0 or (not isspike.any()):
        return np.zeros(0), rp, isspike

    return (
        np.round(np.array([region.centroid[0] for region in compress(rp, isspike)])),
        rp,
        isspike,
    )


def convert_labels_to_bipolar(labels):
    '''
    Given a dataframe of labels and coordinates, convert them to a bipolar montage
    
    Inputs:
        labels: a pandas dataframe containing the labels and coordinates
    Outputs:
        bipolar_labels: a pandas dataframe containing the bipolar labels and coordinates
        
    '''
    
    labels = labels[labels['DataValid']!='n']
    labels['bipolar_channels'] = labels['Pinbox'] + '-' + labels['Pinbox'].shift(-1)
    # Drop rows where the prefix of the two items in the bipolar channel are not the same
    bipolar_labels = labels[labels['bipolar_channels'].str.split('-').str[0].str.replace(r'\d+', '', regex=True) == labels['bipolar_channels'].str.split('-').str[1].str.replace(r'\d+', '', regex=True)]
    
    return bipolar_labels

def plot_3d_brain(plotting_df, outdir, prefix, symmetric_cmap=True, cmap='RdBu_r', threshold=0.01):
    '''
    Plot a 3D brain using AAL regions and weights
    
    Parameters
    ----------
    plotting_df : pd.DataFrame
        DataFrame containing the following columns:
            - aal_region: AAL region name
            - weight: The weight to plot
    subj_outdir : str
        Output directory for the plots
    prefix : str
        Prefix for the output files
        
    Returns
    -------
    None
    
    '''
 
    roi_names = plotting_df['aal_region']
    
    weights = plotting_df['weight'].values
    
    # roi_names = [roi.split('_2')[0] + r oi.split('2')[-1] if '_2_' in roi else roi for roi in roi_names]

    # Load the atlas image
    atlas = nib.load('/d/gmi/1/karimmithani/utilities/aal_v3/AAL3v1_1mm.nii.gz')
    atlas_labels = pd.read_csv('/d/gmi/1/karimmithani/utilities/aal_v3/AAL3v1_1mm.nii.txt', sep=' ', header=None)
    atlas_labels.rename(columns={2: 'indices', 1: 'labels'}, inplace=True)
    atlas_labels = atlas_labels.drop(0, axis=1)
    atlas_data = atlas.get_fdata()

    # Find specific ROIs in the atlas
    # roi_indices = np.array([atlas_labels['indices'][atlas_labels['labels'].index(name)] for name in roi_names], dtype=int)
    roi_indices = [atlas_labels.loc[atlas_labels['labels'] == roi, 'indices'].values for roi in roi_names]

    # Set the values of atlas data not in the roi_indices array to 0
    for val in np.unique((atlas_data)):
        if val not in roi_indices:
            atlas_data[atlas_data == val] = 0

    for ri, roi_index in enumerate(roi_indices):
        aal_label = atlas_labels.loc[atlas_labels['indices'].values == roi_index, 'labels'].values[0]
        value = weights[ri]
        atlas_data[atlas_data == int(roi_index)] = value

    # # Assign a common vmax for all the plots
    # vmax = np.max(np.abs(loadings))

    new_atlas = nib.Nifti1Image(atlas_data, atlas.affine, atlas.header)
    big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    big_texture = vol_to_surf(new_atlas, big_fsaverage.pial_right)
    plot = plotting.view_surf(big_fsaverage.pial_right, big_texture, cmap=cmap, bg_map=big_fsaverage.sulc_right, threshold=threshold, symmetric_cmap=symmetric_cmap)
    plot.save_as_html(os.path.join(outdir, f'{prefix}_right.html'))

    new_atlas = nib.Nifti1Image(atlas_data, atlas.affine, atlas.header)
    big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    big_texture = vol_to_surf(new_atlas, big_fsaverage.pial_left)
    plot = plotting.view_surf(big_fsaverage.pial_left, big_texture, cmap=cmap, bg_map=big_fsaverage.sulc_left, threshold=threshold, symmetric_cmap=symmetric_cmap)
    plot.save_as_html(os.path.join(outdir, f'{prefix}_left.html'))


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def convert_labels_to_bipolar(labels):
    '''
    Given a dataframe of labels and coordinates, convert them to a bipolar montage
    
    Inputs:
        labels: a pandas dataframe containing the labels and coordinates
    Outputs:
        bipolar_labels: a pandas dataframe containing the bipolar labels and coordinates
        
    '''
    
    labels = labels[labels['DataValid']!='n']
    labels['bipolar_channels'] = labels['Pinbox'] + '-' + labels['Pinbox'].shift(-1)
    # Drop rows where the prefix of the two items in the bipolar channel are not the same
    bipolar_labels = labels[labels['bipolar_channels'].str.split('-').str[0].str.replace(r'\d+', '', regex=True) == labels['bipolar_channels'].str.split('-').str[1].str.replace(r'\d+', '', regex=True)]
    
    return bipolar_labels

def EEGNet_PSD_custom(Chans, Samples, dropoutRate = 0.50, 
                        F1 = 4, D = 2, mode = 'multi_channel',
                        num_days = 2):
    
    '''
    Custom-built CNN, based on the EEGNet architecture, using PSDs as input features
    
    '''
    
    input1   = Input(shape = (Chans, Samples, 1))
    
    # Add a layer to account for inter-day non-stationarity
    day_input = Input(shape=(1,))
    
    # Day-specific linear transformation layer
    day_embedding = Embedding(input_dim=num_days, output_dim=Chans*Samples, input_length=1)(day_input)
    day_embedding = tf.reshape(day_embedding, (-1, Chans, Samples, 1))
    
    x = Multiply()([input1, day_embedding])
    
    ##################################################################

    block1       = Conv2D(F1, (1, 12), use_bias = False, padding = 'same')(x) # Original kernel size = (1, 10)
    block1       = BatchNormalization()(block1)

    if mode == 'multi_channel':
        
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)

    block1       = Activation('relu')(block1)
    block1       = AveragePooling2D((1, 2), padding = 'valid')(block1) # 8 is also good
    block1       = Dropout(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F1*D, (1, 4), use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)
    block2       = AveragePooling2D((1, 2), padding = 'valid')(block2) # Can be used
    block2       = Dropout(dropoutRate)(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(1, name = 'dense')(flatten)
    out      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=[input1, day_input], outputs=out)

def get_run_logdir(root_logdir):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

#######################################################################################################################################
# Main
#######################################################################################################################################

for idx, subj in enumerate(subjects):
    
    for n_ch in n_chans:
    
        # if idx == 2: break # For debugging
        # if subj not in validation_data.keys(): continue # For debugging
        
        subj_outdir = os.path.join(outdir, f'{n_ch}_channels', subj)
        
        # if os.path.exists(os.path.join(subj_outdir, f'{subj}_completed.BAK')):
        #     print(f'{subj} already processed. Skipping...')
        #     continue
        
        print(f'\nProcessing {subj}...')
        
        if not os.path.exists(subj_outdir):
            os.makedirs(subj_outdir)
                
        subj_dict = {subj: subjects[subj]} # Done this way to allow the data loader to work with a single subject
        subj_epochs = gonogo_dataloader(subj_dict, target_sfreq)
        subj_epochs = subj_epochs[subj][interested_events]
        event_ids = subj_epochs.event_id
        
        # Load subject labels
        subj_labels = pd.read_csv(os.path.join(labels_dir, f'{subj}.labels.csv'))
        subj_labels = convert_labels_to_bipolar(subj_labels)
        subj_labels = subj_labels[subj_labels['Type']=='SEEG']
        
        # Some chnanels may have been dropped during pre-procesing steps
        subj_labels = subj_labels[subj_labels['bipolar_channels'].isin(subj_epochs.ch_names)].reset_index(drop=True)
        
        if len(subj_epochs.ch_names) != len(subj_labels):
            print(f'Number of channels in epochs ({len(subj_epochs.ch_names)}) does not match number of channels in labels ({len(subj_labels)}). Skipping...')
            continue
        
        # Detect spikes
        nonresampled_epochs = gonogo_dataloader(subj_dict, 2048)[subj][interested_events]
        epochs_array = nonresampled_epochs.get_data()
        spike_trials = []
        spike_channels = []
        Fs = nonresampled_epochs.info['sfreq']
        
        if not os.path.exists(os.path.join(subj_outdir, f'{subj}_spikes.csv')):
            print('Detecting spikes...')
            for trial in tqdm(range(epochs_array.shape[0])):
                trial_spike_channels = []
                for channel in range(epochs_array.shape[1]):
                    spikes, rp, isspike = detect_spikes(epochs_array[trial, channel, :], Fs)
                    if len(spikes) > 0:
                        trial_spike_channels.append(channel)
                if len(trial_spike_channels) > 0:
                    spike_trials.append(trial)
                    spike_channels.append(trial_spike_channels)

            spike_df = pd.DataFrame({'trial': spike_trials, 'channels': spike_channels})
            spike_df.to_csv(os.path.join(subj_outdir, f'{subj}_spikes.csv'))
        else:
            spike_df = pd.read_csv(os.path.join(subj_outdir, f'{subj}_spikes.csv'))
        
        # Remove trials with spikes
        subj_epochs.drop(spike_df['trial'].values)

        # Before cropping, baseline the data if requested
        # if with_baselining:
        #     subj_epochs.apply_baseline((None, interested_timeperiod[0]))
        
        # Crop data to the time period of interest
        subj_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
        
        #%% Normalize data
        subj_epochs_data = subj_epochs.get_data()
        subj_epochs_data = zscore(subj_epochs_data, axis=2)
        
        btr_sos = butter(4, [0.5, 40], btype='bandpass', fs=target_sfreq, output='sos')
        filtered_data = sosfiltfilt(btr_sos, subj_epochs_data, axis=2)
        # psds, freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=0.5, fmax=40, n_fft=int(target_sfreq/4), n_overlap=int(target_sfreq/8), n_jobs=10)
        signal_length = subj_epochs_data.shape[2]
        psds, freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=1, fmax=fmax, n_fft=signal_length, n_overlap=int(signal_length/2), n_jobs=10)
        # Save the frequencies as a CSV
        freqs_df = pd.DataFrame({'freqs': freqs})
        freqs_df.to_csv(os.path.join(subj_outdir, f'{subj}_freqs.csv'))
        # Normalize psds by the Simpson integral
        
        psds_normalized = np.zeros(psds.shape)
        for trial in range(psds.shape[0]):
            psds_normalized[trial, :, :] = 100 * (psds[trial, :, :] / simpson(psds[trial, :, :], freqs)[:,None])

        # Use RFE to select the best features
        ##  Aggregate data across frequency bands
        # rfe_data_orig = np.zeros((psds_normalized.shape[0], psds_normalized.shape[1], len(frequency_bands)))
        # for band in frequency_bands:
        #     fmin, fmax = frequency_bands[band]
        #     if fmax not in freqs:
        #         continue
        #     band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        #     rfe_data_orig[:, :, list(frequency_bands.keys()).index(band)] = np.mean(psds_normalized[:, :, band_indices], axis=2)
        # rfe_data = rfe_data_orig.reshape(rfe_data_orig.shape[0], -1)
        ## Alternate: Use more coarse frequency bands
        rfe_psds, rfe_freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=1, fmax=fmax, n_fft=int(target_sfreq/4), n_overlap=int(signal_length/8), n_jobs=10)
        rfe_psds_normalized = np.zeros(rfe_psds.shape)
        for trial in range(rfe_psds.shape[0]):
            rfe_psds_normalized[trial, :, :] = 100 * (rfe_psds[trial, :, :] / simpson(rfe_psds[trial, :, :], rfe_freqs)[:,None])
        rfe_data = rfe_psds_normalized.reshape(rfe_psds_normalized.shape[0], -1)
        
        #%% Extract events and binarize them for better interpretability
        events_orig = subj_epochs.events[:,-1]
        inverse_event_dict = {v: k for k, v in event_ids.items()}
        events = [inverse_event_dict[event] for event in events_orig]
        events = [0 if event == interested_events[0] else 1 for event in events]
        events = np.array(events)
        
        #%% Feature selection
        if use_rfe:
            if rfe_method == 'SVC':
                estimator = SVC(kernel='linear')
            elif rfe_method == 'LogisticRegression':
                estimator = LogisticRegression(max_iter=10000)
            elif rfe_method == 'RandomForest':
                estimator = RandomForestClassifier(random_state=42)
            
            selector = RFE(estimator, n_features_to_select=n_ch, step=1, verbose=0)
            selector = selector.fit(rfe_data, events)
            
            important_indices = np.where(selector.support_)[0]    
            # important_channels, important_freqs = np.unravel_index(important_indices, psds_normalized.shape[1:])
            important_channels, important_freqs = np.unravel_index(important_indices, rfe_psds_normalized.shape[1:])
            important_freqs = freqs[important_freqs]
            
            # Extract labels for the important channels and save as a CSV
            important_channels_df = pd.DataFrame({'chidx': important_channels, 'freq': important_freqs, 'label': subj_labels.loc[important_channels, 'bipolar_channels']}).reset_index(drop=True)
            important_channels_df.to_csv(os.path.join(subj_outdir, f'{subj}_rfe_channels.csv'))
            
            aal_regions = []
            for ch in important_channels:
                row = subj_labels.iloc[ch, :]
                coordinates = row['mni_x'], row['mni_y'], row['mni_z']
                if np.isnan(coordinates).any():
                    aal_regions.append('Unknown')
                    continue
                aal_regions.append(fuzzyquery_aal.lookup_aal_region(coordinates, fuzzy_dist=10)[1])
            
            plotting_df = pd.DataFrame({'aal_region': aal_regions, 'weight': important_freqs})
            plotting_df.to_csv(os.path.join(subj_outdir, f'{subj}_rfe_channels_frequencies.csv'))
            # Drop unknown regions
            plotting_df = plotting_df[plotting_df['aal_region'] != 'Unknown']
            plot_3d_brain(plotting_df, subj_outdir, f'{rfe_method}_rois', symmetric_cmap=False, cmap='viridis', threshold=0.01)
            
            # And also plot the coefficients
            if rfe_method == 'SVC' or rfe_method == 'LogisticRegression':
                plotting_df = pd.DataFrame({'aal_region': aal_regions, 'weight': selector.estimator_.coef_[0]})
            elif rfe_method == 'RandomForest':
                plotting_df = pd.DataFrame({'aal_region': aal_regions, 'weight': selector.estimator_.feature_importances_})
            plotting_df.to_csv(os.path.join(subj_outdir, f'{subj}_rfe_channels_coefs.csv'))
            # Drop unknown regions
            plotting_df = plotting_df[plotting_df['aal_region'] != 'Unknown']
            plot_3d_brain(plotting_df, subj_outdir, f'{rfe_method}_coefs', symmetric_cmap=True, cmap='RdBu_r', threshold=0.01)
            
            # training_data = rfe_data[:, important_indices]
            training_data = psds_normalized[:, important_channels, :]
        else:
            training_data = psds_normalized
        
        X_train, X_test, y_train, y_test = train_test_split(training_data, events, test_size=0.2, random_state=42, stratify=events)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        day_train = np.zeros(X_train.shape[0])
        day_test = np.zeros(X_test.shape[0])
        
        # Add some data from a different day to improve model performance
        if subj in validation_data.keys():
            validation_dict = {subj: validation_data[subj]}
            subj_validation_data = gonogo_dataloader(validation_dict, target_sfreq)
            subj_validation_epochs = subj_validation_data[subj][interested_events]
            subj_validation_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
            
            # Normalize data
            subj_validation_epochs_data = subj_validation_epochs.get_data()
            subj_validation_epochs_data = zscore(subj_validation_epochs_data, axis=2)
            
            btr_sos = butter(4, [0.5, 40], btype='bandpass', fs=target_sfreq, output='sos')
            filtered_data_validation = sosfiltfilt(btr_sos, subj_validation_epochs_data, axis=2)
            # psds, freqs = mne.time_frequency.psd_array_welch(filtered_data_validation, sfreq=target_sfreq, fmin=0.5, fmax=40, n_fft=int(target_sfreq/4), n_overlap=int(target_sfreq/8), n_jobs=10)
            psds, freqs = mne.time_frequency.psd_array_welch(filtered_data_validation, sfreq=target_sfreq, fmin=1, fmax=fmax, n_fft=signal_length, n_overlap=int(signal_length/2), n_jobs=10)
            # Normalize psds by the Simpson integral
            psds_normalized_validation = np.zeros(psds.shape)
            for trial in range(psds.shape[0]):
                psds_normalized_validation[trial, :, :] = 100 * (psds[trial, :, :] / simpson(psds[trial, :, :], freqs)[:,None])
            # for trial in range(psds.shape[0]):
            #     for channel in range(psds.shape[1]):
            #         psds_normalized_validation[trial, channel, :] = psds[trial, channel, :] / simpson(psds[trial, channel, :], freqs)
            
            # Extract events and binarize them for better interpretability
            events_orig = subj_validation_epochs.events[:,-1]
            inverse_event_dict = {v: k for k, v in event_ids.items()}
            events = [inverse_event_dict[event] for event in events_orig]
            events = [0 if event == interested_events[0] else 1 for event in events]
            events = np.array(events)
            
            if use_rfe:
                psds_normalized_validation = psds_normalized_validation[:, important_channels, :]
            
            if use_online:
                X_val, X_boost, y_val, y_boost = train_test_split(psds_normalized_validation, events, test_size=0.5, random_state=42, stratify=events)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
                X_boost = X_boost.reshape(X_boost.shape[0], X_boost.shape[1], X_boost.shape[2], 1)
                # Split boost further to augment both the training and testing data
                X_boost_train, X_boost_test, y_boost_train, y_boost_test = train_test_split(X_boost, y_boost, test_size=0.5, random_state=42, stratify=y_boost)
                X_train = np.concatenate((X_train, X_boost_train), axis=0)
                y_train = np.concatenate((y_train, y_boost_train), axis=0)
                X_boost_test = np.squeeze(X_boost_test)
                X_test = np.concatenate((X_test, X_boost_test), axis=0)
                y_test = np.concatenate((y_test, y_boost_test), axis=0)
                day_train = np.concatenate((day_train, np.ones(X_boost_test.shape[0])))
                day_test = np.concatenate((day_test, np.ones(X_boost_test.shape[0])))
                day_val = np.ones(X_val.shape[0])
            else:
                X_val = psds_normalized_validation
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
                y_val = events
                day_val = np.ones(X_val.shape[0])
        else:
            print(f'No different day data for {subj}, cannot use online training...')
            # For simplicity, use the test data as the validation data (results will be redundant)
            X_val = X_test
            y_val = y_test
            day_val = np.ones(X_val.shape[0])
        
        #%%
        # Save the training and validation data
        out_data_dir = os.path.join(subj_outdir, 'data')
        if not os.path.exists(out_data_dir):
            os.makedirs(out_data_dir)
        np.save(os.path.join(out_data_dir, f'{subj}_X_train.npy'), X_train)
        np.save(os.path.join(out_data_dir, f'{subj}_X_val.npy'), X_val)
        np.save(os.path.join(out_data_dir, f'{subj}_day_train.npy'), day_train)
        np.save(os.path.join(out_data_dir, f'{subj}_y_train.npy'), y_train)
        np.save(os.path.join(out_data_dir, f'{subj}_y_val.npy'), y_val)
        np.save(os.path.join(out_data_dir, f'{subj}_day_val.npy'), day_val)
        
        #%% Train the model
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        checkpoint_dir = os.path.join(subj_outdir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        num_chans = X_train.shape[1]
        num_samples = X_train.shape[2]
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f'{subj}_model'), monitor='val_loss', verbose=1, save_best_only=True)
        run_logdir = get_run_logdir(os.path.join(subj_outdir, 'logs'))
        tensorboard_cb = TensorBoard(log_dir=run_logdir)
        print('*'*50)
        print(f'\nPoint TensorBoard to:\n{run_logdir}')
        print('*'*50)
        # Pause for a bit to allow the user to copy the logdir
        time.sleep(10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
        model = EEGNet_PSD_custom(Chans=num_chans, Samples=num_samples)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # fitted_model = model.fit(X_train, y_train, epochs=100000, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint, tensorboard_cb], class_weight={0: 1., 1: 2.})
        fitted_model = model.fit([X_train, day_train], y_train, epochs=100000, batch_size=16, validation_data=([X_test, day_test], y_test), callbacks=[early_stopping, model_checkpoint, tensorboard_cb], class_weight={0: 1., 1: 2.})
        
        plt.plot(fitted_model.history['accuracy'], color='blue', label='train')
        plt.plot(fitted_model.history['val_accuracy'], color='orange', label='test')
        plt.legend()
        plt.title(f'{subj} EEGNet PSD model accuracy')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_accuracy.png'))
        plt.close()
        # plt.show()
        
        plt.plot(fitted_model.history['loss'], color='blue', label='train')
        plt.plot(fitted_model.history['val_loss'], color='orange', label='test')
        plt.legend()
        plt.title(f'{subj} EEGNet PSD model loss')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_loss.png'))
        plt.close()
        # plt.show()
        
        # Save the model using SavedModel format
        model.save(os.path.join(checkpoint_dir, f'{subj}_gonogo_model'))
        # And also in HDF5 format
        model.save(os.path.join(checkpoint_dir, f'{subj}_gonogo_model.h5'))
        
        #%% Test the model
        # Get model predictions and metrics
        from keras.models import load_model
        # model = load_model(os.path.join(checkpoint_dir, f'{subj}_model.h5'), custom_objects={'precision_metric': precision_metric})
        model = load_model(os.path.join(checkpoint_dir, f'{subj}_gonogo_model'), custom_objects={'precision_metric': precision_metric})
        # y_pred_probs = model.predict(X_test)
        y_pred_probs = model.predict([X_test, day_test])
        predictions_df = pd.DataFrame(y_pred_probs)
        predictions_df['truth'] = y_test
        predictions_df.to_csv(os.path.join(subj_outdir, f'{subj}_predictions.csv'))
        auc_roc = metrics.roc_auc_score(y_test, y_pred_probs)
        auc_prc = metrics.average_precision_score(y_test, y_pred_probs)
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_probs)
        # Find the threshold that maximizes the F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probs)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred_probs > optimal_threshold)
        
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        disp.plot(cmap='Blues')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_confmat.png'))
        plt.close()
        
        # Plot ROC and PRC curves
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{subj} ROC curve')
        plt.text(0.85, 0.05, f'AUC: {auc_roc:.2f}')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_roc.png'))
        plt.close()
        # plt.show()
        
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{subj} Precision-Recall curve')
        plt.text(0.95, 0.95, f'AUPRC = {(auc_prc):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
        noskill = len(y_test[y_test==1]) / len(y_test)
        plt.axhline(noskill, linestyle='--', color='red')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_prc.png'))
        plt.close()
        # plt.show()
        
        #%%
        # Validate on data from a different day
        # if subj in validation_data.keys():
            # validation_dict = {subj: validation_data[subj]}
            # subj_validation_data = gonogo_dataloader(validation_dict, target_sfreq)
            # subj_validation_epochs = subj_validation_data[subj][interested_events]
            # subj_validation_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
            
            # # Normalize data
            # subj_validation_epochs_data = subj_validation_epochs.get_data()
            # subj_validation_epochs_data = zscore(subj_validation_epochs_data, axis=2)
            
            # btr_sos = butter(4, [0.5, 40], btype='bandpass', fs=target_sfreq, output='sos')
            # filtered_data_validation = sosfiltfilt(btr_sos, subj_validation_epochs_data, axis=2)
            # psds, freqs = mne.time_frequency.psd_array_welch(filtered_data_validation, sfreq=target_sfreq, fmin=0.5, fmax=40, n_fft=int(target_sfreq/4), n_overlap=int(target_sfreq/8), n_jobs=10)
            # # Normalize psds by the Simpson integral
            # psds_normalized_validation = np.zeros(psds.shape)
            # for trial in range(psds.shape[0]):
            #     for channel in range(psds.shape[1]):
            #         psds_normalized_validation[trial, channel, :] = psds[trial, channel, :] / simpson(psds[trial, channel, :], freqs)
            
            # # Extract events and binarize them for better interpretability
            # events_orig = subj_validation_epochs.events[:,-1]
            # inverse_event_dict = {v: k for k, v in event_ids.items()}
            # events = [inverse_event_dict[event] for event in events_orig]
            # events = [0 if event == interested_events[0] else 1 for event in events]
            # events = np.array(events)
            
            # if use_rfe:
            #     psds_normalized_validation = psds_normalized_validation[:, important_channels, :]
            
            # X_val, y_val = psds_normalized_validation, events
            
            # y_pred_probs_val = model.predict(X_val)
        y_pred_probs_val = model.predict([X_val, day_val])
        predictions_val_df = pd.DataFrame(y_pred_probs_val)
        predictions_val_df['truth'] = y_val
        predictions_val_df.to_csv(os.path.join(subj_outdir, f'{subj}_validation_predictions.csv'))
        auc_roc_val = metrics.roc_auc_score(y_val, y_pred_probs_val)
        auc_prc_val = metrics.average_precision_score(y_val, y_pred_probs_val)
        precision_val, recall_val, _ = metrics.precision_recall_curve(y_val, y_pred_probs_val)
        fpr_val, tpr_val, _ = metrics.roc_curve(y_val, y_pred_probs_val)
        val_confusion_matrix = metrics.confusion_matrix(y_val, y_pred_probs_val > optimal_threshold)
        
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
        disp.plot(cmap='Blues')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_confmat_val.png'))
        plt.close()
        
        # Plot ROC and PRC curves
        plt.figure()
        plt.plot(fpr_val, tpr_val)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{subj} ROC curve')
        plt.text(0.85, 0.05, f'AUC: {auc_roc_val:.2f}')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_roc_val.png'))
        plt.close()
        # plt.show()
        
        plt.figure()
        plt.plot(recall_val, precision_val)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{subj} Precision-Recall curve')
        plt.text(0.95, 0.95, f'AUPRC = {(auc_prc_val):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
        noskill = len(y_val[y_val==1]) / len(y_val)
        plt.axhline(noskill, linestyle='--', color='red')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_prc_val.png'))
        plt.close()
        # plt.show()
        
    





#%%
#######################################################################################################################################
# TEMPORARY CODE FOR TESTING
#######################################################################################################################################

# Use a temporary directory
subj_outdir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/tmp'

optimizer=keras.optimizers.Adam(learning_rate=0.0001)
checkpoint_dir = os.path.join(subj_outdir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
num_chans = X_train.shape[1]
num_samples = X_train.shape[2]
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f'{subj}_model'), monitor='val_loss', verbose=1, save_best_only=True)
run_logdir = get_run_logdir(os.path.join(subj_outdir, 'logs'))
tensorboard_cb = TensorBoard(log_dir=run_logdir)
print('*'*50)
print(f'\nPoint TensorBoard to:\n{run_logdir}')
print('*'*50)
# Pause for a bit to allow the user to copy the logdir
# time.sleep(10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
model = EEGNet_PSD_custom(Chans=num_chans, Samples=num_samples)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
fitted_model = model.fit([X_train, day_train], y_train, epochs=100000, batch_size=16, validation_data=([X_test, day_test], y_test), callbacks=[early_stopping, model_checkpoint, tensorboard_cb], class_weight={0: 1., 1: 4.})

plt.plot(fitted_model.history['accuracy'], color='blue', label='train')
plt.plot(fitted_model.history['val_accuracy'], color='orange', label='test')
plt.legend()
plt.title(f'{subj} EEGNet PSD model accuracy')
plt.show()

plt.plot(fitted_model.history['loss'], color='blue', label='train')
plt.plot(fitted_model.history['val_loss'], color='orange', label='test')
plt.legend()
plt.title(f'{subj} EEGNet PSD model loss')
plt.show()

# Test the model

y_pred_probs = model.predict([X_test, day_test])
predictions_df = pd.DataFrame(y_pred_probs)
predictions_df['truth'] = y_test
predictions_df.to_csv(os.path.join(subj_outdir, f'{subj}_predictions.csv'))
auc_roc = metrics.roc_auc_score(y_test, y_pred_probs)
auc_prc = metrics.average_precision_score(y_test, y_pred_probs)
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_probs)
# Find the threshold that maximizes the F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probs)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_probs > optimal_threshold)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot(cmap='Blues')

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{subj} ROC curve')
plt.text(0.85, 0.05, f'AUC: {auc_roc:.2f}')
plt.show()

plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{subj} Precision-Recall curve')
plt.text(0.95, 0.95, f'AUPRC = {(auc_prc):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
noskill = len(y_test[y_test==1]) / len(y_test)
plt.axhline(noskill, linestyle='--', color='red')
plt.show()

# Validate on data from a different day
y_pred_probs_val = model.predict([X_val, day_val])
predictions_val_df = pd.DataFrame(y_pred_probs_val)
predictions_val_df['truth'] = y_val
auc_roc_val = metrics.roc_auc_score(y_val, y_pred_probs_val)
auc_prc_val = metrics.average_precision_score(y_val, y_pred_probs_val)
precision_val, recall_val, _ = metrics.precision_recall_curve(y_val, y_pred_probs_val)
fpr_val, tpr_val, _ = metrics.roc_curve(y_val, y_pred_probs_val)
val_confusion_matrix = metrics.confusion_matrix(y_val, y_pred_probs_val > optimal_threshold)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
disp.plot(cmap='Blues')

# Plot ROC and PRC curves
plt.figure()
plt.plot(fpr_val, tpr_val)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{subj} ROC curve')
plt.text(0.85, 0.05, f'AUC: {auc_roc_val:.2f}')
# plt.savefig(os.path.join(subj_outdir, f'{subj}_roc_val.png'))
# plt.close()
plt.show()

plt.figure()
plt.plot(recall_val, precision_val)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{subj} Precision-Recall curve')
plt.text(0.95, 0.95, f'AUPRC = {(auc_prc_val):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
noskill = len(y_val[y_val==1]) / len(y_val)
plt.axhline(noskill, linestyle='--', color='red')
# plt.savefig(os.path.join(subj_outdir, f'{subj}_prc_val.png'))
# plt.close()
plt.show()
