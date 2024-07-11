####################################################################################################
# A script to train and test the EEGNet model on the timeseries data.
#
#
# Karim Mithani
# June 2024
####################################################################################################

import argparse
parser = argparse.ArgumentParser(description='Train and test the EEGNet model on the timeseries data.')
# parser.add_argument('--use_cles_array', action='store_true', help='Whether or not to filter channels based on the CLES array')
parser.add_argument('--with_baselining', action='store_true', help='Whether or not to baseline the data')
parser.add_argument('--use_rfe', action='store_true', help='Whether or not to use recursive feature elimination to select the best channels based on broadband power')
parser.add_argument('--rfe_method', type=str, choices=["SVC", "LogisticRegression"], help='The type of RFE to use. Options are "SVC" or "LogisticRegression"', nargs='?')
args = parser.parse_args()

# # For debugging, assign the arguments manually
# class Args:
#     def __init__(self):
#         self.use_rfe = True
#         self.rfe_method = 'LogisticRegression'
#         self.with_baselining = False
# args = Args()

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from EEGModels import EEGNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

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

# User-defined variables
processed_dir = '/d/gmi/1/karimmithani/seeg/processed'
outdir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/sfreq_256_1600ms'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'
cles_array_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/cles'

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
    use_rfe = True
else:
    use_rfe = False

# if args.use_cles_array:
#     print()
#     print('*'*50)
#     print('Using CLES array to filter channels')
#     print('*'*50)
#     use_cles_array = True
#     outdir = os.path.join(outdir, 'using_cles_array')
# else:
#     use_cles_array = False

if args.with_baselining:
    print()
    print('*'*50)
    print('Baselining enabled')
    print('*'*50)
    with_baselining = True
    outdir = os.path.join(outdir, 'with_baselining')
else:
    with_baselining = False
    
# figures_dir = os.path.join(outdir, 'figures')
montage = 'bipolar'
target_sfreq = 256 # Resampling frequency

subjects = {
    'SEEG-SK-53': {'day3': ['GoNogo']},
    'SEEG-SK-54': {'day2': ['GoNogo_py']},
    'SEEG-SK-55': {'day2': ['GoNogo_py']},
    'SEEG-SK-62': {'day1': ['GoNogo_py']},
    'SEEG-SK-63': {'day1': ['GoNogo_py']},
    'SEEG-SK-64': {'day1': ['GoNogo_py']},
    'SEEG-SK-66': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-67': {'day1': ['GoNogo_py']}, # Not enough NoGo trials
    'SEEG-SK-68': {'day1': ['GoNogo_py']}
}

validation_data = {'SEEG-SK-54': {'day4': ['GoNogo_py']},
                   'SEEG-SK-55': {'day3': ['GoNogo_py']},
                   'SEEG-SK-62': {'day2': ['GoNogo_py']},
                   'SEEG-SK-63': {'day2': ['GoNogo_py']},
                   'SEEG-SK-64': {'day2': ['GoNogo_py']},
                   'SEEG-SK-66': {'day2': ['GoNogo_py']},
                #    'SEEG-SK-67': {'day2': ['GoNogo_py']},
                   'SEEG-SK-68': {'day2': ['GoNogo_py']}
}

interested_events = ['Nogo Correct', 'Nogo Incorrect']

interested_timeperiod = (-1.6, 0)

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


#######################################################################################################################################
# Main
#######################################################################################################################################


for idx, subj in enumerate(subjects):
    
    # if idx == 1: break # For debugging
    if subj not in validation_data.keys(): continue # For debugging
    
    subj_outdir = os.path.join(outdir, subj)
    
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
    subj_labels = subj_labels[subj_labels['Type']=='SEEG']
    subj_labels = convert_labels_to_bipolar(subj_labels)
    
    # Detect spikes
    if not os.path.exists(os.path.join(subj_outdir, f'{subj}_spikes.csv')):
        nonresampled_epochs = gonogo_dataloader(subj_dict, 2048)[subj][interested_events]
        epochs_array = nonresampled_epochs.get_data()
        spike_trials = []
        spike_channels = []
        Fs = nonresampled_epochs.info['sfreq']

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
    if with_baselining:
        subj_epochs.apply_baseline((None, interested_timeperiod[0]))
    
    # Crop data to the time period of interest
    subj_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
    
    # Drop bad channels
    bad_channels = pd.read_csv(os.path.join(processed_dir, subj, f'{subj}_bad_channels.csv'), header=None)[0].values
    subj_epochs.drop_channels(bad_channels)
    
    # Retain only those subject labels that are present in the epochs
    subj_labels = subj_labels[subj_labels['bipolar_channels'].isin(subj_epochs.ch_names)]
    
    # Normalize data
    subj_epochs_data = subj_epochs.get_data()
    subj_epochs_data = zscore(subj_epochs_data, axis=2)
    
    # # Retain only channels considered significant on logistic regression if requested
    # if use_cles_array:
    #     cles_array = pd.read_csv(os.path.join(cles_array_dir, f'{subj}_cles_array.csv'))
    #     channel_idxs = cles_array['Channel_idx'].values
    #     subj_epochs_data = subj_epochs_data[:, channel_idxs, :]
        
    # Extract events and binarize them for better interpretability
    events_orig = subj_epochs.events[:,-1]
    inverse_event_dict = {v: k for k, v in event_ids.items()}
    events = [inverse_event_dict[event] for event in events_orig]
    events = [0 if event == interested_events[0] else 1 for event in events]
    events = np.array(events)
    
    # Use RFE to select the best features based on broadband power
    if use_rfe:
        btr_sos = butter(4, [0.5, 40], btype='bandpass', fs=target_sfreq, output='sos')
        filtered_data = sosfiltfilt(btr_sos, subj_epochs_data, axis=2)
        psds, freqs = mne.time_frequency.psd_array_welch(filtered_data, sfreq=target_sfreq, fmin=0.5, fmax=40, n_fft=int(target_sfreq/4), n_overlap=int(target_sfreq/8), n_jobs=10)
        # Normalize psds by the Simpson integral
        psds_normalized = np.zeros(psds.shape)
        for trial in range(psds.shape[0]):
            for channel in range(psds.shape[1]):
                psds_normalized[trial, channel, :] = psds[trial, channel, :] / simpson(psds[trial, channel, :], freqs)
        rfe_data = np.mean(psds, axis=2)
        
        # reshaped_data = subj_epochs_data.reshape(subj_epochs_data.shape[0], -1)
        if rfe_method == 'SVC':
            estimator = SVC(kernel='linear')
        elif rfe_method == 'LogisticRegression':
            estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select=10, step=1)
        selector = selector.fit(rfe_data, events)
        important_channels_indices = np.where(selector.support_)[0]
        with open (os.path.join(subj_outdir, f'{subj}_important_channels.txt'), 'w') as f:
            for channel in important_channels_indices:
                f.write(f'{channel}\n')
        subj_epochs_data = subj_epochs_data[:, important_channels_indices, :]
        
        # Plot regions identified through RFE
        aal_regions = []
        for ch in important_channels_indices:
            row = subj_labels.iloc[ch, :]
            coordinates = row['mni_x'], row['mni_y'], row['mni_z']
            if np.isnan(coordinates).any():
                aal_regions.append('Unknown')
                continue
            aal_regions.append(fuzzyquery_aal.lookup_aal_region(coordinates, fuzzy_dist=10)[1])
        
        plotting_df = pd.DataFrame({'aal_region': aal_regions, 'weight': selector.estimator_.coef_[0]})
        plotting_df = plotting_df.groupby('aal_region').mean().reset_index()
        plotting_df = plotting_df[plotting_df['aal_region'] != 'Unknown']
        
        plot_3d_brain(plotting_df, subj_outdir, f'{rfe_method}_rois_coefs', symmetric_cmap=True, cmap='RdBu_r', threshold=0.01)
        
        # subj_labels = pd.read_csv(os.path.join(labels_dir, f'{subj}_CCEP.csv'))
        # subj_labels = subj_labels[subj_labels['Type']=='SEEG']
        # subj_labels = convert_labels_to_bipolar(subj_labels)
        # important_channels = subj_labels.loc[selector.support_, 'Pinbox'].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(subj_epochs_data, events, test_size=0.2, random_state=42, stratify=events)
    
    # Add some data from a different day to try and boost performance
    if subj in validation_data.keys():
        validation_dict = {subj: validation_data[subj]}
        subj_validation_data = gonogo_dataloader(validation_dict, target_sfreq)
        subj_validation_epochs = subj_validation_data[subj][interested_events]
        if with_baselining:
            subj_validation_epochs.apply_baseline((None, interested_timeperiod[0]))
        subj_validation_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
        subj_validation_epochs.drop_channels(bad_channels)
        
        # Normalize data
        subj_validation_epochs_data = subj_validation_epochs.get_data()
        subj_validation_epochs_data = zscore(subj_validation_epochs_data, axis=2)
        # if use_cles_array:
        #     subj_validation_epochs_data = subj_validation_epochs_data[:, channel_idxs, :]
        if use_rfe:
            subj_validation_epochs_data = subj_validation_epochs_data[:, important_channels_indices, :]
        
        # Extract events and binarize them for better interpretability
        events_orig = subj_validation_epochs.events[:,-1]
        inverse_event_dict = {v: k for k, v in event_ids.items()}
        events = [inverse_event_dict[event] for event in events_orig]
        events = [0 if event == interested_events[0] else 1 for event in events]
        events = np.array(events)
        
        X_val, X_boost, y_val, y_boost = train_test_split(subj_validation_epochs_data, events, test_size=0.5, random_state=42, stratify=events)
        
        X_train = np.concatenate((X_train, X_boost), axis=0)
        y_train = np.concatenate((y_train, y_boost), axis=0)
    else:
        continue
    
    # Define and train EEGNet model
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
    checkpoint_dir = os.path.join(subj_outdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, f'{subj}_model.h5'), monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
    model = EEGNet(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2], dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16, dropoutType='Dropout')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    fitted_model = model.fit(X_train, y_train, epochs=1000, batch_size=4, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    plt.plot(fitted_model.history['accuracy'], color='blue', label='train')
    plt.plot(fitted_model.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.title(f'{subj} EEGNet model accuracy')
    plt.savefig(os.path.join(subj_outdir, f'{subj}_accuracy.png'))
    plt.close()
    # plt.show()
    
    plt.plot(fitted_model.history['loss'], color='blue', label='train')
    plt.plot(fitted_model.history['val_loss'], color='orange', label='test')
    plt.legend()
    plt.title(f'{subj} EEGNet model loss')
    plt.savefig(os.path.join(subj_outdir, f'{subj}_loss.png'))
    plt.close()
    # plt.show()
    
    # Get model predictions and metrics
    y_pred_probs = model.predict(X_test)
    # Save y_pred_probs as a csv
    predictions_df = pd.DataFrame(y_pred_probs)
    # Add truth values
    predictions_df['truth'] = y_test
    predictions_df.to_csv(os.path.join(subj_outdir, f'{subj}_predictions.csv'))
    # pd.DataFrame(y_pred_probs).to_csv(os.path.join(subj_outdir, f'{subj}_predictions.csv'))
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    test_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    test_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    model_metrics = pd.DataFrame({'accuracy': [test_accuracy],
                                    'roc_auc': [test_roc_auc],
                                    'confusion_matrix': [test_confusion_matrix]})
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix)
    disp.plot()
    plt.savefig(os.path.join(subj_outdir, f'{subj}_confusion_matrix.png'))
    plt.close()
    # plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probs[:,1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # Add legend
    plt.legend([f'ROC AUC: {metrics.roc_auc_score(y_test, y_pred)}'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(subj_outdir, f'{subj}_roc_curve.png'))
    plt.close()
    # plt.show()
    
    # Validate on data from a different day, if available
    if subj in validation_data.keys():
        # validation_dict = {subj: validation_data[subj]}
        # subj_validation_data = gonogo_dataloader(validation_dict, target_sfreq)
        # subj_validation_epochs = subj_validation_data[subj][interested_events]
        # subj_validation_epochs.crop(tmin=interested_timeperiod[0], tmax=interested_timeperiod[1])
        
        # # Normalize data
        # subj_validation_epochs_data = subj_validation_epochs.get_data()
        # subj_validation_epochs_data = zscore(subj_validation_epochs_data, axis=2)
        
        # # Extract events and binarize them for better interpretability
        # events_orig = subj_validation_epochs.events[:,-1]
        # inverse_event_dict = {v: k for k, v in event_ids.items()}
        # events = [inverse_event_dict[event] for event in events_orig]
        # events = [0 if event == interested_events[0] else 1 for event in events]
        # events = np.array(events)
        
        # X_val, y_val = subj_validation_epochs_data, events
        
        # Get model predictions and metrics
        y_pred_probs_val = model.predict(X_val)
        predictions_val_df = pd.DataFrame(y_pred_probs_val)
        predictions_val_df['truth'] = y_val
        predictions_val_df.to_csv(os.path.join(subj_outdir, f'{subj}_validation_predictions.csv'))
        # pd.DataFrame(y_pred_probs_val).to_csv(os.path.join(subj_outdir, f'{subj}_validation_predictions.csv'))
        y_pred_val = np.argmax(y_pred_probs_val, axis=1)
        val_accuracy = metrics.accuracy_score(y_val, y_pred_val)
        val_roc_auc = metrics.roc_auc_score(y_val, y_pred_val)
        val_confusion_matrix = metrics.confusion_matrix(y_val, y_pred_val)
        
        model_metrics['val_accuracy'] = [val_accuracy]
        model_metrics['val_roc_auc'] = [val_roc_auc]
        model_metrics['val_confusion_matrix'] = [val_confusion_matrix]
        
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix)
        disp.plot()
        plt.savefig(os.path.join(subj_outdir, f'{subj}_diffday_confusion_matrix.png'))
        plt.close()
        # plt.show()
        
        # Plot ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_val, y_pred_probs_val[:,1])
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        # Add legend
        plt.legend([f'ROC AUC: {metrics.roc_auc_score(y_val, y_pred_val)}'])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_diffday_roc_curve.png'))
        plt.close()
        # plt.show()
    
    model_metrics.to_csv(os.path.join(subj_outdir, f'{subj}_model_metrics.csv'))
    
    # Save the model
    model.save(os.path.join(subj_outdir, f'{subj}_model.h5'))
        
    
#######################################################################################################################################
# Archive

# # Plot ROC curve
    # fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # # Add legend
    # plt.legend([f'ROC AUC: {metrics.roc_auc_score(y_test, y_pred)}'])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.savefig(os.path.join(subj_outdir, f'{subj}_roc_curve.png'))
    # plt.close()
    # # plt.show()