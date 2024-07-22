##############################################################################################################
# Logistic regression to predict lapses in response inhibition on the Go/Nogo task using PSD estimates
#
#
# Karim Mithani
# June 2024
##############################################################################################################

import argparse
parser = argparse.ArgumentParser(description='Train and test the EEGNet model using PSD estimates.')
parser.add_argument('--use_rfe', action='store_true', help='Whether or not to use recursive feature elimination to select the best channels based on broadband power')
parser.add_argument('--use_pca', action='store_true', help='Whether or not to use PCA for dimensionality reduction')
parser.add_argument('--rfe_method', type=str, choices=["SVC", "LogisticRegression"], help='The type of RFE to use. Options are "SVC" or "LogisticRegression"', nargs='?')
parser.add_argument('--online', action='store_true', help='Whether or not to use data from a different day to improve model performance')
parser.add_argument('--fmax', type=int, help='The maximum frequency to use for the PSD estimates', nargs='?')
args = parser.parse_args()

#%%
# For debugging, assign the arguments manually
class Args:
    def __init__(self):
        self.use_rfe = False
        self.use_pca = True
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA

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
fmax = args.fmax
if not fmax:
    raise ValueError('Please specify the maximum frequency to use for the PSD estimates')
outdir = f'/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/logisticregression/analysis/psd_{fmax}Hz'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'
cles_array_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/cles'

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
    use_rfe = True
else:
    use_rfe = False

if args.use_pca:
    print()
    print('*'*50)
    print('Using PCA for dimensionality reduction')
    print('*'*50)
    outdir = os.path.join(outdir, 'using_pca')
    use_pca = True
else:
    use_pca = False

# use_rfe = True
# rfe_method = 'LogisticRegression' # Options are 'SVC' or 'LogisticRegression'
# if use_rfe:
#     outdir = os.path.join(outdir, 'rfe', f'{rfe_method}')
# use_online = False # If true, will inject data from a different day to improve model performance

# figures_dir = os.path.join(outdir, 'figures')
montage = 'bipolar'
target_sfreq = 250 # Resampling frequency

subjects = {
    'SEEG-SK-53': {'day3': ['GoNogo']},
    'SEEG-SK-54': {'day2': ['GoNogo_py']},
    'SEEG-SK-55': {'day2': ['GoNogo_py']},
    'SEEG-SK-62': {'day1': ['GoNogo_py']},
    'SEEG-SK-63': {'day1': ['GoNogo_py']},
    'SEEG-SK-64': {'day1': ['GoNogo_py']},
    'SEEG-SK-66': {'day1': ['GoNogo_py']},
    # 'SEEG-SK-67': {'day1': ['GoNogo_py']}, # Not enough NoGo trials
    'SEEG-SK-68': {'day1': ['GoNogo_py']},
    'SEEG-SK-69': {'day1': ['GoNogo_py']}
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
    'SEEG-SK-69': {'day2': ['GoNogo_py']}
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

def map_pca_components(components, pca, n_channels, n_freqs):
    '''
    Map PCA components back to their original space to identify important features
    
    Inputs:
        components: a numpy array of shape (n_components, n_loadings), where n_loadings = (n_channels * n_freqs)
        pca: the PCA object
        n_channels: the number of channels in the original data
        n_freqs: the number of frequency bands in the original data
    Outputs:
        components_mapped: a numpy array of shape (n_trials, n_channels, n_freqs)
    '''
    
    components_mapped = np.zeros((components.shape[0], n_channels, n_freqs))
    for pc in range(components.shape[0]):
        components_mapped[pc, :, :] = pca.components_[pc].reshape(n_channels, n_freqs)
    
    return components_mapped

#######################################################################################################################################
# Main
#######################################################################################################################################

for idx, subj in enumerate(subjects):
    
    # if idx == 1: break # For debugging
    #%%
    if subj not in validation_data.keys(): continue # For debugging
    
    #%%
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
    subj_labels = convert_labels_to_bipolar(subj_labels)
    subj_labels = subj_labels[subj_labels['Type']=='SEEG']
    
    # Some chnanels may have been dropped during pre-procesing steps
    subj_labels = subj_labels[subj_labels['bipolar_channels'].isin(subj_epochs.ch_names)].reset_index(drop=True)
    
    #%%
    if len(subj_epochs.ch_names) != len(subj_labels):
        print(f'Number of channels in epochs ({len(subj_epochs.ch_names)}) does not match number of channels in labels ({len(subj_labels)}). Skipping...')
        continue
    
    #%%
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
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        selector = RFE(estimator, n_features_to_select=10, step=1, verbose=0)
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
        plotting_df = pd.DataFrame({'aal_region': aal_regions, 'weight': selector.estimator_.coef_[0]})
        plotting_df.to_csv(os.path.join(subj_outdir, f'{subj}_rfe_channels_coefs.csv'))
        # Drop unknown regions
        plotting_df = plotting_df[plotting_df['aal_region'] != 'Unknown']
        plot_3d_brain(plotting_df, subj_outdir, f'{rfe_method}_coefs', symmetric_cmap=True, cmap='RdBu_r', threshold=0.01)
        
        # training_data = rfe_data[:, important_indices]
        training_data = psds_normalized[:, important_channels, :]
    else:
        training_data = psds_normalized
    
    if use_pca:
        #%%
        pca = PCA(n_components=0.95)
        # Training data is in shape n_trials x n_channels x n_freqs
        tmp = pca.fit_transform(training_data.reshape(training_data.shape[0], -1))
        n_components = pca.n_components_
        print(f'\nFitted PCA with {pca.n_components_} components to explain 95% of the variance in the training data')
        
        # Plot the explained variance
        plt.plot(np.arange(0, len(pca.explained_variance_ratio_), 1), pca.explained_variance_ratio_)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.title(f'{subj} PCA Cumulative Explained Variance')
        plt.savefig(os.path.join(subj_outdir, f'{subj}_explained_variance.png'))
        plt.close()
        # plt.show()
        
        # Map the components back to the original space
        components_mapped = map_pca_components(pca.components_, pca, training_data.shape[1], training_data.shape[2])
        # Save the components
        pca_components = pd.DataFrame(components_mapped.reshape(components_mapped.shape[0], -1))
        pca_components.to_csv(os.path.join(subj_outdir, f'{subj}_pca_components.csv'))
        
        training_data = tmp
    
    X_train, X_test, y_train, y_test = train_test_split(training_data, events, test_size=0.2, random_state=42, stratify=events)
    if not use_pca: # If using PCA, the data is already 2-dimensional
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    
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
            
        if use_pca:
            if psds_normalized_validation.shape[0] < n_components:
                print(f'Not enough trials to fit PCA. Skipping...')
                continue
            pca = PCA(n_components=n_components)
            tmp = pca.fit_transform(psds_normalized_validation.reshape(psds_normalized_validation.shape[0],-1))
            psds_normalized_validation = tmp
        
        if use_online:
            X_val, X_boost, y_val, y_boost = train_test_split(psds_normalized_validation, events, test_size=0.5, random_state=42)
            if not use_pca: # If using PCA, the data is already 2-dimensional
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
                X_boost = X_boost.reshape(X_boost.shape[0], X_boost.shape[1], X_boost.shape[2], 1)
            X_train = np.concatenate((X_train, X_boost), axis=0)
            y_train = np.concatenate((y_train, y_boost), axis=0)
        else:
            X_val = psds_normalized_validation
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            y_val = events
    # else:
    #     continue
    
    #%% Train the model
    model = LogisticRegression(C=30, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    #%%
    # Plot the features to help contextualize the results
    if not use_pca:
        tmp = X_train.squeeze()
        n_channels = tmp.shape[1]
        n_cols = 5
        n_rows = ceil(n_channels/n_cols)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))
        axes = axes.flatten()
        for ch in range(tmp.shape[1]):
            plotting_df = pd.DataFrame(tmp[:,ch,:])
            plotting_df['outcome']=y_train
            plotting_df = plotting_df.melt(id_vars='outcome')
            sns.lineplot(data=plotting_df, x='variable', y='value', hue='outcome', ax=axes[ch])
        plt.savefig(os.path.join(subj_outdir, f'{subj}_training_psds.png'))
        plt.close()
        
        tmp = X_val.squeeze()
        n_channels = tmp.shape[1]
        n_cols = 5
        n_rows = ceil(n_channels/n_cols)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))
        axes = axes.flatten()
        for ch in range(tmp.shape[1]):
            plotting_df = pd.DataFrame(tmp[:,ch,:])
            plotting_df['outcome']=y_val
            plotting_df = plotting_df.melt(id_vars='outcome')
            sns.lineplot(data=plotting_df, x='variable', y='value', hue='outcome', ax=axes[ch])
        plt.savefig(os.path.join(subj_outdir, f'{subj}_validation_psds.png'))
        plt.close()
    
    #%% Evaluate the model
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    y_pred_probs = model.predict(X_test)
    predictions_df = pd.DataFrame(y_pred_probs)
    predictions_df['truth'] = y_test
    predictions_df.to_csv(os.path.join(subj_outdir, f'{subj}_predictions.csv'))
    auc_roc = metrics.roc_auc_score(y_test, y_pred_probs)
    auc_prc = metrics.average_precision_score(y_test, y_pred_probs)
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_probs)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probs)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_probs > 0.5)
    
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
    if subj in validation_data.keys():
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
        
        y_pred_probs_val = model.predict(X_val)
        predictions_val_df = pd.DataFrame(y_pred_probs_val)
        predictions_val_df['truth'] = y_val
        predictions_val_df.to_csv(os.path.join(subj_outdir, f'{subj}_validation_predictions.csv'))
        auc_roc_val = metrics.roc_auc_score(y_val, y_pred_probs_val)
        auc_prc_val = metrics.average_precision_score(y_val, y_pred_probs_val)
        precision_val, recall_val, _ = metrics.precision_recall_curve(y_val, y_pred_probs_val)
        fpr_val, tpr_val, _ = metrics.roc_curve(y_val, y_pred_probs_val)
        val_confusion_matrix = metrics.confusion_matrix(y_val, y_pred_probs_val > 0.5)
        
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