#%%
##################################################################################################################################
# Generate SHAP values for EEGNet model
#
#
# Karim Mithani
# July 2024
##################################################################################################################################

import os
import numpy as np
import pandas as pd

# ML libraries
from tensorflow.keras.models import load_model
import shap

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Neuroimaging libraries
import nibabel as nib
from nilearn import plotting, datasets
from nilearn.surface import vol_to_surf

# Misc libraries
from glob import glob
from seegloc import fuzzyquery_aal

# User-defined variables
analysis_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_40Hz/online/'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'

subjects = [x.split('/')[-1] for x in glob(os.path.join(analysis_dir, 'SEEG-*'))]

frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

##################################################################################################################################
# Functions
##################################################################################################################################

def model_predict(data):
    return model.predict(data)

def f(X):
    return model.predict([X[i, :] for i in range(X.shape[0])]).flatten()

def get_frequency_band(frequency, frequency_bands=frequency_bands):
    '''
    Given a frequency, return the frequency band that it belongs to
    
    '''
    for band, (low, high) in frequency_bands.items():
        if frequency >= low and frequency <= high:
            return band
    return None

def aggregate_features(channels, frequencies, shap_values_mean, labels):
    '''
    Given arrays of important channels, frequencies, and shap values, generate a dataframe
    that aggregates the important features
    
    '''
    features = pd.DataFrame({'chidx': channels, 'freqidx': frequencies})
    features['channel_labels'] = [labels['Label'].values[x] for x in channels]
    features = pd.merge(features, labels[['Label', 'mni_x', 'mni_y', 'mni_z']], left_on='channel_labels', right_on='Label').drop(columns='Label')
    features['frequency_band'] = [get_frequency_band(frequency+1) for frequency in features['freqidx']]
    features['shap_value'] = [shap_values_mean[chidx, freqidx] for chidx, freqidx in zip(features['chidx'], features['freqidx'])]
    
    for rowidx, row in features.iterrows():
        coordinates = [row['mni_x'], row['mni_y'], row['mni_z']]
        if np.isnan(coordinates).any():
            features.loc[rowidx, 'aal_region'] = np.nan
            continue
        aal_region = fuzzyquery_aal.lookup_aal_region(coordinates, fuzzy_dist=10)[1]
        features.loc[rowidx, 'aal_region'] = aal_region
    
    return features


def plot_3d_brain(plotting_df, outdir, prefix, weight_label='weight', symmetric_cmap=True, cmap='RdBu_r', threshold=0.01):
    '''
    Plot a 3D brain using AAL regions and weights
    
    Parameters
    ----------
    plotting_df : pd.DataFrame
        DataFrame containing the following columns:
            - aal_region: AAL region name
            - weight: The weight to plot
            - loading_zscore: Weight of the connection between the two contacts
    subj_outdir : str
        Output directory for the plots
    prefix : str
        Prefix for the output files
        
    Returns
    -------
    None
    
    '''
 
    roi_names = plotting_df['aal_region']
    
    weights = plotting_df[weight_label].values
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
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
        if len(roi_index) == 0:
            continue
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


def plot_contacts(plotting_df, outdir, weights_colname='weight', colors_colname=None, filename='ccep_contacts.html', cmap='Set1'):
    '''
    Plot SEEG contacts on a 3D brain
    
    Parameters
    ----------
    plotting_df : pd.DataFrame
        DataFrame containing the following columns:
            - Channel: Channel name
            - weight: The weight to plot
            - mni_x: MNI x-coordinate
            - mni_y: MNI y-coordinate
            - mni_z: MNI z-coordinate
    outdir : str
        Output directory for the plots
    '''
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    marker_coords = plotting_df[['mni_x', 'mni_y', 'mni_z']].values
    if weights_colname is not None:
        marker_size = plotting_df[weights_colname].values
    else:
        marker_size = 5
    if colors_colname is not None:
        marker_color_values = plotting_df[colors_colname].values
        cmaptmp = plt.get_cmap('Set1')
        cmap_values = cmaptmp(np.arange(len(np.unique(marker_color_values))))
        marker_color = [cmap_values[np.where(np.unique(marker_color_values) == x)[0][0]] for x in marker_color_values]
        # if len(np.unique(marker_color_values)) < 2:
        #     cmap_values = plt.get_cmap(cmap)(np.linspace(0, 0.2, len(np.unique(marker_color_values))))[:len(np.unique(marker_color_values))]
        #     marker_color = [cmap_values[np.where(np.unique(marker_color_values) == x)[0][0]] for x in marker_color_values]
        # else:
        #     cmaptmp = plt.get_cmap('Set1')
        #     cmap_values = cmaptmp(np.arange(len(np.unique(marker_color_values))))
        #     cmap_values = plt.get_cmap(cmap)(np.linspace(0, 0.5, len(np.unique(marker_color_values))))[:len(np.unique(marker_color_values))]
        #     marker_color = [cmap_values[np.where(np.unique(marker_color_values) == x)[0][0]] for x in marker_color_values]
        plot = plotting.view_markers(marker_coords=marker_coords, marker_size=marker_size, marker_color=marker_color)
    else:
        plot = plotting.view_markers(marker_coords=marker_coords, marker_size=marker_size, marker_color='red')
    plot.save_as_html(os.path.join(outdir, filename))

#%%
##################################################################################################################################
# Main
##################################################################################################################################

for subj in subjects:
    if subj != 'SEEG-SK-54': continue # For debugging
    
    subj_analysis_dir = os.path.join(analysis_dir, subj)
    subj_outdir = os.path.join(subj_analysis_dir, 'shap')
    if not os.path.exists(subj_outdir):
        os.makedirs(subj_outdir)
    
    if not os.path.exists(os.path.join(subj_analysis_dir, 'checkpoints', f'{subj}_gonogo_model')):
        print(f'{subj} missing predictions')
        continue
    
    # Load model
    model = load_model(os.path.join(subj_analysis_dir, 'checkpoints', f'{subj}_gonogo_model'))
    
    subj_data_dir = os.path.join(subj_analysis_dir, 'data')
    
    if not os.path.exists(subj_data_dir):
        print(f'{subj} missing data')
        continue
    
    # Load data
    X_train = np.load(os.path.join(subj_data_dir, f'{subj}_X_train.npy'))
    X_val = np.load(os.path.join(subj_data_dir, f'{subj}_X_val.npy'))
    y_train = np.load(os.path.join(subj_data_dir, f'{subj}_y_train.npy'))
    y_val = np.load(os.path.join(subj_data_dir, f'{subj}_y_val.npy'))
    
    
    #%%
    
    X_shap = X_val
    # X_shap = X_val[np.where(y_val == 1), :, :, :].squeeze(axis=0)
    
    explainer = shap.GradientExplainer(model, X_train)
    
    shap_values = explainer.shap_values(X_shap[:,:,:,:])
    
    #%%
    shap_values = shap_values.squeeze()
    
    shap_values_mean = np.mean(shap_values, axis=0)
    
    cmap_threshold = np.max(np.abs(shap_values_mean))
    sns.heatmap(shap_values_mean, cmap='RdBu_r', center=0).invert_yaxis()
    
    #%%
    subj_labels = pd.read_csv(os.path.join(labels_dir, f'{subj}.labels.csv'))
    subj_brainplots_dir = os.path.join(subj_outdir, 'brainplots')
    if not os.path.exists(subj_brainplots_dir):
        os.makedirs(subj_brainplots_dir)
    
    #%% Aggregate and plot positive features
    positive_thresh = np.percentile(shap_values_mean, 98)
    
    positive_channels, positive_frequencies = np.where(shap_values_mean > positive_thresh)
    positive_features = aggregate_features(positive_channels, positive_frequencies, shap_values_mean, subj_labels)
    positive_features.to_csv(os.path.join(subj_outdir, 'positive_features.csv'), index=False)
    # Project all the features onto the left hemisphere
    positive_features['aal_region'] = positive_features['aal_region'].apply(lambda x: x.replace('_R', '_L') if isinstance(x, str) else x)
    positive_features_agg = positive_features.groupby(['aal_region', 'frequency_band']).agg({'shap_value': 'mean'}).reset_index()
    
    for freq_band in positive_features_agg['frequency_band'].unique():
        freq_band_features = positive_features_agg[positive_features_agg['frequency_band'] == freq_band]
        # Scale up the shap values for better visualization
        freq_band_features.loc[:, 'shap_value'] = freq_band_features['shap_value'] * 1000
        # Raise an error if any of the shap_values are below the threshold
        if (np.abs(freq_band_features['shap_value']) < 0.01).any():
            raise ValueError("Infra-threshold shap values detected! These will not be visualized.")
        plot_3d_brain(freq_band_features, os.path.join(subj_brainplots_dir, 'positive'), f'positive_{freq_band}', weight_label='shap_value', symmetric_cmap=False, cmap='Reds', threshold=0.01)
    
    #%% Aggregate and plot negative features
    negative_thresh = np.percentile(shap_values_mean, 2)
    
    negative_channels, negative_frequencies = np.where(shap_values_mean < negative_thresh)
    negative_features = aggregate_features(negative_channels, negative_frequencies, shap_values_mean, subj_labels)
    negative_features.to_csv(os.path.join(subj_outdir, 'negative_features.csv'), index=False)
    negative_features_agg = negative_features.groupby(['aal_region', 'frequency_band']).agg({'shap_value': 'mean'}).reset_index()
    
    for freq_band in negative_features_agg['frequency_band'].unique():
        freq_band_features = negative_features_agg[negative_features_agg['frequency_band'] == freq_band]
        # Scale up the shap values for better visualization
        freq_band_features.loc[:, 'shap_value'] = freq_band_features['shap_value'] * 1000
        # Raise an error if any of the shap_values are below the threshold
        if (np.abs(freq_band_features['shap_value']) < 0.01).any():
            raise ValueError("Infra-threshold shap values detected! These will not be visualized.")
        plot_3d_brain(freq_band_features, os.path.join(subj_brainplots_dir, 'negative'), f'negative_{freq_band}', weight_label='shap_value', symmetric_cmap=False, cmap='Blues', threshold=0.01)
        
    #%% Alternatively, let's plot all features together, using the channel contacts rather than AAL regions
    positive_features_agg_contacts = positive_features.groupby(['channel_labels', 'frequency_band']).agg({'shap_value': 'mean'}).reset_index()
    negative_features_agg_contacts = negative_features.groupby(['channel_labels', 'frequency_band']).agg({'shap_value': 'mean'}).reset_index()
    all_features_agg_contacts = pd.concat([positive_features, negative_features], axis=0)
    # Record sign of the shap value for plotting
    all_features_agg_contacts['sign'] = [1 if x > 0 else 2 for x in all_features_agg_contacts['shap_value']]
    # And then convert to absolute value
    all_features_agg_contacts['shap_value'] = np.abs(all_features_agg_contacts['shap_value'])
    
    contactsplots_dir = os.path.join(subj_outdir, 'contacts_plots')
    if not os.path.exists(contactsplots_dir):
        os.makedirs(contactsplots_dir)
    
    for freq_band in all_features_agg_contacts['frequency_band'].unique():
        freq_band_features = all_features_agg_contacts[all_features_agg_contacts['frequency_band'] == freq_band]
        # Scale up the shap values for better visualization
        freq_band_features.loc[:, 'shap_value'] = freq_band_features['shap_value'] * 10000
        # Raise an error if any of the shap_values are below the threshold
        if (np.abs(freq_band_features['shap_value']) < 0.01).any():
            raise ValueError("Infra-threshold shap values detected! These will not be visualized.")
        plot_contacts(freq_band_features, contactsplots_dir, weights_colname='shap_value', colors_colname='sign', filename=f'{freq_band}_contacts.html')
    