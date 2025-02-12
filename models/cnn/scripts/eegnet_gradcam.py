#%%
##################################################################################################################################
# Use GradCAM to explain subject-specific EEGNet models
#
#
# Karim Mithani
# February 2025
##################################################################################################################################

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from scipy.stats import ttest_1samp
import nibabel as nib

import cv2
import re



# User-defined variables
outdir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/gradcam'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'
top_models_csv = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_40Hz/online/using_rfe/LogisticRegression/top_models/top_models.csv'
count_threshold = 3 # Number of occurrences to keep in the count plot

mni_path = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
sphere_radius = 10
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

##################################################################################################################################
# Functions
##################################################################################################################################

def make_gradcam_heatmap(model, psd_input, day_input, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given model and inputs.

    Args:
        model: The trained tf.keras.Model.
        psd_input: A numpy array with shape (1, Chans, Samples, 1).
        day_input: A numpy array with shape (1, 1) for the day index.
        last_conv_layer_name: String, the name of the target convolutional layer.
        pred_index: (Optional) Index of the target class. If None, use the predicted class.
    
    Returns:
        heatmap: A 2D numpy array of the Grad-CAM heatmap.
    """
    # Create a model that maps the inputs to the activations of the target layer
    # as well as the model's output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass: compute activations and predictions
        conv_outputs, predictions = grad_model([psd_input, day_input])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # For a classification model, select the score of the target class.
        loss = predictions[:, pred_index]

    # Compute the gradient of the loss with respect to the convolutional layer outputs
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute the mean intensity of the gradients for each channel (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # assuming conv_outputs shape: (batch, H, W, channels)

    # Obtain the activations of the convolutional layer for the first (and only) image in the batch
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    
    # Multiply each channel in the feature map array by the corresponding gradient importance
    conv_outputs_weighted = conv_outputs * pooled_grads[..., tf.newaxis, tf.newaxis]
    # Sum along the channel dimension to get the raw heatmap
    heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)

    # Apply ReLU to the heatmap (only keep features that have a positive influence)
    heatmap = tf.nn.relu(heatmap)
    
    # Normalize the heatmap between 0 and 1 for visualization
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        # Avoid division by zero
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap /= max_val
    
    return heatmap.numpy()

def generate_volumetric(img_df, template_img_path=mni_path, weight_colname='weight', smoothing_kernel=None, sphere_radius=sphere_radius, return_img=True):
    
    '''
    Generate a volumetric mask from a dataframe containing MNI coordinates and weights
    
    Parameters
    ----------
    img_df: DataFrame containing MNI coordinates and weights
    template_img_path: Path to the template image
    weight_colname: Name of the column containing the weights
    smoothing_kernel: Sigma of the Gaussian kernel used for smoothing
    sphere_radius: Radius of the sphere around each coordinate
    return_img: Return the mask as a Nifti1Image object
    
    Returns
    ----------
    mask_img: Numpy array or Nifti1Image object
    
    '''
    
    template_img = nib.load(template_img_path)
    affine = template_img.affine
    mask_img = np.zeros_like(template_img.get_fdata())
    voxel_indices = []
    voxel_weights = []
    for rowidx, row in img_df.iterrows():
        ch_coord = np.array([row['mni_x'], row['mni_y'], row['mni_z']])
        if np.isnan(ch_coord).any():
                continue
        ch_coord_homogeneous = np.append(ch_coord, 1)
        voxel_index = np.dot(np.linalg.inv(affine), ch_coord_homogeneous)[:3].astype(int)
        voxel_indices.append(voxel_index)
        voxel_weights.append(row[weight_colname])
    voxel_indices = np.array(voxel_indices)
    for idx, (x, y, z) in enumerate(voxel_indices):
        for i in range(-sphere_radius, sphere_radius):
            for j in range(-sphere_radius, sphere_radius):
                for k in range(-sphere_radius, sphere_radius):
                    if np.sqrt(i**2 + j**2 + k**2) <= sphere_radius:
                        mask_img[x+i, y+j, z+k] += voxel_weights[idx]
    if smoothing_kernel is not None:
        mask_img = scipy.ndimage.gaussian_filter(mask_img, sigma=smoothing_kernel)
    if not return_img: # Return the mask as a numpy array
        return mask_img
    else:
        mask_img = nib.Nifti1Image(mask_img, affine)
        return mask_img

def convert_labels_to_bipolar(labels):
    '''
    Given labels of SEEG channels, convert them to bipolar.
    '''
    
    transformed = []
    for s in labels:
        match = re.match(r"(\D+)(\d+)", s)
        if match:
            prefix, num = match.groups()
            num = int(num)  # Convert to integer
            transformed.append(f"{prefix}{num}-{prefix}{num+1}")
    return transformed





#%%
##################################################################################################################################
# SUBJECT-SPECIFIC ANALYSIS
##################################################################################################################################

top_models = pd.read_csv(top_models_csv)

subj_specific_outdir = os.path.join(outdir, 'subj_specific')

for rowidx, row in top_models.iterrows():
    subj = row['subject']
    # if subj != 'SEEG-SK-75': continue # For debugging
    
    subj_analysis_dir = row['model_path']
    subj_outdir = os.path.join(subj_specific_outdir, subj)
    
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
    
    subj_optimal_threshold_fname = os.path.join(subj_analysis_dir, f'{subj}_optimal_threshold.txt')
    with open(subj_optimal_threshold_fname, 'r') as f:
        subj_optimal_threshold = float(f.read())
    
    # Load data
    X_train = np.load(os.path.join(subj_data_dir, f'{subj}_X_train.npy'))
    X_day_train = np.load(os.path.join(subj_data_dir, f'{subj}_day_train.npy'))
    X_val = np.load(os.path.join(subj_data_dir, f'{subj}_X_val.npy'))
    X_day_val = np.load(os.path.join(subj_data_dir, f'{subj}_day_val.npy'))
    y_train = np.load(os.path.join(subj_data_dir, f'{subj}_y_train.npy'))
    y_val = np.load(os.path.join(subj_data_dir, f'{subj}_y_val.npy'))
    
    if len(X_val.shape) != 4: # In cases where no different day data was available, an extra dimension needs to be added
        X_val = np.expand_dims(X_val, axis=-1)
    
    X_day_train = X_day_train.reshape(-1, 1)
    X_day_val   = X_day_val.reshape(-1, 1)
    
    #%% Get the last convolutional layer
    last_conv_layer_name = model.layers[-8].name
    
    pred = model.predict([X_val, X_day_val])
    X_val_pred = X_val[np.where(pred >= subj_optimal_threshold)[0], :, :, :]
    X_day_val_pred = X_day_val[np.where(pred >= subj_optimal_threshold)[0]]
    
    heatmap = make_gradcam_heatmap(model, X_val_pred, X_day_val_pred, last_conv_layer_name)
    
    # Reshape the heatmap to the same shape as the input
    heatmap = cv2.resize(heatmap, (X_val_pred.shape[2], X_val_pred.shape[1]))
    
    plt.imshow(heatmap, alpha=1, cmap='OrRd', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(os.path.join(subj_outdir, 'heatmap.png'))
    plt.close()
    
    # #%% Repeat, but with correct predictions
    # X_val_pred_indices = np.where(pred >= subj_optimal_threshold)[0]
    # X_val_correct_indices = np.where(y_val == 1)[0]
    # X_val_pred_correct_indices = np.intersect1d(X_val_pred_indices, X_val_correct_indices)
    # X_val_pred_correct = X_val[X_val_pred_correct_indices, :, :, :]
    # X_day_val_pred_correct = X_day_val[X_val_pred_correct_indices]
    
    # heatmap_correct = make_gradcam_heatmap(model, X_val_pred_correct, X_day_val_pred_correct, last_conv_layer_name)
    
    # # Reshape the heatmap to the same shape as the input
    # heatmap_correct = cv2.resize(heatmap_correct, (X_val_pred_correct.shape[2], X_val_pred_correct.shape[1]))
    
    # plt.imshow(heatmap_correct, alpha=1, cmap='OrRd', vmin=0, vmax=1)
    # plt.colorbar()
    
    
    
     #%% Load channel and frequency labels
    subj_freqs_fname = os.path.join(subj_analysis_dir, f'{subj}_freqs.csv')
    subj_freqs = pd.read_csv(subj_freqs_fname)
    # Ensure that the number of frequencies in the dataframe matches the number of frequencies in the data
    if subj_freqs.shape[0] != X_val_pred.shape[2]:
        raise ValueError(f'Number of frequencies in {subj_freqs_fname} does not match the number of frequencies in the data')
    
    subj_chans_fname = os.path.join(subj_analysis_dir, f'{subj}_rfe_channels.csv')
    subj_chans = pd.read_csv(subj_chans_fname)
    # Ensure that the number of channels in the dataframe matches the number of channels in the data
    if subj_chans.shape[0] != X_val_pred.shape[1]:
        raise ValueError(f'Number of channels in {subj_chans_fname} does not match the number of channels in the data')
    
    #%% Calculate the mean difference between aggregate PSDs of predicted conditions to inform the importance of the top indices
    X_val_pred_nolapse = X_val[np.where(pred < subj_optimal_threshold)[0], :, :, :]
    X_val_pred_lapse = X_val_pred.copy() # To keep naming conventions sensible
    
    if X_val_pred_nolapse.shape[0] == 0:
        # If there are no nolapse predictions, set the mean difference to the mean of the lapse predictions
        X_val_diff = X_val_pred_lapse.mean(axis=0).squeeze()
    elif X_val_pred_lapse.shape[0] == 0:
        # If there are no lapse predictions, set the mean difference to the mean of the nolapse predictions
        X_val_diff = X_val_pred_nolapse.mean(axis=0).squeeze()
    else:
        X_val_diff = X_val_pred_lapse.mean(axis=0).squeeze() - X_val_pred_nolapse.mean(axis=0).squeeze()
    
    # Plot mean difference
    vmax = np.max(np.abs(X_val_diff))
    plt.imshow(X_val_diff, cmap='RdBu_r', alpha=1, vmax=vmax, vmin=-vmax)
    plt.colorbar()
    plt.savefig(os.path.join(subj_outdir, 'mean_diff.png'))
    plt.close()
    
    #%% 
    feature_df = pd.DataFrame(heatmap) # Channels x Frequencies
    # Assign the channel and frequency labels
    feature_df['channel'] = subj_chans['label'].values
    feature_df = pd.melt(feature_df, id_vars='channel', var_name='freqidx', value_name='importance')
    feature_df['freqidx'] = feature_df['freqidx'].astype(int)
    feature_df['frequency'] = subj_freqs.loc[feature_df['freqidx'], 'freqs'].values
    feature_df['freq_band'] = feature_df['frequency'].apply(lambda x: [k for k, v in frequency_bands.items() if v[0] <= x < v[1]][0])
    # Assign the mean difference
    for rowidx, row in feature_df.iterrows():
        chidx = subj_chans[subj_chans['label'] == row['channel']].index[0]
        freqidx = row['freqidx']
        row['mean_diff'] = X_val_diff[chidx, freqidx]
        feature_df.loc[rowidx, 'mean_diff'] = row['mean_diff']
    # Merge with subject MNI coordinates
    subj_labels = pd.read_csv(os.path.join(labels_dir, f'{subj}.labels.csv'))
    subj_labels = subj_labels[subj_labels['Type'] == 'SEEG']
    subj_labels['Label'] = convert_labels_to_bipolar(subj_labels['Label'])
    feature_df = pd.merge(feature_df, subj_labels, left_on='channel', right_on='Label', how='left')
    # Project all to left side
    feature_df['mni_x'] = feature_df['mni_x'].apply(lambda x: -x if x > 0 else x)
    feature_df.to_csv(os.path.join(subj_outdir, 'feature_importance.csv'), index=False)
    
    #%% Threshold the heatmap
    # threshold = np.percentile(heatmap, 90)
    threshold = 0.50
    
    #%% Extract the indices of the top 5% of the heatmap
    top_indices = np.where(heatmap > threshold)
    top_indices = np.array(top_indices).T
    top_indices_df = pd.DataFrame(top_indices, columns=['chidx', 'freqidx'])
    
    # Assign the channel and frequency labels
    top_indices_df['channel'] = subj_chans.loc[top_indices_df['chidx'], 'label'].values
    top_indices_df['frequency'] = subj_freqs.loc[top_indices_df['freqidx'], 'freqs'].values
    top_indices_df['freq_band'] = top_indices_df['frequency'].apply(lambda x: [k for k, v in frequency_bands.items() if v[0] <= x < v[1]][0])
    
    
    top_indices_df['mean_diff'] = X_val_diff[top_indices_df['chidx'], top_indices_df['freqidx']]
    
    # # Plot mean difference
    # X_val_diff_thresholded = X_val_diff.copy()
    # heatmap_binary = heatmap.copy()
    # heatmap_binary[heatmap_binary > 0] = 1
    # X_val_diff_thresholded = X_val_diff_thresholded * heatmap_binary
    # vmax = np.max(np.abs(X_val_diff_thresholded))
    # plt.imshow(X_val_diff_thresholded, cmap='RdBu_r', alpha=1, vmax=vmax, vmin=-vmax)
    # plt.colorbar()
    # plt.savefig(os.path.join(subj_outdir, 'mean_diff.png'))
    # plt.close()
    
    #%% Plot PSDs of top channels
    psds_dir = os.path.join(subj_outdir, 'psds')
    if not os.path.exists(psds_dir):
        os.makedirs(psds_dir)
    top_channel_indices = top_indices_df['chidx'].unique()
    for chidx in top_channel_indices:
        ch_pred_nolapse = X_val_pred_nolapse[:, chidx, :, :].squeeze()
        ch_pred_lapse = X_val_pred_lapse[:, chidx, :, :].squeeze()
        ch_df_nolapse = pd.DataFrame(ch_pred_nolapse)
        ch_df_nolapse['condition'] = 'no_lapse'
        ch_df_lapse = pd.DataFrame(ch_pred_lapse)
        ch_df_lapse['condition'] = 'lapse'
        ch_df = pd.concat((ch_df_nolapse, ch_df_lapse), axis=0)
        ch_df = ch_df.reset_index(drop=False)
        ch_df_melted = ch_df.melt(id_vars=['index', 'condition'], var_name='freq', value_name='psd')
        plt.figure()
        sns.lineplot(data=ch_df_melted, x='freq', y='psd', hue='condition', errorbar=("se", 1))
        plt.title(f'{subj_chans.loc[chidx, "label"]}')
        sns.despine()
        plt.savefig(os.path.join(psds_dir, f'psd_{subj_chans.loc[chidx, "label"]}.png'))
        plt.close()

    #%% Merge with subject MNI coordinates and plot volumetric image
    subj_labels = pd.read_csv(os.path.join(labels_dir, f'{subj}.labels.csv'))
    subj_labels = subj_labels[subj_labels['Type'] == 'SEEG']
    subj_labels['Label'] = convert_labels_to_bipolar(subj_labels['Label'])
    top_indices_df = pd.merge(top_indices_df, subj_labels, left_on='channel', right_on='Label', how='left')
    # Project all to left side
    top_indices_df['mni_x'] = top_indices_df['mni_x'].apply(lambda x: -x if x > 0 else x)
    for freqband in top_indices_df['freq_band'].unique():
        plotting_df = top_indices_df[top_indices_df['freq_band'] == freqband]
        # Collapse the dataframe to unique coordinates
        plotting_df = plotting_df[['channel', 'freq_band', 'mean_diff', 'mni_x', 'mni_y', 'mni_z']].groupby(['channel', 'freq_band']).mean().reset_index()
        subj_fb_img = generate_volumetric(plotting_df, template_img_path=mni_path, weight_colname='mean_diff', smoothing_kernel=2, sphere_radius=10, return_img=True)
        nib.save(subj_fb_img, os.path.join(subj_outdir, f'important_features_{freqband}.nii.gz'))
        # Also plot a binary plot
        plotting_df['weight'] = plotting_df['mean_diff'].apply(lambda x: 1 if x > 0 else -1)
        subj_fb_binary_img = generate_volumetric(plotting_df, template_img_path=mni_path, weight_colname='weight', smoothing_kernel=2, sphere_radius=10, return_img=True)
        nib.save(subj_fb_binary_img, os.path.join(subj_outdir, f'important_features_{freqband}_binary.nii.gz'))
        

    #%% Save the top indices
    top_indices_df.to_csv(os.path.join(subj_outdir, 'top_features.csv'), index=False)





#%% 
######################################################################################################
# GROUP ANALYSIS
######################################################################################################

# Plot channels based on their importance
group_outdir = os.path.join(outdir, 'group_analysis')
if not os.path.exists(group_outdir):
    os.makedirs(group_outdir)

all_features = pd.DataFrame()
for rowidx, row in top_models.iterrows():
    subj = row['subject']
    subj_outdir = os.path.join(subj_specific_outdir, subj)
    if not os.path.exists(subj_outdir):
        print(f'{subj} missing data')
        continue
    subj_features = pd.read_csv(os.path.join(subj_outdir, 'feature_importance.csv'))
    subj_features['subject'] = subj
    all_features = pd.concat((all_features, subj_features), axis=0)

# Plot the top channels based on their importance
all_channels = all_features[['subject', 'channel', 'importance', 'mni_x', 'mni_y', 'mni_z']].groupby(['subject', 'channel']).mean().reset_index()
all_channels_img = generate_volumetric(all_channels, template_img_path=mni_path, weight_colname='importance', smoothing_kernel=2, sphere_radius=10, return_img=True)
nib.save(all_channels_img, os.path.join(group_outdir, 'all_channels.nii.gz'))

# Identify areas that were densely sampled
all_channels['weight'] = 1
all_channels_count_img = generate_volumetric(all_channels, template_img_path=mni_path, weight_colname='weight', smoothing_kernel=2, sphere_radius=10, return_img=False)
# Threshold to voxels with at least N occurrences
all_channels_count_img[all_channels_count_img < count_threshold] = 0
all_channels_count_img = nib.Nifti1Image(all_channels_count_img, all_channels_img.affine)
nib.save(all_channels_count_img, os.path.join(group_outdir, 'all_channels_count.nii.gz'))

#%% Plot feature importances for each frequency band
all_channels_count_img = nib.load(os.path.join(group_outdir, 'all_channels_count.nii.gz'))
all_channels_count_img_data = all_channels_count_img.get_fdata()
all_channels_count_img_data[all_channels_count_img_data > 0] = 1
all_features['importance_diff_combined'] = all_features['importance'] * np.sign(all_features['mean_diff'])
for freqband in all_features['freq_band'].unique():
    fb_importances = all_features[all_features['freq_band'] == freqband].copy()
    fb_importances = fb_importances[['subject', 'channel', 'importance_diff_combined', 'mni_x', 'mni_y', 'mni_z']].groupby(['subject', 'channel']).mean().reset_index()
    fb_img = generate_volumetric(fb_importances, template_img_path=mni_path, weight_colname='importance_diff_combined', smoothing_kernel=2, sphere_radius=10, return_img=True)
    # Filter to only include densely sampled areas
    fb_img_data = fb_img.get_fdata()
    fb_img_data[all_channels_count_img_data == 0] = 0
    fb_img = nib.Nifti1Image(fb_img_data, fb_img.affine)
    nib.save(fb_img, os.path.join(group_outdir, f'importances_{freqband}.nii.gz'))

#%% Combine all subjects to plot a group-level volumetric image
all_top_indices = pd.DataFrame()
for rowidx, row in top_models.iterrows():
    subj = row['subject']
    subj_outdir = os.path.join(subj_specific_outdir, subj)
    if not os.path.exists(subj_outdir):
        continue
    subj_top_indices = pd.read_csv(os.path.join(subj_outdir, 'top_features.csv'))
    subj_top_indices['subject'] = subj
    all_top_indices = pd.concat((all_top_indices, subj_top_indices), axis=0)

template_img = nib.load(mni_path)
template_affine = template_img.affine
for freqband in all_top_indices['freq_band'].unique():
    plotting_df = all_top_indices[all_top_indices['freq_band'] == freqband].copy()
    # Collapse the dataframe to unique coordinates
    plotting_df = plotting_df[['subject', 'channel', 'freq_band', 'mean_diff', 'mni_x', 'mni_y', 'mni_z']].groupby(['subject', 'channel', 'freq_band']).mean().reset_index()
    all_fb_img = generate_volumetric(plotting_df, template_img_path=mni_path, weight_colname='mean_diff', smoothing_kernel=2, sphere_radius=10, return_img=True)
    nib.save(all_fb_img, os.path.join(outdir, f'important_features_{freqband}.nii.gz'))
    # Also plot a binary plot
    plotting_df['weight'] = plotting_df['mean_diff'].apply(lambda x: 1 if x > 0 else -1)
    all_fb_binary_img = generate_volumetric(plotting_df, template_img_path=mni_path, weight_colname='weight', smoothing_kernel=2, sphere_radius=10, return_img=True)
    nib.save(all_fb_binary_img, os.path.join(outdir, f'important_features_{freqband}_binary.nii.gz'))
    # Also plot a count plot
    plotting_df['weight'] = 1
    all_fb_count_img = generate_volumetric(plotting_df, template_img_path=mni_path, weight_colname='weight', smoothing_kernel=2, sphere_radius=10, return_img=False)
    # Threshold to voxels with at least N occurrences
    all_fb_count_img[all_fb_count_img < count_threshold] = 0
    all_fb_count_img = nib.Nifti1Image(all_fb_count_img, all_fb_img.affine)
    nib.save(all_fb_count_img, os.path.join(outdir, f'important_features_{freqband}_count.nii.gz'))
    # And also save a thresholded images
    all_fb_img_data = all_fb_img.get_fdata()
    all_fb_img_data[all_fb_count_img.get_fdata() == 0] = 0
    all_fb_img_thresholded = nib.Nifti1Image(all_fb_img_data, all_fb_img.affine)
    nib.save(all_fb_img_thresholded, os.path.join(outdir, f'important_features_{freqband}_thresholded.nii.gz'))
    
    # Compile a 4D image by concatenating each subject's image
    plotting_df = all_top_indices[all_top_indices['freq_band'] == freqband].copy()
    freqband_4d_img_data = np.zeros((template_img.shape[0], template_img.shape[1], template_img.shape[2], len(plotting_df['subject'].unique())))
    for subjidx, subj in enumerate(plotting_df['subject'].unique()):
        subj_outdir = os.path.join(subj_specific_outdir, subj)
        subj_img = nib.load(os.path.join(subj_outdir, f'important_features_{freqband}_binary.nii.gz'))
        freqband_4d_img_data[:, :, :, subjidx] = subj_img.get_fdata()
    freqband_4d_img = nib.Nifti1Image(freqband_4d_img_data, template_affine)
    nib.save(freqband_4d_img, os.path.join(outdir, f'{freqband}_aggregated.nii.gz'))
    # Voxel-wise one-sample t-test
    t_img_data, p_img_data = ttest_1samp(freqband_4d_img_data, 0, axis=-1)
    # Convert NaNs to 0
    t_img_data[np.isnan(t_img_data)] = 0
    t_img = nib.Nifti1Image(t_img_data, template_affine)
    nib.save(t_img, os.path.join(outdir, f'tstat_{freqband}.nii.gz'))
    