#%%
####################################################################################################
# Aggregate SHAP values for each feature across all patients
#
#
# Karim Mithani
# July 2024
####################################################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.stats import ttest_1samp
from glob import glob

import nibabel as nib
from nilearn import plotting, datasets
from nilearn.surface import vol_to_surf


# User-defined variables
analysis_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_40Hz/online/all_channels'
outdir = os.path.join(analysis_dir, 'shap_aggregated')

subjects = [x.split('/')[-1] for x in glob(os.path.join(analysis_dir, 'SEEG-*'))]

exclude_subjects = ['SEEG-SK-55', 'SEEG-SK-64', 'SEEG-SK-69']

# Miscellaneous
plt.rcParams['font.family'] = 'FreeSans'
plt.rcParams.update({'font.size': 15})
# Set the font color and axis color
custom_colour = '#3c3c3c'  # Define the dark grey color
plt.rcParams['text.color'] = custom_colour
plt.rcParams['axes.labelcolor'] = custom_colour
plt.rcParams['xtick.color'] = custom_colour
plt.rcParams['ytick.color'] = custom_colour
plt.rcParams['axes.linewidth'] = 1
gmfcs_palette = sns.palettes.color_palette('Set1', 5)
gmfcs_palette = [gmfcs_palette[1], gmfcs_palette[0]]

if not os.path.exists(outdir):
    os.makedirs(outdir)
    
####################################################################################################
# Functions
####################################################################################################


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

    # new_atlas = nib.Nifti1Image(atlas_data, atlas.affine, atlas.header)
    # big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    # big_texture = vol_to_surf(new_atlas, big_fsaverage.pial_right)
    # plot = plotting.view_surf(big_fsaverage.pial_right, big_texture, cmap=cmap, bg_map=big_fsaverage.sulc_right, threshold=threshold, symmetric_cmap=symmetric_cmap)
    # plot.save_as_html(os.path.join(outdir, f'{prefix}_right.html'))

    new_atlas = nib.Nifti1Image(atlas_data, atlas.affine, atlas.header)
    big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    big_texture = vol_to_surf(new_atlas, big_fsaverage.pial_left)
    plot = plotting.view_surf(big_fsaverage.pial_left, big_texture, cmap=cmap, bg_map=big_fsaverage.sulc_left, threshold=threshold, symmetric_cmap=symmetric_cmap)
    plot.save_as_html(os.path.join(outdir, f'{prefix}_left.html'))

#%%
####################################################################################################
# Main
####################################################################################################

all_shap_values = pd.DataFrame()

for subj in subjects:
    if subj in exclude_subjects:
        continue
    
    if not os.path.exists(os.path.join(analysis_dir, subj, 'shap', 'shap_values_aal_freqbands.csv')):
        continue

    print(f'Processing {subj}...')

    # Load data
    subj_shap_values = pd.read_csv(os.path.join(analysis_dir, subj, 'shap', 'shap_values_aal_freqbands.csv'))
    
    # # Z-score the entire dataframe
    scaler = StandardScaler()
    subj_shap_values.iloc[:,1:] = scaler.fit_transform(subj_shap_values.iloc[:,1:])
    # subj_shap_values = pd.concat((pd.DataFrame(subj_shap_values['aal_region']), np.abs(subj_shap_values.iloc[:,1:]).apply(zscore) * np.sign(subj_shap_values.iloc[:,1:])), axis=1)
    
    all_shap_values = pd.concat([all_shap_values, subj_shap_values], axis=0)

#%%

# Project all aal regions to left side
all_shap_values['aal_region'] = all_shap_values['aal_region'].apply(lambda x: x.replace('_R', '_L'))

# Aggregate SHAP values
all_shap_values_aggregated = all_shap_values.groupby('aal_region').mean().reset_index()

# Sort
all_shap_values_aggregated = all_shap_values_aggregated.sort_values(by='delta', ascending=False)

plt.figure(figsize=(10, 10))
sns.heatmap(all_shap_values_aggregated.iloc[:,1:], cmap='coolwarm', center=0, cbar_kws={'label': 'SHAP value'})
plt.xlabel('Frequency band')
plt.ylabel('AAL region')
plt.title('Aggregated SHAP values')
# Add channel labels
plt.yticks(np.arange(len(all_shap_values_aggregated['aal_region'])), all_shap_values_aggregated['aal_region'], rotation=0)
# plt.show()
plt.savefig(os.path.join(outdir, 'shap_aggregated_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

for freq_band in all_shap_values_aggregated.columns[1:]:
    plot_3d_brain(all_shap_values_aggregated[['aal_region', freq_band]], outdir, f'shap_{freq_band}', weight_label=freq_band, symmetric_cmap=True, cmap='coolwarm', threshold=0.0001)