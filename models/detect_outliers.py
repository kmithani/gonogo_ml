#%%
####################################################################################################
# Detect subjects with invalid data based on reaction time / accuracy trade-off
#
#
# Karim Mithani
# July 2024
####################################################################################################

import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

# User-defined variables
processed_dir = '/d/gmi/1/karimmithani/seeg/processed'
# outdir = f'/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_{fmax}Hz'
labels_dir = '/d/gmi/1/karimmithani/seeg/labels'
cles_array_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/cles'
n_chans = [10, 15, 20, 25] # Hyperparameter for number of channels when using RFE

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

#%%
#######################################################################################################################################
# Main
#######################################################################################################################################

summary_df = pd.DataFrame()

for idx, subj in enumerate(subjects):
    
    # if idx == 2: break # For debugging
    if subj not in validation_data.keys(): continue # For debugging    
    
    #%%
    print(f'\nProcessing {subj}...')
    
    subj_summary_df = pd.DataFrame()
    
    task_summary = pd.DataFrame()
    for day in subjects[subj]:
        for task in subjects[subj][day]:
            print(f'Processing {subj} - {day} - {task}...')
            # Load task summary
            specific_task_summary = pd.read_csv(os.path.join(processed_dir, subj, day, task, f'task_summary.csv'))
            specific_task_summary['Day'] = day
            specific_task_summary['Task'] = task
            task_summary = pd.concat((task_summary, specific_task_summary))
            task_accuracy = pd.read_pickle(os.path.join(processed_dir, subj, day, task, f'accuracy.pkl'))

    # Calculate aggregate statistics
    subj_summary_df.loc[idx, 'id'] = subj
    subj_summary_df.loc[idx, 'rt_mean_go'] = task_summary[task_summary['event']=='Go Correct']['rt'].mean()
    subj_summary_df.loc[idx, 'rt_mean_nogo'] = task_summary[task_summary['event']=='Nogo Incorrect']['rt'].mean()
    subj_summary_df.loc[idx, 'go_accuracy'] = task_accuracy['go_accuracy']
    subj_summary_df.loc[idx, 'nogo_accuracy'] = task_accuracy['nogo_accuracy']
    subj_summary_df.loc[idx, 'n_go_correct'] = len(task_summary[task_summary['event']=='Go Correct'])
    subj_summary_df.loc[idx, 'n_nogo_incorrect'] = len(task_summary[task_summary['event']=='Nogo Incorrect'])
    subj_summary_df.loc[idx, 'ratio_nogo_go'] = subj_summary_df.loc[idx, 'n_nogo_incorrect'] / subj_summary_df.loc[idx, 'n_go_correct']
    subj_summary_df.loc[idx, 'ratio_rt'] = subj_summary_df.loc[idx, 'rt_mean_nogo'] / subj_summary_df.loc[idx, 'rt_mean_go']
    subj_summary_df.loc[idx, 'nogo_speed_accuracy_tradeoff'] = subj_summary_df.loc[idx, 'rt_mean_nogo'] * subj_summary_df.loc[idx, 'nogo_accuracy']
    
    summary_df = pd.concat((summary_df, subj_summary_df))
    
    # validation_task_summary = pd.DataFrame()
    # for day in validation_data[subj]:
    #     for task in validation_data[subj][day]:
    #         print(f'Processing {subj} - {day} - {task}...')
    #         # Load task summary
    #         specific_task_summary = pd.read_csv(os.path.join(processed_dir, subj, day, task, f'task_summary.csv'))
    #         specific_task_summary['Day'] = day
    #         specific_task_summary['Task'] = task
    #         validation_task_summary = pd.concat((validation_task_summary, specific_task_summary))
    
    # subj_task_summary = pd.concat((task_summary, validation_task_summary))
    
