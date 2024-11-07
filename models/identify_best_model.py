###################################################################################################################
# Search through specified hyperparameter combinations to identify the best bespoke model
#
#
# Karim Mithani
# July 2024
###################################################################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from scipy.stats import ttest_1samp
from glob import glob

# User-defined variables
analysis_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/cnn/analysis/psd_40Hz/online/using_rfe/LogisticRegression'
hyperparameters_pre = { # Hyperparameters where the hyperparam directory occurs before the subject directory
    'n_chans': '_channels'
}
hyperparameters_post = { # Hyperparameters where the hyperparam directory occurs after the subject directory
    'tp_class_weight': 'tp_weight_'
}
sort_models_by = 'combined_auc_prc' # Options include: combined_auc, combined_auc_prc, roc_auc, val_roc_auc, auc_prc, val_auc_prc, val_f1

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

#%%
##################################################################################################
# Main
##################################################################################################

model_metrics = pd.DataFrame()
predictions_df = pd.DataFrame()

for param in hyperparameters_pre:
    param_dirs = [x for x in glob(os.path.join(analysis_dir, '*' + hyperparameters_pre[param])) if os.path.isdir(x)]
    
    for param_dir in param_dirs:
        hparam = param_dir.split('/')[-1]
        
        if hparam == '25_channels': continue
        
        subjects = [x.split('/')[-1] for x in glob(os.path.join(param_dir, 'SEEG-*'))]

        for subj in subjects:
            
            if subj != 'SEEG-SK-74': continue # For debugging
            
            for hparam_post in hyperparameters_post:
                hparam_post_dirs = [x for x in glob(os.path.join(param_dir, subj, hyperparameters_post[hparam_post] + '*')) if os.path.isdir(x)]
                
                for hparam_post_dir in hparam_post_dirs:
                    
                    hparam_post_value = hparam_post_dir.split('/')[-1]
                    
                    if not os.path.exists(os.path.join(hparam_post_dir, f'{subj}_predictions.csv')) or not os.path.exists(os.path.join(hparam_post_dir, f'{subj}_validation_predictions.csv')):
                        print(f'{subj} missing predictions')
                        continue
                    if subj in exclude_subjects:
                        continue
                    subj_predictions = pd.read_csv(os.path.join(hparam_post_dir, f'{subj}_predictions.csv')).drop(columns=['Unnamed: 0'])
                    subj_predictions['subject'] = subj
                    subj_predictions['condition'] = 'holdout'
                    subj_predictions_diffday = pd.read_csv(os.path.join(hparam_post_dir, f'{subj}_validation_predictions.csv')).drop(columns=['Unnamed: 0'])
                    subj_predictions_diffday['subject'] = subj
                    subj_predictions_diffday['condition'] = 'diffday'
                    subj_df = pd.concat([subj_predictions, subj_predictions_diffday])
                    subj_df['hyperparameter_1'] = hparam
                    subj_df['hyperparameter_2'] = hparam_post_value
                    predictions_df = pd.concat([predictions_df, subj_df])   
                    
                    if not os.path.exists(os.path.join(hparam_post_dir, f'{subj}_model_metrics.csv')):
                        subj_df_holdout = subj_df[subj_df['condition'] == 'holdout']
                        subj_df_diffday = subj_df[subj_df['condition'] == 'diffday']
                        subj_auc = metrics.roc_auc_score(subj_df_holdout['truth'], subj_df_holdout['0'])
                        subj_val_auc = metrics.roc_auc_score(subj_df_diffday['truth'], subj_df_diffday['0'])
                        subj_auc_prc = metrics.average_precision_score(subj_df_holdout['truth'], subj_df_holdout['0'])
                        subj_val_auc_prc = metrics.average_precision_score(subj_df_diffday['truth'], subj_df_diffday['0'])
                        subj_f1 = metrics.f1_score(subj_df_diffday['truth'], subj_df_diffday['0'] > 0.5)
                        # # Uncomment if needed:
                        # ## Make sure to also adjust the dataframe columns if uncommented
                        # subj_accuracy = metrics.accuracy_score(subj_df['truth'], subj_df['0'] > 0.5)
                        # subj_confmat = metrics.confusion_matrix(subj_df['truth'], subj_df['0'] > 0.5)
                        # subj_model_metrics = pd.DataFrame({'roc_auc': [subj_auc], 'accuracy': [subj_accuracy], 'confusion_matrix': [subj_confmat]})
                        subj_model_metrics = pd.DataFrame({'roc_auc': [subj_auc], 'val_roc_auc': [subj_val_auc], 'auc_prc': [subj_auc_prc], 'val_auc_prc': [subj_val_auc_prc], 'val_f1': [subj_f1]})
                        subj_model_metrics['subject'] = subj
                        subj_model_metrics['hyperparameter_1'] = hparam
                        subj_model_metrics['hyperparameter_2'] = hparam_post_value
                        model_metrics = pd.concat([model_metrics, subj_model_metrics])
                    else:
                        subj_model_metrics = pd.read_csv(os.path.join(param_dir, subj, f'{subj}_model_metrics.csv')).drop(columns=['Unnamed: 0'])
                        subj_model_metrics['subject'] = subj
                        subj_model_metrics['hyperparameter_1'] = hparam
                        subj_model_metrics['hyperparameter_2'] = hparam_post_value
                        model_metrics = pd.concat([model_metrics, subj_model_metrics])

using_rfe = 'using_rfe' in analysis_dir
model_metrics['using_rfe'] = using_rfe
if using_rfe:
    rfe_method = analysis_dir.split('/')[-1]
    model_metrics['rfe_method'] = rfe_method

model_metrics['combined_auc'] = model_metrics['roc_auc'] * model_metrics['val_roc_auc']
model_metrics['combined_auc_prc'] = model_metrics['auc_prc'] * model_metrics['val_auc_prc']

top_models = model_metrics.groupby('subject').apply(lambda x: x.sort_values(sort_models_by, ascending=False).head(1)).reset_index(drop=True)

top_models_dir = os.path.join(analysis_dir, 'top_models')
if not os.path.exists(top_models_dir):
    os.makedirs(top_models_dir)

model_metrics.to_csv(os.path.join(top_models_dir, 'all_model_metrics.csv'), index=False)
predictions_df.to_csv(os.path.join(top_models_dir, 'all_predictions.csv'), index=False)
# Add path to model
for rowidx, row in top_models.iterrows():
    top_models.loc[rowidx, 'model_path'] = os.path.join(analysis_dir, row['hyperparameter_1'], row['subject'], row['hyperparameter_2'])

top_models.to_csv(os.path.join(top_models_dir, 'top_models.csv'), index=False)

if '1' not in predictions_df.columns:
    predictions_df = predictions_df.rename(columns={'0': '1'})
    predictions_df['0'] = 1 - predictions_df['1']

top_model_predictions = pd.DataFrame()
for subj in top_models['subject']:
    top_hparam_1 = top_models[top_models['subject'] == subj]['hyperparameter_1'].values[0]
    top_hparam_2 = top_models[top_models['subject'] == subj]['hyperparameter_2'].values[0]
    subj_df_top = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['hyperparameter_1'] == top_hparam_1) & (predictions_df['hyperparameter_2'] == top_hparam_2)]
    top_model_predictions = pd.concat([top_model_predictions, subj_df_top])
 
top_model_predictions.to_csv(os.path.join(top_models_dir, 'top_model_predictions.csv'), index=False)
  
predictions_df = top_model_predictions # To allow compatibility with older plotting code 

#%%

# Plot precision-recall curve
y_preds_all = []
y_all = []
auprc_all = []
prediction_thresholds = {}
plt.figure(figsize=(8, 6), dpi=300)
for subj in predictions_df['subject'].unique():
    # Plot ROC curve
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')]['truth'].values
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred[:, 1])
    # Identify the optimal prediction threshold for each subject
    f1 = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.nanargmax(f1)]
    prediction_thresholds[subj] = optimal_threshold
    y_preds_all.append(y_pred)
    y_all.append(y)
    auprc_all.append(metrics.auc(recall, precision))
    plt.plot(recall, precision, color='gray', alpha=0.2)
mean_precision, mean_recall, _ = metrics.precision_recall_curve(np.concatenate(y_all), np.concatenate(y_preds_all)[:, 1])
auprc_noskill = np.sum(y) / len(y)
plt.axhline(auprc_noskill, linestyle='--', color='black', label='No Skill')
t_auprc, p_auprc = ttest_1samp(auprc_all, auprc_noskill)
if p_auprc < 0.05:
    plt.text(0.95, 0.05, f'AUPRC = {np.mean(auprc_all):.2f}* ± {np.std(auprc_all):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUPRC = {np.mean(auprc_all):.2f} ± {np.std(auprc_all):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.plot(mean_recall, mean_precision, color='darkred', label='Mean Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Holdout Set')
sns.despine()
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6), dpi=300)
y_preds_all = []
y_all = []
for subj in predictions_df['subject'].unique():
    # Plot ROC curve
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')]['truth'].values
    fpr, tpr, _ = metrics.roc_curve(y, y_pred[:, 1])
    y_preds_all.append(y_pred)
    y_all.append(y)
    plt.plot(fpr, tpr, color='gray', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--')
mean_fpr, mean_tpr, _ = metrics.roc_curve(np.concatenate(y_all), np.concatenate(y_preds_all)[:, 1])
plt.plot(mean_fpr, mean_tpr, color='darkred', label='Mean ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Holdout Set')
t, p = ttest_1samp(top_models['roc_auc'], 0.5)
if p < 0.05:
    plt.text(0.95, 0.05, f'AUC = {top_models["roc_auc"].mean():.2f}* ± {top_models["roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUC = {top_models["roc_auc"].mean():.2f} ± {top_models["roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
sns.despine()
plt.show()

# Calculate F1 score
y_preds_all = []
y_all = []
f1_all = []
for subj in predictions_df['subject'].unique():
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')]['truth'].values
    f1 = metrics.f1_score(y, y_pred[:, 1] > prediction_thresholds[subj])
    y_preds_all.append(y_pred)
    y_all.append(y)
    f1_all.append(f1)
print(f'Mean F1 for Holdout: {np.mean(f1_all):.2f} ± {np.std(f1_all):.2f}')
t, p = ttest_1samp(f1_all, 0.5)
print(f'One-sample t-test for F1 score from 0.5: t = {t}, p = {p}')
    

#%% Summarize model performance on different day dataset
# print(f'\nMean diff-day AUC: {model_metrics["val_roc_auc"].mean()} ± {model_metrics["val_roc_auc"].std()}')
# t, p = ttest_1samp(model_metrics['val_roc_auc'], 0.5)
# print(f'One-sample t-test: t = {t}, p = {p}')

# Plot precision-recall curve
y_preds_all = []
y_all = []
auprc_all = []
plt.figure(figsize=(8, 6), dpi=300)
for subj in predictions_df['subject'].unique():
    # Plot ROC curve
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')]['truth'].values
    precision, recall, _ = metrics.precision_recall_curve(y, y_pred[:, 1])
    y_preds_all.append(y_pred)
    y_all.append(y)
    auprc_all.append(metrics.auc(recall, precision))
    plt.plot(recall, precision, color='gray', alpha=0.2)
mean_precision, mean_recall, _ = metrics.precision_recall_curve(np.concatenate(y_all), np.concatenate(y_preds_all)[:, 1])
auprc_noskill = np.sum(y) / len(y)
plt.axhline(auprc_noskill, linestyle='--', color='black', label='No Skill')
t_auprc, p_auprc = ttest_1samp(auprc_all, auprc_noskill)
if p_auprc < 0.05:
    plt.text(0.95, 0.05, f'AUPRC = {np.mean(auprc_all):.2f}* ± {np.std(auprc_all):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUPRC = {np.mean(auprc_all):.2f} ± {np.std(auprc_all):.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.plot(mean_recall, mean_precision, color='darkred', label='Mean Precision-Recall curve')
plt.plot(mean_recall, mean_precision, color='darkred', label='Mean Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Different Day')
sns.despine()
plt.show()

y_preds_all = []
y_all = []
plt.figure(figsize=(8, 6), dpi=300)
for subj in predictions_df['subject'].unique():
    # Plot ROC curve
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')]['truth'].values
    # auc = metrics.roc_auc_score(y, np.argmax(y_pred, axis=1))
    fpr, tpr, _ = metrics.roc_curve(y, y_pred[:, 1])
    y_preds_all.append(y_pred)
    y_all.append(y)
    plt.plot(fpr, tpr, color='gray', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # # Add legend
    # plt.legend([f'ROC AUC: {metrics.roc_auc_score(y, y_pred[:, 1])}'])
mean_fpr, mean_tpr, _ = metrics.roc_curve(np.concatenate(y_all), np.concatenate(y_preds_all)[:, 1])
plt.plot(mean_fpr, mean_tpr, color='darkred', label='Mean ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Different Day')
t, p = ttest_1samp(top_models['val_roc_auc'], 0.5)
if p < 0.05:
    plt.text(0.95, 0.05, f'AUC = {top_models["val_roc_auc"].mean():.2f}* ± {top_models["val_roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUC = {top_models["val_roc_auc"].mean():.2f} ± {top_models["val_roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
sns.despine()
plt.show()

# Calculate F1 score
y_preds_all = []
y_all = []
f1_all = []
for subj in predictions_df['subject'].unique():
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')]['truth'].values
    f1 = metrics.f1_score(y, y_pred[:, 1] > prediction_thresholds[subj])
    y_preds_all.append(y_pred)
    y_all.append(y)
    f1_all.append(f1)
print(f'Mean F1 for Different Day: {np.mean(f1_all):.2f} ± {np.std(f1_all):.2f}')
t, p = ttest_1samp(f1_all, 0.5)
print(f'One-sample t-test for F1 score from 0.5: t = {t}, p = {p}')

