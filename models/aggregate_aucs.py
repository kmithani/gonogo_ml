#%%
##################################################################################################
# Quick script to aggregate AUCs across subjects
#
#
# Karim Mithani
# June 2024
##################################################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from scipy.stats import ttest_1samp
from glob import glob

# User-defined variables
analysis_dir = '/d/gmi/1/karimmithani/seeg/analysis/gonogo/models/svm/analysis/psd_40Hz/online/using_rfe/RandomForest/'

subjects = [x.split('/')[-1] for x in glob(os.path.join(analysis_dir, 'SEEG-*'))]

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

##################################################################################################
# Main
##################################################################################################

model_metrics = pd.DataFrame()
predictions_df = pd.DataFrame()
# for subj in subjects:
#     if not os.path.exists(os.path.join(analysis_dir, subj, f'{subj}_model_metrics.csv')):
#         continue
#     subj_model_metrics = pd.read_csv(os.path.join(analysis_dir, subj, f'{subj}_model_metrics.csv')).drop(columns=['Unnamed: 0'])
#     subj_model_metrics['subject'] = subj
#     model_metrics = pd.concat([model_metrics, subj_model_metrics])

for subj in subjects:
    if not os.path.exists(os.path.join(analysis_dir, subj, f'{subj}_predictions.csv')) or not os.path.exists(os.path.join(analysis_dir, subj, f'{subj}_validation_predictions.csv')):
        print(f'{subj} missing predictions')
        continue
    subj_predictions = pd.read_csv(os.path.join(analysis_dir, subj, f'{subj}_predictions.csv')).drop(columns=['Unnamed: 0'])
    subj_predictions['subject'] = subj
    subj_predictions['condition'] = 'holdout'
    subj_predictions_diffday = pd.read_csv(os.path.join(analysis_dir, subj, f'{subj}_validation_predictions.csv')).drop(columns=['Unnamed: 0'])
    subj_predictions_diffday['subject'] = subj
    subj_predictions_diffday['condition'] = 'diffday'
    subj_df = pd.concat([subj_predictions, subj_predictions_diffday])
    predictions_df = pd.concat([predictions_df, subj_df])   
    
    if not os.path.exists(os.path.join(analysis_dir, subj, f'{subj}_model_metrics.csv')):
        subj_df_holdout = subj_df[subj_df['condition'] == 'holdout']
        subj_df_diffday = subj_df[subj_df['condition'] == 'diffday']
        subj_auc = metrics.roc_auc_score(subj_df_holdout['truth'], subj_df_holdout['0'])
        subj_val_auc = metrics.roc_auc_score(subj_df_diffday['truth'], subj_df_diffday['0'])
        # # Uncomment if needed:
        # ## Make sure to also adjust the dataframe columns if uncommented
        # subj_accuracy = metrics.accuracy_score(subj_df['truth'], subj_df['0'] > 0.5)
        # subj_confmat = metrics.confusion_matrix(subj_df['truth'], subj_df['0'] > 0.5)
        # subj_model_metrics = pd.DataFrame({'roc_auc': [subj_auc], 'accuracy': [subj_accuracy], 'confusion_matrix': [subj_confmat]})
        subj_model_metrics = pd.DataFrame({'roc_auc': [subj_auc], 'val_roc_auc': [subj_val_auc]})
        model_metrics = pd.concat([model_metrics, subj_model_metrics])
    else:
        subj_model_metrics = pd.read_csv(os.path.join(analysis_dir, subj, f'{subj}_model_metrics.csv')).drop(columns=['Unnamed: 0'])
        subj_model_metrics['subject'] = subj
        model_metrics = pd.concat([model_metrics, subj_model_metrics])

if '1' not in predictions_df.columns:
    predictions_df = predictions_df.rename(columns={'0': '1'})
    predictions_df['0'] = 1 - predictions_df['1']
    
#%% Summarize model performance on holdout set

# Plot precision-recall curve
y_preds_all = []
y_all = []
auprc_all = []
plt.figure(figsize=(8, 6), dpi=300)
for subj in predictions_df['subject'].unique():
    # Plot ROC curve
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')]['truth'].values
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
t, p = ttest_1samp(model_metrics['roc_auc'], 0.5)
if p < 0.05:
    plt.text(0.95, 0.05, f'AUC = {model_metrics["roc_auc"].mean():.2f}* ± {model_metrics["roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUC = {model_metrics["roc_auc"].mean():.2f} ± {model_metrics["roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
sns.despine()
plt.show()

# Calculate F1 score
y_preds_all = []
y_all = []
f1_all = []
for subj in predictions_df['subject'].unique():
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'holdout')]['truth'].values
    f1 = metrics.f1_score(y, y_pred[:, 1] > 0.5)
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
t, p = ttest_1samp(model_metrics['val_roc_auc'], 0.5)
if p < 0.05:
    plt.text(0.95, 0.05, f'AUC = {model_metrics["val_roc_auc"].mean():.2f}* ± {model_metrics["val_roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
else:
    plt.text(0.95, 0.05, f'AUC = {model_metrics["val_roc_auc"].mean():.2f} ± {model_metrics["val_roc_auc"].std():.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
sns.despine()
plt.show()

# Calculate F1 score
y_preds_all = []
y_all = []
f1_all = []
for subj in predictions_df['subject'].unique():
    y_pred = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')][['0','1']].values
    y = predictions_df[(predictions_df['subject'] == subj) & (predictions_df['condition'] == 'diffday')]['truth'].values
    f1 = metrics.f1_score(y, y_pred[:, 1] > 0.5)
    y_preds_all.append(y_pred)
    y_all.append(y)
    f1_all.append(f1)
print(f'Mean F1 for Different Day: {np.mean(f1_all):.2f} ± {np.std(f1_all):.2f}')
t, p = ttest_1samp(f1_all, 0.5)
print(f'One-sample t-test for F1 score from 0.5: t = {t}, p = {p}')
