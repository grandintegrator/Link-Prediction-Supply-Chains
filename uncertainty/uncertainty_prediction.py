# import parquet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    average_precision_score
)
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import plotly
plotly.offline.init_notebook_mode()


def calc_auroc(nll_inliers_train_mean, nll_inliers_valid_mean):
    y_true = np.concatenate((np.zeros(nll_inliers_train_mean.shape[0]),
                             np.ones(nll_inliers_valid_mean.shape[0])))
    y_scores = np.concatenate((nll_inliers_train_mean, nll_inliers_valid_mean))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


# validation_res = pd.read_parquet('validation_frame.parquet', engine='fastparquet')
# validation_res = pd.read_parquet('training_frame.parquet', engine='fastparquet')

base_store_path = 'Link-Prediction-Supply-Chains/data/03_models/'
valid_batches = (
    pd.read_parquet(base_store_path + 'validation_frame.parquet')
)

valid_batches['MODEL_UNCERTAINTY'] = (
    valid_batches['MODEL_SCORE']*(1-valid_batches['MODEL_SCORE'])
)

validation_res = valid_batches

# apply correction to scale to [0,1]
# validation_res["MODEL_SCORE"] = (validation_res["MODEL_SCORE"] - 0.5) / 0.5
validation_res["UNCERTAINTY"] = \
    validation_res["MODEL_SCORE"]*(1-validation_res["MODEL_SCORE"])

# Behaviour of model score vs uncertainty
plt.figure()
plt.scatter(validation_res["MODEL_SCORE"], validation_res["UNCERTAINTY"])
plt.xlabel("Model Score")
plt.ylabel("Uncertainty")
plt.show()


# Select link index
link_index =3 # {0-6}
link_types = validation_res["LINK_TYPE"].unique()
subset_res = validation_res[validation_res["LINK_TYPE"] == link_types[link_index]]

score_label_0 = subset_res[subset_res["LABELS"]==0]["MODEL_SCORE"]
score_label_1 = subset_res[subset_res["LABELS"]==1]["MODEL_SCORE"]

print("POSITIVE CLASS RATIO : {:.2f}".format(subset_res["MODEL_SCORE"].mean()))

# Histogram of class probability
# plt.figure()
# plt.hist(score_label_0, density=True)
# plt.hist(score_label_1, density=True)

# plt.figure()
# plt.xlim(0, 1)
# sns.kdeplot(score_label_0)
# sns.kdeplot(score_label_1)
# plt.legend(["LABEL 0", "LABEL 1"])
# # plt.title(link_types[link_index])
# plt.show()
# auroc = calc_auroc(score_label_0, score_label_1)
# print("CLASS AUROC : {:.3f} ".format(np.round(auroc, 3)))

#-----Uncertainty Analysis------
y_unc = subset_res["UNCERTAINTY"].values

# set classification threshold to get binary predictions
prob_threshold = 0.5
y_hard_pred = subset_res["MODEL_SCORE"].values
y_hard_pred[np.argwhere(y_hard_pred < prob_threshold)[:, 0]] = 0
y_hard_pred[np.argwhere(y_hard_pred >= prob_threshold)[:, 0]] = 1
y_hard_pred = y_hard_pred.astype(int)

y_true = subset_res["LABELS"].values.astype(int)

indices_tp =  np.argwhere((y_true ==1) & (y_hard_pred ==1))[:,0]
indices_tn =  np.argwhere((y_true ==0) & (y_hard_pred ==0))[:,0]
indices_fp =  np.argwhere((y_true ==0) & (y_hard_pred ==1))[:,0]
indices_fn =  np.argwhere((y_true ==1) & (y_hard_pred ==0))[:,0]
indices_0_error = np.concatenate((indices_tp,indices_tn))
indices_all_error = np.concatenate((indices_fp,indices_fn))

error_type1 = np.concatenate((np.ones(len(indices_fp)), np.zeros(len(indices_tp)))).astype(int)
error_type2 = np.concatenate((np.ones(len(indices_fn)), np.zeros(len(indices_tn)))).astype(int)
error_all = np.abs((y_true - y_hard_pred))

y_unc_type1 = np.concatenate((y_unc[indices_fp], y_unc[indices_tp]))
y_unc_type2 = np.concatenate((y_unc[indices_fn], y_unc[indices_tn]))
y_unc_all = y_unc

precision_type1, recall_type1, thresholds = precision_recall_curve(error_type1, y_unc_type1)
precision_type2, recall_type2, thresholds = precision_recall_curve(error_type2, y_unc_type2)
precision_type_all, recall_type_all, thresholds = precision_recall_curve(error_all, y_unc)

auprc_type1 = auc(recall_type1, precision_type1)
auprc_type2 = auc(recall_type2, precision_type2)
auprc_type_all = auc(recall_type_all, precision_type_all)

baseline_type1 = error_type1.mean()
baseline_type2 = error_type2.mean()
baseline_all = error_all.mean()

lift_type1 = auprc_type1/ baseline_type1
lift_type2 = auprc_type2/ baseline_type2
lift_all = auprc_type_all/ baseline_all

# fig, ax1= plt.subplots(1,1)
# ax1.boxplot([y_unc[indices_tp],
#              y_unc[indices_tn],
#              y_unc[indices_0_error],
#              y_unc[indices_fp],
#              y_unc[indices_fn],
#              y_unc[indices_all_error],
#              ]
#             ,
#             [0,1,2,3,4],''
#             )
#
# ax1.set_xticklabels(["TP","TN","TP+TN","Type 1 (FP)","Type 2 (FN)", "FP+FN"])
# ax1.set_ylabel("Uncertainty")
# ax1.set_title(link_types[link_index])
# plt.show()
# print("AUPRC-TYPE1 (TP VS FP): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type1, baseline_type1, lift_type1))
# print("AUPRC-TYPE2 (TN VS FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type2, baseline_type2, lift_type2))
# print("AUPRC-ALL   (TP+TN VS FP+FN): {:.2f} , BASELINE: {:.3f}, AUPRC-RATIO: {:.2f}".format(auprc_type_all, baseline_all, lift_all))
# print(link_types)




# data with different sizes
# x1 = np.random.randn(300)-2
# x2 = np.random.randn(200)
# x3 = np.random.randn(4000)+2
# x4 = np.random.randn(50)+4

# Group data together
hist_data = [y_unc[indices_tp],
             y_unc[indices_tn],
             y_unc[indices_0_error],
             y_unc[indices_fp],
             y_unc[indices_fn],
             y_unc[indices_all_error]]
# hist_data = [x1, x2, x3, x4]

# use custom names
group_labels = ["TP", "TN", "TP+TN", "Type 1 (FP)", "Type 2 (FN)", "FP+FN"]

# Create distplot with custom bin_size
pio.templates.default = "plotly_white"
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(font_family='Arial',
                  yaxis_title='Uncertainty',
                  xaxis_title='Type',
                  font=dict(size=24))
home_dir = 'Link-Prediction-Supply-Chains/data/04_results/'
fig.write_html(home_dir + 'uncertainty_type.html')

# change that if you don't want to plot offline

fig = go.Figure()
for trace, label in zip([y_unc[indices_tp],
                         y_unc[indices_tn],
                         y_unc[indices_0_error],
                         y_unc[indices_fp],
                         y_unc[indices_fn],
                         y_unc[indices_all_error]], group_labels):
    fig.add_trace(
        go.Violin(y=trace,
                  box_visible=True,
                  meanline_visible=True,
                  x0=label)
    )

title_1 = f'Model Uncertainty for {link_types[link_index]} '
title_2 = "POSITIVE CLASS RATIO : {:.2f}".format(subset_res["MODEL_SCORE"].mean())
fig.update_layout(font_family='Arial',
                  title=title_1 + title_2,
                  yaxis_title='Uncertainty',
                  xaxis_title='Type',
                  font=dict(size=24),
                  yaxis_zeroline=False,
                  violingap=0,
                  violinmode='overlay',
                  yaxis_range=[0, 1]
)

fig.write_html(home_dir + 'uncertainty_type_1.html')
