from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

################################################################################
# Plotting options
################################################################################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Seaborn parameters
sns.set_context("paper", font_scale=1.8)
sns.set_style('ticks')
# Use LaTeX
plt.rcParams.update({
    "text.usetex": True,
})

################################################################################
# Analysis of AUC plots...mimicking the curves from weights and biases
################################################################################
# Depending on which link type I am running + saving the plot for
# ++ Which run was most fruitful.
links = ['makes_product',
         'has_cert',
         'has_capability',
         'capability_produces',
         'complimentary_product_to',
         'located_in',
         'buys_from']

config = dict(
    best_run='confused-blaze-99',
    link_type='Training AUC located_in'
)

# Import the results frame for that specific run.
home_dir = 'Link-Prediction-Supply-Chains/data/04_results/wanb_csv/'
results_frame = pd.read_csv(home_dir+'training_auc_' + 'located_in' + '.csv')

# Rename the column to make it easier to work with.
rename_mapper = {
    config['best_run'] + ' - ' + config['link_type']: 'y'
}
results_frame = (
    results_frame.rename(columns=rename_mapper)
)

# Smooth the curve
results_frame['smoothed'] = gaussian_filter1d(results_frame['y'], sigma=2)

# Plot the lineplot.
plt.figure(figsize=(10, 3.5))
ax = sns.lineplot(
    x='Step',
    y='value', hue='variable',
    data=results_frame[['Step', 'y', 'smoothed']].melt('Step'),
    palette={'y': '#bad6eb', 'smoothed': '#0b559f'}
)

plt.tight_layout()
plt.xlabel(r'Step')
plt.ylabel(r'AUC Value')
plt.title(r'Training AUC for link type \textbf{locatin$\_$in}')
plt.tight_layout()
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.2)
plt.legend([r'Batch $\mathcal{B}$ AUC Value', r'Gaussian Smoothed $\sigma = 2$'])
plt.savefig(home_dir + f"{config['link_type']}_saved.png",
            bbox_inches='tight')
plt.show()

################################################################################
# Parallel coordinate plots:
################################################################################
# pio.templates.default = "plotly_white"
# parallel_df = pd.read_csv(home_dir + '/parallel_coordinate_data.csv')

# fig = px.parallel_coordinates(parallel_df.drop(columns=['Name'], axis=1),
#                               color='Training AUC buys_from',
#                               color_continuous_scale=px.colors.sequential.ice)

# title = 'Log Log of Weight Distribution for (Product -> Product Graph)'
# fig.update_layout(font_family='Arial',
#                   title=title,
#                   yaxis_title='Fraction of Edges (log)',
#                   xaxis_title='Weight (log)',
#                   font=dict(size=24))

# Numerical columns

# colour_line = parallel_df['Training AUC buys_from']
#
# parallel_df = parallel_df.drop(columns=['Name', 'Training AUC buys_from'],
#                                axis=1)
# numerical_coordinates = \
#     [col for col in parallel_df.columns if parallel_df[col].dtype != object]
#
# numerical_dicts = [dict(range=[parallel_df[col].min(), parallel_df[col].max()],
#                         label=col,
#                         values=parallel_df[col].values)
#                    for col in numerical_coordinates]
#
#
# categorical_coordinates = \
#     [col for col in parallel_df.columns if col not in numerical_coordinates]
#
#
# categorical_dicts = [
#     dict(range=[min(list(range(len(parallel_df[col].unique())))),
#                 max(list(range(len(parallel_df[col].unique()))))],
#          tickvals=list(range(len(parallel_df[col].unique()))),
#          label=col,
#          values=parallel_df[col].values,
#          ticktext=list(parallel_df[col].unique())
#          )
#     for col in categorical_coordinates
# ]
#
# fig = go.Figure(data=go.Parcoords(
#         line=dict(color=colour_line,
#                   colorscale='Electric',
#                   showscale=True),
#         dimensions=list([numerical_dicts[i] for i in range(len(numerical_dicts))])
#     )
# )
#
# fig.update_layout(font_family='Arial',
#                   # title=title,
#                   # yaxis_title='Tuning Parameter',
#                   # xaxis_title='Tuning Para',
#                   font=dict(size=24))
#
# fig.write_html(home_dir + '/parallel_coordinate_plot.html')
#

