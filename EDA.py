#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:46:53 2022

@author: thomasmernagh
"""


#rudimental EDA
#Importing dataset, checking for outliers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import geopandas
import pysal
import contextily
from sklearn.cluster import DBSCAN

plt.style.use('bmh')


df = pd.read_csv("provincie-utrecht_2021_04_12_2021_04_19.csv")
print (df)

df.head()

df.info() #The dataset is full, just good practice to check for nulls
# df.count() does not include NaN values
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
print("List of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
print('\n')
df = df2
#%%

#Investigating distribution of sensors over sequence

print(df['trip_sequence'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['trip_sequence'], color='g', bins=100, hist_kws={'alpha': 0.4});

#Numerical data distribution
list(set(df.dtypes.tolist()))

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations

#Possible correlations
df_num_corr = df_num.corr()['trip_sequence'][:-14] 
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with trip_sequence:\n{}".format(len(golden_features_list), golden_features_list))

for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['trip_sequence'])


#So now lets remove these 0 values and repeat the process of finding correlated values:
import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 14): # 
    tmpDf = df_num[[df_num.columns[i], 'trip_sequence']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['trip_sequence'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))
    
#Removing zero values didn't make a difference! 
#%%
#Feature to feature relationship
corr = df_num.drop('trip_sequence', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


#https://geographicdata.science/book/notebooks/08_point_pattern_analysis.html
#Further spatial EDA

#Dealing with weights
from pysal.lib import weights
utrecht = geopandas.read_file('UtrechtW15.shp')
w_queen = weights.contiguity.Queen.from_dataframe(utrecht)

# Plot tract geography
f, axs = plt.subplots(1,2,figsize=(8,4))
for i in range(2):
    ax = utrecht.plot(edgecolor='k', facecolor='w', ax=axs[i])
    # Plot graph connections
    w_queen.plot(
        utrecht, 
        ax=axs[i], 
        edge_kws=dict(color='r', linestyle=':', linewidth=1),
        node_kws=dict(marker='')
    )
# Remove the axis
    axs[i].set_axis_off()
axs[1].axis([-13040000,  -13020000, 3850000, 3860000]);

print(w_queen.n)
print(w_queen.pct_nonzero)

s = pd.Series(w_queen.cardinalities)
s.plot.hist(bins=s.unique().shape[0]);

w_rook = weights.contiguity.Rook.from_dataframe(utrecht)
print(w_rook.pct_nonzero)
s = pd.Series(w_rook.cardinalities)
s.plot.hist(bins=s.unique().shape[0]);

#Kernal weights
#They reflect the case where similarity/spatial proximity is assumed or expected to decay with distance. 
w_kernel = weights.distance.Kernel.from_dataframe(df)

# Show the first five values of bandwidths
w_kernel.bandwidth[0:5]

# Create subset of tracts
sub_30 = utrecht.query("sub_30 == True")
# Plot polygons
ax = sub_30.plot(facecolor='w', edgecolor='k')
# Create and plot centroids
sub_30.head(30).centroid.plot(color='r', ax=ax)
# Remove axis
ax.set_axis_off();
# Build weights with adaptive bandwidth
w_adaptive = weights.distance.Kernel.from_dataframe(
    sub_30,fixed=False, k=15
)
# Print first five bandwidth values
w_adaptive.bandwidth[:5]

# Create full matrix version of weights
full_matrix, ids = w_adaptive.full()
# Set up figure with two subplots in a row
f,ax = plt.subplots(
    1, 2, figsize=(12,6), subplot_kw=dict(aspect='equal')
)
# Append weights for first polygon and plot on first subplot
sub_30.assign(
    weight_0 = full_matrix[0]
).plot("weight_0", cmap='plasma', ax=ax[0])
# Append weights for 18th polygon and plot on first subplot
sub_30.assign(
    weight_18 = full_matrix[17]
).plot("weight_18", cmap='plasma', ax=ax[1])
# Add centroid of focal tracts
sub_30.iloc[[0], :].centroid.plot(
    ax=ax[0], marker="*", color="k", label='Focal Tract'
)
sub_30.iloc[[17], :].centroid.plot(
    ax=ax[1], marker="*", color="k", label='Focal Tract'
)
# Add titles
ax[0].set_title("Kernel centered on first tract")
ax[1].set_title("Kernel centered on 18th tract")
# Remove axis
[ax_.set_axis_off() for ax_ in ax]
# Add legend
[ax_.legend(loc='upper left') for ax_ in ax];
#%%
#Global Spatial Autocorrelation - https://geographicdata.science/book/notebooks/06_spatial_autocorrelation.html
# Graphics
import matplotlib.pyplot as plt
import seaborn
from pysal.viz import splot
from splot.esda import plot_moran
import contextily
# Analysis
import geopandas
import pandas
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed
#%%
# Generate scatter plot
sns.jointplot(x='lon', y='lat', data=df, s=0.5);

# Generate scatter plot
joint_axes = sns.jointplot(
    x='lon', y='lat', data=df, s=0.5
)
contextily.add_basemap(
    joint_axes.ax_joint,
    crs="EPSG:4326",
    source=contextily.providers.CartoDB.PositronNoLabels
);
#%%

#Showing density with hex-binning
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(12, 9))
# Generate and add hexbin with 50 hexagons in each 
# dimension, no borderlines, half transparency,
# and the reverse viridis colormap
hb = ax.hexbin(
    df['lon'], 
    df['lat'],
    gridsize=50, 
    linewidths=0,
    alpha=0.5, 
    cmap='viridis_r'
)
# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
# Add colorbar
plt.colorbar(hb)
# Remove axes
ax.set_axis_off()

#Kernal Density Estimation - KDE
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Generate and add KDE with a shading of 50 gradients 
# coloured contours, 75% of transparency,
# and the reverse viridis colormap
sns.kdeplot(
    df['lon'], 
    df['lat'],
    n_levels=50, 
    shade=True,
    alpha=0.55, 
    cmap='viridis_r'
)
# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
#%%

#Centrography
#1 - Tendency
from pointpats import centrography

mean_center = centrography.mean_center(df[['lon', 'lat']])
med_center = centrography.euclidean_median(df[['lon', 'lat']])

# Generate scatter plot
joint_axes = sns.jointplot(
    x='lon', y='lat', data=df, s=0.75, height=9
)
# Add mean point and marginal lines
joint_axes.ax_joint.scatter(
    *mean_center, color='red', marker='x', s=50, label='Mean Center'
)
joint_axes.ax_marg_x.axvline(mean_center[0], color='red')
joint_axes.ax_marg_y.axhline(mean_center[1], color='red')
# Add median point and marginal lines
joint_axes.ax_joint.scatter(
    *med_center, color='limegreen', marker='o', s=50, label='Median Center'
)
joint_axes.ax_marg_x.axvline(med_center[0], color='limegreen')
joint_axes.ax_marg_y.axhline(med_center[1], color='limegreen')
# Legend
joint_axes.ax_joint.legend()
# Add basemap
contextily.add_basemap(
    joint_axes.ax_joint, 
    source=contextily.providers.CartoDB.Positron
)
# Clean axes
joint_axes.ax_joint.set_axis_off()
# Display
plt.show()

#%%
#Dispersion
d = centrography.std_distance(df[['lon','lat']])

#Another helpful visualization is the standard deviational ellipse, or standard ellipse. This is an ellipse drawn from the data that reflects its center, dispersion, and orientation
major, minor, rotation = centrography.ellipse(df[['lon','lat']])

from matplotlib.patches import Ellipse

# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot photograph points
ax.scatter(df['lon'], df['lat'], s=0.75)
ax.scatter(*mean_center, color='red', marker='x', label='Mean Center')
ax.scatter(*med_center, color='limegreen', marker='o', label='Median Center')

# Construct the standard ellipse using matplotlib
ellipse = Ellipse(xy=mean_center, # center the ellipse on our mean center
                  width=major*2, # centrography.ellipse only gives half the axis
                  height=minor*2, 
                  angle = numpy.rad2deg(rotation), # Angles for this are in degrees, not radians
                  facecolor='none', 
                  edgecolor='red', linestyle='--',
                  label='Std. Ellipse')
ax.add_patch(ellipse)

ax.legend()
# Display
# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
plt.show()
#%%
#Extent
coordinates = df[['lon','lat']].values

convex_hull_vertices = centrography.hull(coordinates)

import libpysal
alpha_shape, alpha, circs = libpysal.cg.alpha_shape_auto(coordinates, return_circles=True)

from descartes import PolygonPatch #to plot the alpha shape easily
f,ax = plt.subplots(1,1, figsize=(9,9))

# Plot a green alpha shape
ax.add_patch(
    PolygonPatch(
        alpha_shape, 
        edgecolor='green', 
        facecolor='green', 
        alpha=.2, 
        label = 'Tighest single alpha shape'
    )
)

# Include the points for our prolific user in black
ax.scatter(
    *coordinates.T, color='k', marker='.', label='Source Points'
)

# plot the circles forming the boundary of the alpha shape
for i, circle in enumerate(circs):
    # only label the first circle of its kind
    if i == 0:
        label = 'Bounding Circles'
    else:
        label = None
    ax.add_patch(
        plt.Circle(
            circle, 
            radius=alpha, 
            facecolor='none', 
            edgecolor='r', 
            label=label
        )
    )

# add a blue convex hull
ax.add_patch(
    plt.Polygon(
        convex_hull_vertices, 
        closed=True, 
        edgecolor='blue', 
        facecolor='none', 
        linestyle=':', 
        linewidth=2,
        label='Convex Hull'
    )
)

# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)

plt.legend();

min_rect_vertices = centrography.minimum_bounding_rectangle(coordinates)
(center_x, center_y), radius = centrography.minimum_bounding_circle(coordinates)

from matplotlib.patches import Polygon, Circle, Rectangle
from descartes import PolygonPatch

# Make a purple alpha shape
alpha_shape_patch = PolygonPatch(
    alpha_shape, 
    edgecolor='purple', 
    facecolor='none', 
    linewidth=2,
    label='Alpha Shape'
)

# a blue convex hull
convex_hull_patch = Polygon(
    convex_hull_vertices, 
    closed=True, 
    edgecolor='blue', facecolor='none', 
    linestyle=':', linewidth=2,
    label='Convex Hull'
)

# a green minimum rotated rectangle
"""
# Commented out until functionality is added to pointpats
min_rot_rect_patch = Polygon(
    min_rot_rect, 
    closed=True, 
    edgecolor='green', 
    facecolor='none', 
    linestyle='--', 
    label='Min Rotated Rectangle', 
    linewidth=2
)
"""

# compute the width and height of the 
min_rect_width = min_rect_vertices[2] - min_rect_vertices[0]
min_rect_height = min_rect_vertices[2] - min_rect_vertices[0]

# a goldenrod minimum bounding rectangle
min_rect_patch = Rectangle(
    min_rect_vertices[0:2], 
    width = min_rect_width,
    height = min_rect_height,
    edgecolor='goldenrod', facecolor='none', 
    linestyle='dashed', linewidth=2, 
    label='Min Bounding Rectangle', 
)

# and a red minimum bounding circle
circ_patch = Circle(
    (center_x, center_y), 
    radius=radius,
    edgecolor='red', 
    facecolor='none', 
    linewidth=2,
    label='Min Bounding Circle'
)

f,ax = plt.subplots(1, figsize=(10,10))

ax.add_patch(alpha_shape_patch)
ax.add_patch(convex_hull_patch)
# Commented out until functionality is added to pointpats
#ax.add_patch(min_rot_rect_patch)
ax.add_patch(min_rect_patch)
ax.add_patch(circ_patch)

ax.scatter(df.lon, df.lat, s=.75, color='grey')
#ax.scatter(user.x, user.y, s=100, color='r', marker='x')
ax.legend(ncol=1, loc='center left')

# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
plt.show()

#%%
#Exploting the degree of overall clustering
# Define DBSCAN
from sklearn.cluster import DBSCAN
clusterer = DBSCAN()
# Fit to our data
clusterer.fit(df[["lon", "lat"]])
DBSCAN()

# Print the first 5 elements of `cs`
clusterer.core_sample_indices_[:5]
clusterer.labels_[:5]

lbls = pd.Series(clusterer.labels_, index=df.index)

# Setup figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Subset points that are not part of any cluster (noise)
noise = df.loc[lbls==-1, ['lon', 'lat']]
# Plot noise in grey
ax.scatter(noise['lon'], noise['lat'], c='grey', s=5, linewidth=0)
# Plot all points that are not noise in red
# NOTE how this is done through some fancy indexing, where
#      we take the index of all points (tw) and substract from
#      it the index of those that are noise
ax.scatter(df.loc[df.index.difference(noise.index), 'lon'], \
           df.loc[df.index.difference(noise.index), 'lat'], \
          c='red', linewidth=0)
# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
# Display the figure
plt.show()

# Obtain the number of points 1% of the total represents
minp = numpy.round(df.shape[0] * 0.01)
minp

# Rerun DBSCAN
clusterer = DBSCAN(eps=500, min_samples=minp)
clusterer.fit(df[['lon', 'lat']])
# Turn labels into a Series
lbls = pd.Series(clusterer.labels_, index=df.index)
# Setup figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Subset points that are not part of any cluster (noise)
noise = df.loc[lbls==-1, ['lon', 'lat']]
# Plot noise in grey
ax.scatter(noise['lon'], noise['lat'], c='grey', s=5, linewidth=0)
# Plot all points that are not noise in red
# NOTE how this is done through some fancy indexing, where
#      we take the index of all points (db) and substract from
#      it the index of those that are noise
ax.scatter(
    df.loc[df.index.difference(noise.index), 'lon'],
    df.loc[df.index.difference(noise.index), 'lat'],
    c='red', 
    linewidth=0
)
# Add basemap
contextily.add_basemap(
    ax, 
    source=contextily.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
# Display the figure
plt.show()






