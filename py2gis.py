#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:51:51 2022

@author: thomasmernagh
"""

#Packages
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import IsolationForest
from shapely.geometry import Point, LineString, shape

#Import raw datafile
df = pd.read_csv("provincie-utrecht_2021_04_12_2021_04_19.csv")
#Check and isolate outliers, exclude outliers
df["outlier"] = IsolationForest().fit_predict(df[["lat", "lon"]])
cleaned_df = df[df["outlier"] != -1]
cleaned_df.to_csv("points_without_outliers.csv")

#Rename X,Y for lat,lon
df = pd.read_csv("points_without_outliers.csv")
print (df)
df.head()
df['X'] = df['lat']
df['Y'] = df['lon']

#combine lat,long to be geometries
# combine lat and lon column to a shapely Point() object
df['geometry'] = df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
#Create points shapefiles
df = gpd.GeoDataFrame(df, geometry='geometry')
df.to_file('UtrechtW15.shp', driver='ESRI Shapefile')

#Import and read outliers file
df = pd.read_csv("points_without_outliers.csv", sep='\s*,\s*')

#zip the coordinates into a point object and convert to a GeoData Frame
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

geo_df2 = geo_df.groupby(['sensor'])['geometry'].apply(lambda x:LineString(x.tolist()))
geo_df2 = gpd.GeoDataFrame(geo_df2, geometry='geometry')
#Output lines shape file
geo_df2.to_file('UtrechtW15_lines.shp', driver='ESRI Shapefile')