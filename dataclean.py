#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:08:14 2022

@author: thomasmernagh
"""

#Packages
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import IsolationForest
from shapely.geometry import Point, LineString 

#Import raw datafile
df41 = pd.read_csv("Week41.csv")
#Check and isolate outliers, exclude outliers
df41["outlier"] = IsolationForest().fit_predict(df41[["lat", "lon"]])
cleaned_df41 = df41[df41["outlier"] != -1]
cleaned_df41.to_csv("points_without_outliers41.csv")



#Rename X,Y for lat,lon
df41 = pd.read_csv("points_without_outliers41.csv")
print (df41)
df41.head()
df41['X'] = df41['lat']
df41['Y'] = df41['lon']

#combine lat,long to be geometries
# combine lat and lon column to a shapely Point() object
df41['geometry'] = df41.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
#Create points shapefiles
df41 = gpd.GeoDataFrame(df41, geometry='geometry')
df41.to_file('UtrechtW41.shp', driver='ESRI Shapefile')

for sensor in df41:
    #print(df41[sensor])
    #zip the coordinates into a point object and convert to a GeoData Frame
    geometry = [Point(xy) for xy in zip(df41.lon, df41.lat)]
    
geo_df = gpd.GeoDataFrame(df41, geometry=geometry)

geo_df41 = geo_df.groupby(['trip_sequence'])['geometry'].apply(lambda x: LineString(x.tolist()))

geo_df41 = gpd.GeoDataFrame(geo_df41, geometry='geometry')
 #Output lines shape file
geo_df41.to_file('UtrechtW41_lines.shp', driver='ESRI Shapefile')
    
print(geo_df41)