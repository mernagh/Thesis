#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:41:17 2022

@author: thomasmernagh
"""
import pandas as pd
import geopandas
from shapely.geometry import Point

df = pd.read_csv("provincie-utrecht_2021_04_12_2021_04_19.csv")
print (df)
df.head()

# combine lat and lon column to a shapely Point() object
df['geometry'] = df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)


df = geopandas.GeoDataFrame(df, geometry='geometry')
df.to_file('UtrechtW15.shp', driver='ESRI Shapefile')
