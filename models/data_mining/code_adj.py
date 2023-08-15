import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

con = sqlite3.connect("database.sqlite")
trips_df = pd.read_sql_query("SELECT * from trip", con)

ids = pd.read_sql_query("SELECT DISTINCT(id) from station", con)

reverse = {}
for i in range(ids.shape[0]):
    j = int(ids.iloc[i])
    reverse[j] = i

adjm = np.zeros((len(reverse),len(reverse)))
for i,row in trips_df.iterrows():
    if i%10000 == 0:print(i)
    adjm[reverse[row.start_station_id], reverse[row.end_station_id]] += 1
