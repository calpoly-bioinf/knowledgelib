import pandas as pd
import numpy as np

# a row is type pd.Series

# raw distance between rows
def L0(row1,row2,ixs):
    s = list(row1.values[ixs] - row2.values[ixs])
    return s

def L1(row1,row2):
    s = ((row1 - row2).abs()).mean()
    return s

def L2(row1,row2):
    s = np.sqrt(((row1 - row2)**2).sum())
    return s

# Fraction Matching Signs
def FSIGN(row1,row2):
    s = ((row1 > 0) == (row2 > 0)).sum()/len(row1)
    return s


def L3(row1,row2):
    s = ((row1 - row2)).mean()
    return s


def L4(row1,row2):
    s = 1 - spatial.distance.cosine(row1, row2)
    return s


def chunk_distance(df,chunk,distance_func):
    distances = []
    index1 = []
    index2 = []
    for ix1,ix2 in chunk:
        index1.append(df.index[ix1])
        index2.append(df.index[ix2])
        distances.append(distance_func(df.iloc[ix1],df.iloc[ix2]))
    return pd.DataFrame({"distance":distances,"index1":index1,"index2":index2}).set_index(["index1","index2"])


def remove_self_ref(distance1):
    new_indices = []
    for ix1,ix2 in distance1.index:
        if ix1 < ix2:
            new_indices.append((ix1,ix2))
    return distance1.loc[new_indices]
