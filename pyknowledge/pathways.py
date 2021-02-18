# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd 
import matplotlib.pyplot as plt
import pyknowledge #common
from joblib import Parallel, delayed
import pickle 
import numpy as np


def get_pathway_genes(csv, minimum_entities=0):
    pathways_df = pd.read_csv(csv)
    pathways_df = pathways_df[pathways_df['#Entities found'] > minimum_entities]
    pathways_df['Submitted entities found'] = pathways_df['Submitted entities found'].apply(lambda x: x.split(';'))
    pathways_df = pathways_df.explode('Submitted entities found')
    return pathways_df


def get_pathway_distances(pathways_df, genes_df_scaled, output_file=None):
    ncores = 7
    pair_ixs = pyknowledge.common.get_pair_inxs(genes_df_scaled.shape[0])
    chunks_pair_ixs = list(pyknowledge.common.divide_chunks(pair_ixs,pyknowledge.common.calc_len_chunk(len(pair_ixs),ncores)))

    D_s = {}
    
    for e, i in enumerate(np.unique(pathways_df['Pathway identifier'])):
        print(f'Now generating distances for Pathway {i}...')
        to_include = pathways_df[pathways_df['Pathway identifier'] == i]['Submitted entities found'].values

        helper = lambda distance_func: pd.concat(Parallel(n_jobs=-1)(delayed(pyknowledge.distance.chunk_distance)(np.arctan(genes_df_scaled[to_include]),chunk,distance_func) for chunk in chunks_pair_ixs))
        DL1 = helper(pyknowledge.distance.L1)
        DL2 = helper(pyknowledge.distance.L2)
        DFSIGN = helper(pyknowledge.distance.FSIGN)


        D_s[f'DL1_{i}'] = DL1
        D_s[f'DL2_{i}'] = DL2
        D_s[f'DLSIGN_{i}'] = DFSIGN
        D_s[f'checkpoint'] = i

        if output_file:
            with open(output_file, 'wb') as fp:
                pickle.dump(D_s, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
        return D_s


def luma_lumb_distances(Ds, find_labels, remove_labels):
    
    find = {}
    remove = {}
    
    for key in Ds.keys():

        distances = {}
        for key in Ds.keys():
            distances[key] = pd.DataFrame(list(Ds[key]['distance']),columns=[key],index=Ds[key].index)

        for key in distances.keys():
            distances[key] = pyknowledge.distance.remove_self_ref(distances[key]).dropna()

        for key in distances.keys():
            distances[key] = pyknowledge.common.add_labels(df,distances[key])

        for key in distances.keys():
            t = distances[key]

            lumb = pd.concat([lumb,t[t['label1_label2'] == 'LumB - LumB'][key].reset_index()[key]], axis=1)
            not_lumb = pd.concat([not_lumb,t[t['label1_label2'] != 'LumB - LumB'][key].reset_index()[key]], axis=1)
            luma = pd.concat([luma,t[t['label1_label2'] == 'LumA - LumA'][key].reset_index()[key]], axis=1)
            not_luma = pd.concat([not_luma,t[t['label1_label2'] != 'LumA - LumA'][key].reset_index()[key]], axis=1)

    lumb.to_csv('lumb.csv')
    not_lumb.to_csv('not_lumb.csv')
    luma.to_csv('luma.csv')
    not_luma.to_csv('not_luma.csv')


