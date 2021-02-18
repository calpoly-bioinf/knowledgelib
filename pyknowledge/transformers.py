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

from sklearn.preprocessing import StandardScaler
import pandas as pd


def standard_scale(genes_df):
    
    scaler = StandardScaler()
    scaler.fit(genes_df)

    genes_df_scaled = pd.DataFrame(scaler.transform(genes_df),index=genes_df.index,columns=genes_df.columns).fillna(0)
    return genes_df_scaled


def arctan_scale(genes_df):
    
    scaler = StandardScaler()
    scaler.fit(genes_df)

    genes_df_scaled = pd.DataFrame(scaler.transform(genes_df),index=genes_df.index,columns=genes_df.columns).fillna(0)
    return np.arctan(genes_df_scaled)


