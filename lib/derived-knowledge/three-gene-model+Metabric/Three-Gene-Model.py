#!/usr/bin/env python
import sys
seed = int(sys.argv[1])
# coding: utf-8

# Details:
# 
# https://academic.oup.com/jnci/article/104/4/311/979947
# 
# > Single sample predictors (SSPs) and Subtype classification models (SCMs) are gene expression–based classifiers used to identify the four primary molecular subtypes of breast cancer (basal-like, HER2-enriched, luminal A, and luminal B). SSPs use hierarchical clustering, followed by nearest centroid classification, based on large sets of tumor-intrinsic genes. SCMs use a mixture of Gaussian distributions based on sets of genes with expression specifically correlated with three key breast cancer genes (estrogen receptor [ER], HER2, and aurora kinase A [AURKA]). The aim of this study was to compare the robustness, classification concordance, and prognostic value of these classifiers with those of a simplified three-gene SCM in a large compendium of microarray datasets.
# 
# AURKA
# 
# ER is ESR1 (Source: https://www.genecards.org/cgi-bin/carddisp.pl?gene=ESR1)
# 
# HER2 is ERBB2 (Source: https://www.genecards.org/cgi-bin/carddisp.pl?gene=ERBB2)

# In[99]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# ### Customize this for each notebook

# In[100]:


OUTPUT_DIR='Three-Gene-Model/'
OPTIONS={'seed':seed}
PREFIX="_".join([f"{key}={OPTIONS[key]}" for key in OPTIONS.keys()])
RESULTS={}
PREFIX


# In[101]:


from pathlib import Path
home = str(Path.home())


# In[102]:


KNOWLEDGE_LIB=f'{home}/knowledgelib'


# In[103]:


from IPython.display import display, Markdown, Latex
import sys
sys.path.insert(0,f'{KNOWLEDGE_LIB}')
import pyknowledge
import pandas as pd
import scipy.io
import pandas as pd
import numpy as np
import joblib


# ## Load the input data

# In[104]:


## Customize this load to read in the data and format it with the correct columns
def load_data_all(seed):
    mat = scipy.io.loadmat("/disk/metabric/BRCA1View20000.mat")
    #gene_labels = open("/disk/metabric/gene_labels.txt").read().split("\n")
    gene_labels = [g[0] for g in mat['gene'][0]]
    df = pd.DataFrame(mat['data'].transpose(), columns=gene_labels)
    [n_dim, n_sample] = df.shape
    for i in range(n_dim):
        m1 = min(df.iloc[:,i])
        m2 = max(df.iloc[:,i])
        df.iloc[:,i] =(df.iloc[:,i] - m1)/(m2 - m1)
    df['target'] = mat['targets']
    df['Subtype'] = df.target.map({1:'Basal',2:'HER2+',3:'LumA',4:'LumB',5:'Normal Like',6:'Normal'})
    df['color'] = df.target.map({1:'red',2:'green',3:'purple',4:'cyan',5:'blue',6:'green'})
    df['graph_color'] = df.target.map({1:'#FFFFFF',2:'#F5F5F5',3:'#FFFAFA',4:'#FFFFF0',5:'#FFFAF0',6:'#F5FFFA'})
    index = joblib.load(f'/disk/metabric/index_{seed}.joblib.z')    
    df = df.iloc[index,:]
    df = df.set_index(np.arange(len(df)))
    

    return df

def load_data_train(seed):
    mat = scipy.io.loadmat("/disk/metabric/BRCA1View20000.mat")
    df = joblib.load(f'/disk/metabric/data_train_{seed}.joblib.z')
    gene_labels = [g[0] for g in mat['gene'][0]]
    targets_train = joblib.load(f'/disk/metabric/targets_train_{seed}.joblib.z')
    df.columns = gene_labels
    #df = pd.DataFrame(data_train, columns=gene_labels)
    targets_train = joblib.load(f'/disk/metabric/targets_train_{seed}.joblib.z')
    df['target'] = targets_train.apply(lambda x: x.idxmax(),axis=1)+1
    df['Subtype'] = df.target.map({1:'Basal',2:'HER2+',3:'LumA',4:'LumB',5:'Normal Like',6:'Normal'})
    df['color'] = df.target.map({1:'red',2:'green',3:'purple',4:'cyan',5:'blue',6:'green'})
    df['graph_color'] = df.target.map({1:'#FFFFFF',2:'#F5F5F5',3:'#FFFAFA',4:'#FFFFF0',5:'#FFFAF0',6:'#F5FFFA'})
    return df


# In[105]:


df_all = load_data_all(OPTIONS['seed'])


# In[106]:


df_all.Subtype.value_counts() # basal-like, HER2-enriched, luminal A, and luminal B


# In[107]:


df = load_data_train(OPTIONS['seed'])


# In[108]:


df.tail()


# In[109]:


df_val = df_all.iloc[len(df):(len(df)+170),:]
df_val


# ## Knowledge

# #### Genes

# In[110]:


knowledge_genes = ["ERBB2","ESR1","AURKA"]


# In[111]:


genes_df = pyknowledge.common.subset_columns(df,knowledge_genes)
genes_df.head()


# In[112]:


genes_df_all = pyknowledge.common.subset_columns(df_all,knowledge_genes+["Subtype"])
genes_df_all.head()


# In[113]:


genes_df_all.to_csv('genes_df_all.csv')


# In[114]:


genes_df_all['Subtype'].value_counts()


# ## EDA

# ### Each gene individually

# In[115]:


source = genes_df.join(df[['Subtype','target']]).melt(id_vars=['Subtype','target'])
source.columns = ["Subtype","target","Gene","Value"]
counts = source.groupby('Subtype')['target'].count().to_frame()
counts.columns = ['Count']
source = source.set_index('Subtype').join(counts).reset_index()
# use the url approach to keep notebook smaller
#url = f'{OUTPUT_DIR}/{PREFIX}_three_gene_model_to_graph_fig1.json'
#pyknowledge.plot.prepare_url(source,url)
# TODO: Figure out what happened to this graph
#pyknowledge.plot.histogram_1(url,'Value','Subtype','Gene',file=f'{OUTPUT_DIR}/{PREFIX}_three_gene_model_to_graph_fig1.png')


# ## Pipeline
# 
# I want this to look like a sklearn pipeline optimization where we can do a grid search. What is our objective function?

# In[116]:


transformer = pyknowledge.transformers.ScaleTransformer()
methods = list(transformer.distributions.set_index('method').index)
methods


# In[117]:


import copy
import itertools
import json
from sklearn.pipeline import Pipeline

# Here you define your pipeline
pipe_factory = [['scaler',  lambda params: pyknowledge.transformers.ScaleTransformer(**params)],
                ['feature', lambda params: pyknowledge.transformers.FeatureSelector(**params)],
                ['knn', lambda params: pyknowledge.transformers.KNNTransformer(**params)]]
                #['distribution', lambda params: pyknowledge.transformers.DistributionTransformer(**params)]]

# Here you specify what arguments you want to search across
param_grid = {
    'scaler__method': ['Unscaled data',
                       'Data after standard scaling',
                       'Data after min-max scaling',
                       'Data after robust scaling'
                      ],
    'feature__columns': [knowledge_genes]
}

def grid_search(X,sub_types,pipe_factory,param_grid,sub_type_subset=None):
    test_dfs = []
    
    grouped = {}
    for key in param_grid.keys():
        key2,key3 = key.split("__")
        if key2 not in grouped:
            grouped[key2] = {}
        grouped[key2][key3] = param_grid[key]
    n = []

    for k in grouped.keys():
        n1 = []
        for k2 in grouped[k].keys():
            for v in grouped[k][k2]:
                n1.append((k,k2,v))
        n.append(n1)
    

    param_options = list(itertools.product(*n))
    
    if sub_type_subset is not None:
        mask = sub_types.isin(sub_type_subset)
        X = X.loc[mask]
        sub_types = sub_types[mask]

    results = None

    #print(param_options[0:3], '\n\n\n')
    for param_options1 in param_options:
        
        steps = []
        step_params = {}
        for step_name,func in pipe_factory:
            #print(step_name, func)
            cloned_params = {}
            for k1,k2,value in param_options1:
                #print(k1, k2, value)
                if k1 == step_name:
                    cloned_params[k2] = value
                    step_params[f"{step_name}__{k2}"] = value
            steps.append((step_name,func(cloned_params)))

        index_cols = list(step_params.keys())
        results1 = pd.DataFrame([],columns=['obj1']+index_cols)
        for k in index_cols:
            v = step_params[k]
            if type(v) == list:
                v = tuple(v)
            results1.loc[0,k] = v

        pipe = Pipeline(steps)
        pipe.fit(X,sub_types)
        distances, indices, labels = pipe.transform(X)
        
        sum_obj = 0
        c = 0
        for i,sub_type in enumerate(sub_types):
            count = np.sum(labels[i,:] == sub_type)
            sum_obj += count
        obj1 = sum_obj/len(sub_types)/labels.shape[1]

        results1.loc[0,'obj1'] = obj1
        
        results1.set_index(index_cols,inplace=True)
        
        if results is None:
            results = results1
        else:
            results = results.append(results1)
            
    return results.infer_objects()


# In[118]:


genes_df_sample = genes_df.copy()
sub_type_subset=['Normal like', 'HER2+', 'LumB', 'LumA']
results = grid_search(genes_df_sample,df.loc[genes_df_sample.index]['Subtype'],pipe_factory,param_grid,sub_type_subset=sub_type_subset)


# In[119]:


results


# In[120]:


for_latex = results.reset_index()
for_latex.columns = ['Scaling method','Genes','Leave-one-out CV']
print(for_latex.to_latex())


# In[121]:


RESULTS['grid_search_results'] = results


# ## Get the best parameters

# In[122]:


best_params = {}
best = results['obj1'].idxmax()
for i,c in enumerate(results.index.names):
    best_params[c] = best[i]
best_params


# In[123]:


RESULTS['best_params'] = best_params


# In[124]:


steps = []
for name,func in pipe_factory:
    params = {}
    for key in best_params.keys():
        fields = key.split("__")
        if f"{name}__" in key:
            if type(best_params[key]) == tuple:
                best_params[key] = list(best_params[key])
            params[fields[1]] = best_params[key]
    steps.append((name,func(params)))

X = genes_df_sample.copy()
sub_types = df.loc[genes_df_sample.index]['Subtype'].copy()
if sub_type_subset is not None:
    mask = sub_types.isin(sub_type_subset)
    X = X.loc[mask]
    sub_types = sub_types[mask]
    df_mask = df.loc[genes_df_sample.index].loc[mask]
    
pipeline = Pipeline(steps)
pipeline.fit(X,sub_types)


# In[125]:


X.shape


# In[126]:


RESULTS['pipeline'] = None # TODO: fix this so it pickles


# In[127]:


distances, indices, labels = pipeline.transform(X)


# In[128]:


labels


# In[129]:


labels


# ### Visualize our graph

# In[130]:


from IPython.display import Image

import networkx as nx

A = pd.DataFrame(index=df_all.index,columns=df_all.index)

G = nx.Graph()
for ix in df.index:
    c = df.loc[ix,'graph_color']
    G.add_node(ix,color='black',style='filled',fillcolor=c)

for i in range(len(indices)):
    iix = df_mask.index[i] # this is wonky one
    for j,ix in enumerate(indices[i,:]):
        jix = df_mask.index[indices[i,j]]
        if labels[i,j] == sub_types.iloc[i]:
            G.add_edge(iix,jix)
            A.loc[iix,jix] = 1
            A.loc[jix,iix] = 1


# In[131]:


RESULTS['A'] = A


# In[132]:


len(df)


# In[133]:


len(df_mask)


# In[134]:


A.to_csv(f"{OUTPUT_DIR}{PREFIX}_graph.csv",)


# In[135]:


f"{OUTPUT_DIR}{PREFIX}_graph.csv"


# In[136]:


def save(A,file="graph.png"):
    g = A.draw(format=file.split(".")[-1], prog='dot')
    open(file,"wb").write(g)
    return Image(g)

pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
A = nx.nx_agraph.to_agraph(G)
A.graph_attr["rankdir"] = "LR"
# draw it in the notebook
save(A,file=f"{OUTPUT_DIR}{PREFIX}_graph.png")


# In[137]:


#get_ipython().system('mkdir {OUTPUT_DIR}{PREFIX}_graphs')


# In[138]:


graphs = list(G.subgraph(c).copy() for c in nx.connected_components(G))

for i,graph in enumerate(graphs):
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
    A = nx.nx_agraph.to_agraph(graph)
    A.graph_attr["rankdir"] = "LR"
    # draw it in the notebook
    save(A,file=f"{OUTPUT_DIR}{PREFIX}_graphs/graph_{i}.png")


# In[139]:


import joblib
joblib.dump(RESULTS,f'{OUTPUT_DIR}{PREFIX}_RESULTS.joblib.z')


# ### Validtaion

# In[140]:


df_val.columns


# In[141]:


X = pyknowledge.common.subset_columns(df_val,knowledge_genes)


# In[142]:


X


# In[143]:


distances, indices, labels = pipeline.transform(X)


# In[144]:


A = pd.DataFrame(index=df_all.index,columns=df_all.index)

for i in range(len(indices)):
    iix = df_val.index[i]
    for j,ix in enumerate(indices[i,:]):
        jix = df_mask.index[indices[i,j]]
        A.loc[iix,jix] = 1
        A.loc[jix,iix] = 1


# In[145]:


A.shape


# In[146]:


A.stack()


# In[147]:


A.to_csv(f"{OUTPUT_DIR}{PREFIX}_graph_val.csv",)


# In[ ]:




