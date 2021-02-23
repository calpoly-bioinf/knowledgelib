from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler
import pandas as pd

from joblib import Parallel, delayed

from . import common
from . import distance

class PassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit( self, X, y = None  ):
        return self
    
    def transform(self, X, y = None ):
        return X

class ScaleTransformer( BaseEstimator, TransformerMixin ):
    distributions = pd.DataFrame([
            ('Unscaled data', lambda X: PassTransformer().fit(X)),
            ('Data after standard scaling',lambda X: StandardScaler().fit(X)),
            ('Data after min-max scaling',lambda X: MinMaxScaler().fit(X)),
            ('Data after max-abs scaling',lambda X: MaxAbsScaler().fit(X)),
            ('Data after robust scaling',lambda X: RobustScaler(quantile_range=(25, 75)).fit(X)),
            ('Data after power transformation (Yeo-Johnson)',lambda X: PowerTransformer(method='yeo-johnson').fit(X)),
            ('Data after power transformation (Box-Cox)',lambda X: PowerTransformer(method='box-cox').fit(X)),
            ('Data after quantile transformation (uniform pdf)',lambda X: QuantileTransformer(output_distribution='uniform').fit(X)),
            ('Data after quantile transformation (gaussian pdf)',lambda X: QuantileTransformer(output_distribution='normal').fit(X)),
            ('Data after sample-wise L2 normalizing',lambda X: Normalizer().fit(X))],columns=['method','scaler'])
    
    def __init__(self,method='Unscaled data'):
        self.method = method
        self.distribution_func = self.distributions.set_index('method').loc[method,'scaler']
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        self.transformer = self.distribution_func(X)
        return self
    
    def transform(self, X, y = None ):
        return self.transformer.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
class DistributionTransformer(BaseEstimator, TransformerMixin):
    distance_funcs = {
        'L1':distance.L1,
        'L2':distance.L2,
        'FSIGN':distance.FSIGN
    }
    def __init__(self,gene_cols=[],ncores=6,distance='L1'):
        self.ncores = ncores
        self.gene_cols = gene_cols
        self.distance = distance
        self.distance_func = self.distance_funcs[distance]
    
    def fit( self, X, y = None  ):
        return self
    
    def transform(self, X, y = None ):
        pair_ixs = common.get_pair_inxs(X.shape[0])
        chunks_pair_ixs = list(common.divide_chunks(pair_ixs,common.calc_len_chunk(len(pair_ixs),self.ncores)))
        return pd.concat(
            Parallel(n_jobs=self.ncores)(delayed(distance.chunk_distance)(X,chunk,self.distance_func) for chunk in chunks_pair_ixs))
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)

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


