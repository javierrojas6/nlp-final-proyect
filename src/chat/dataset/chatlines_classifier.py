import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import torch

class ChatLinesClassifier:
  components_n = None
  class_number = None

  def __init__(self, components_n, class_number, device='cpu', svd_solver='auto', model_name='all-MiniLM-L6-v2'):
    self.device =device
    self.class_number =class_number
    self.components_n = components_n
    self.model_name = model_name
    self.svd_solver = svd_solver
    
  def fit(self, df:pd.DataFrame, random_state=None, verbose=False):
    model = SentenceTransformer(self.model_name)
    
    # generate embeddings
    embeddings = model.encode(df['text'], convert_to_tensor=True)
    
    ìtems = embeddings.to(self.device).detach().numpy()
    
    pca = PCA(n_components=self.components_n, svd_solver=self.svd_solver)
    pca_items = pca.fit_transform(ìtems)
    
    kmeans = KMeans(n_clusters=self.class_number, init='k-means++', random_state=random_state)
    kmeans.fit(pca_items)
    
    df['line_intent'] = kmeans.labels_

    if verbose:
      print('explained_variance_ratio',pca.explained_variance_ratio_)
      print('singular_values',pca.singular_values_)
      
    return df, ìtems, pca_items, kmeans