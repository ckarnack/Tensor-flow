
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import sparse


# In[2]:


data = pd.read_csv("D:/task3.csv", delimiter=';', encoding='cp1251')


# Построим проекции по всем осям данных

# In[3]:


scatter_matrix(data)


# Признаки "возраст" и "стаж вождения" сильно кореллированы и на прокециях с другими признаками дают сильно похожие матрицы разброса данных. Значит их можно рассматривать как один признак (например, стаж).

# Проанализировав построенную матрицу разброса можно утверждать, что для нашей выборки можно с высокой степенью точности посчитать наиболее удачное количество итоговых кластеров.
#  Однако это, скорее, частный случай, который мы имеем в следствие достаточно хорошо разделимой выборки и небольной степени размерности признаков.
# 
# Рассмотрим более общий случай, когда мы не можем качественно оценить число итоговых кластеров. Тогда лучшим выбором в качестве метода кластеризации представляется DBSCAN,
#  который достаточно хорошо работает когда у нас небольшая выборка, и предположительно, небольшое количество кластеров различной формы.

# Для начала необходимо отмасштабировать данные:

# In[4]:


from sklearn.preprocessing import MaxAbsScaler
keys = [x for x in data.keys()][1:]


# In[5]:


maxabs_scaler = MaxAbsScaler()
scaled_features = maxabs_scaler.fit_transform(data[keys])


# Применим DBSCAN:

# In[6]:


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
dbscan = DBSCAN(eps=0.13, min_samples=5)
dbscan_preds = dbscan.fit_predict(scaled_features)


# In[7]:


np.unique(dbscan_preds)


# In[8]:


dbscan_clusters = []
for lbl in np.unique(dbscan_preds):
    indices = [i for i, x in enumerate(dbscan_preds) if x == lbl]
    dbscan_clusters.append(indices)


# In[9]:


for i in dbscan_clusters:
    print(len(i))


# In[24]:


for cluster in dbscan_clusters:
    print(data.loc[cluster].mean())


# In[11]:


from sklearn.cluster.hierarchical import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=4, linkage='average', affinity='manhattan')
aggl_preds = model.fit_predict(scaled_features)


# In[12]:


clusters_aggl = []
for lbl in np.unique(aggl_preds):
    indices = [i for i, x in enumerate(aggl_preds) if x == lbl]
    clusters_aggl.append(indices)


# In[25]:


for cluster in clusters_aggl:
    print(data.loc[cluster].mean())


# Попробуем алгоритм k-means:

# In[15]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, max_iter=50, random_state=22)


# In[16]:


kmeans_preds = kmeans.fit_predict(scaled_features)


# In[18]:


clusters_kmeans = []
km_labels=np.unique(kmeans_preds)
print(km_labels)
for lbl in km_labels:
    indices = [i for i, x in enumerate(kmeans_preds) if x == lbl]
    clusters_kmeans.append(indices)


# In[27]:


for cluster in clusters_kmeans:
    print(data.loc[cluster].mean())

