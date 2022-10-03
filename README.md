# PCAPxDEG
A Python Clustering Analysis Protocol Of Genes Expression Data Sets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler

#the variable containing the path to the input gene expression data set to analyse
data = "./data.csv"

# Gene expression data set is red using NumPy genfromtxt, setting "," as delimiter
# and skipping first column and row since containing the headers info
gene_exp_val = np.genfromtxt(data, delimiter=",", usecols=range(1, 1001), skip_header=1).transpose()

#definition of the MaxAbsScaler estimator object
scaler = MaxAbsScaler()
#run the estimator and store the result into the variable transformed_data
transformed_data = scaler.fit_transform(gene_exp_val)

#definition and execution of the pca object 
pca_data = PCA(n_components=2, random_state=78).fit_transform(transformed_data)

#definition and execution of the kmeans object 
dbscan = DBSCAN(eps=0.5, min_samples=13, metric='euclidean', algorithm='auto').fit(pca_data)

#code to save the detected cluster as eps image
dataframe = pd.DataFrame(pca_data, columns=["X", "Y"])
dataframe["clusters"] = dbscan.labels_

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 6)) #
sns.scatterplot(
"X", "Y",
s=50,
data=dataframe,
hue="clusters",
palette="Set2")
lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.savefig(fname='./clusters.eps',
            dpi=300,
            format='eps',
            bbox_extra_artists=(lg,),
            bbox_inches='tight')
plt.tight_layout()
plt.show()
