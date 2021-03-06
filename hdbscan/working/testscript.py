import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
import sklearn.datasets as data

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])

##plot the test data if you like
#plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
#plt.show()

##the mahalanobis distance covariance matrix (not inverted yet)
ss = np.array([[1.0,0],[0.0,1.0]])
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, V=ss, metric='mahalanobis')
clusterer.fit(test_data)

##plot a single linkage tree of the clustering
#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#plt.show()

##plot a condensed tree
#clusterer.condensed_tree_.plot()
#plt.show()

##plot the clusters found by the HDBSCAN run
palette        = sns.color_palette()
##assign colors by cluster label, and depth/saturation of color by probability
cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
plt.show()





pdb.set_trace()
print 'here'
