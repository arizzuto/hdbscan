import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
import sklearn.datasets as data
import pickle 

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 20, 'linewidths':0}
#moons, _ = data.make_moons(n_samples=50, noise=0.05)
#blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
#test_data = np.vstack([moons, blobs])

##read in a bunch of fake data
synthdir = '/Users/arizz/python/BAFGKM/clustering/synth_data/'

field1=pickle.load(open(synthdir + "field_100000_240.0_290.0_-0.0_-50.0.pkl","rb"))
f1_dat = np.transpose(field1[0])[0:50000]
f1_sig = np.transpose(field1[1])[0:50000]

group1=pickle.load(open(synthdir+"Group_100_259.989768495_-24.9830185392_2.pkl","rb"))
group2=pickle.load(open(synthdir+"Group_100_260.001473435_-24.9965254345_5.pkl","rb"))
group3=pickle.load(open(synthdir+"Group_100_280.000380772_-44.9997631566_8.pkl","rb"))
group4=pickle.load(open(synthdir+"Group_100_240.002650818_-0.00353780488242_9.pkl","rb"))
group5=pickle.load(open(synthdir+"Group_100_279.984127946_-26.9723489418_0.5.pkl","rb"))
group6=pickle.load(open(synthdir+"Group_100_259.924640641_-39.8720758058_0.5.pkl","rb"))


g1_dat = group1[0][0:20]
g1_sig = np.transpose(group1[1])

g2_dat = group2[0][0:20]
g2_sig = np.transpose(group2[1])
##move it slightly off the first group
g2_dat[:,0] += 10
g2_dat[:,1] += 10

g3_dat = group3[0][0:20]
g3_sig = np.transpose(group3[1])

g4_dat = group4[0][0:20]
g4_sig = np.transpose(group4[1])
g4_dat[:,0] += 10
g4_dat[:,1] -= 12

g5_dat = group5[0][0:20]
g5_sig = np.transpose(group5[1])

g6_dat = group6[0][0:20]
g6_sig = np.transpose(group6[1])


X     = np.concatenate((f1_dat,g1_dat,g2_dat,g3_dat,g4_dat,g5_dat,g6_dat))
sig_X = np.concatenate((f1_sig,g1_sig,g2_sig,g3_sig,g4_sig,g5_sig,g6_sig))
rand_index = np.random.permutation(X.shape[0])
X = X[rand_index]
sig_X = sig_X[rand_index]

#X = f1_dat
#sig_X = f1_sig

##save the inputs somewhere
outfile0 = "outputs/cluster_test2_X.pkl"
outfile0 = "outputs/cluster_test2_sigX.pkl"
pickle.dump(X,open( outfile0, "wb" ))
pickle.dump(sig_X,open( outfile0, "wb" ))


#pdb.set_trace()
#plt.scatter(X.T[0], X.T[1], color='b', **plot_kwds)
#plt.show()

##ss = np.array([[1.0,0],[0.0,1.0]])
ms = np.mean(sig_X**2,axis=0)
S_mat = np.eye(5)*ms
##set the spatial coordinate lengths differently
S_mat[0,0] = 1.0
S_mat[1,1] = 1.0

#pdb.set_trace()
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, V=S_mat, metric='mahalanobis',core_dist_n_jobs=1)
clusterer.fit(X)

##plot a single linkage tree of the clustering
#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#plt.show()

##plot a condensed tree
#clusterer.condensed_tree_.plot()
#plt.show()
pdb.set_trace()
##plot the clusters found by the HDBSCAN run
##palette        = sns.color_palette()
##assign colors by cluster label, and depth/saturation of color by probability
##cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]
uclust         = np.unique(clusterer.labels_)
for i in range(len(uclust)):
    thisclust = np.where(clusterer.labels_ == uclust[i])[0]
    pdb.set_trace()
    plt.scatter(X.T[thisclust,0], X.T[thisclust,1], **plot_kwds)
plt.show()





pdb.set_trace()
print 'here'
