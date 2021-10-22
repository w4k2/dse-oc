import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
import strlearn as sl
from ensembles import DeterministicSamplingEnsemble, DSO, DSE_O, OCIS, MultiSamplingRandomSubspace, DeterministicSamplingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from scipy.ndimage import gaussian_filter1d

base_estimator = SVC(probability=True)
base_estimator = KNeighborsClassifier()
base_estimator = GaussianNB()

clfs = {
    "DSO": DSO(),
    "DSC": DeterministicSamplingClassifier(),
    # "MSRS-O": MultiSamplingRandomSubspace(base_classifier=OCIS()),
    # "DSE_O": DSE_O(OCIS()),
    # "DSE-O": DeterministicSamplingEnsemble(OCIS()),
    # "DSE": DeterministicSamplingEnsemble(base_estimator),
    # "OCEIS-OLD": OCEIS(),
    "L++CDS": sl.ensembles.LearnppCDS(base_estimator),
    # "KMC": sl.ensembles.KMC(base_estimator),
    # "L++NIE": sl.ensembles.LearnppNIE(base_estimator),
    # "REA": sl.ensembles.REA(base_estimator),
    # "OUSE": sl.ensembles.OUSE(base_estimator),
    # "MLPC": MLPClassifier(hidden_layer_sizes=(10))
}


stream = sl.streams.ARFFParser("streams/moa_1d/sudden/stream_moa_rbf_s_drift_100k_f10_0.05b_0.05n_rs111.arff", chunk_size=500, n_chunks=50)
# stream = sl.streams.StreamGenerator(n_chunks=50, chunk_size=500, n_features=10, n_informative=8, n_redundant=2, y_flip=0.01, n_drifts=1, weights=[0.9, 0.1], random_state=1111)


metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]
# metrics = [sl.metrics.binary_confusion_matrix]

evaluator = sl.evaluators.TestThenTrain(metrics, verbose=True)
evaluator.process(stream, clfs.values())
# print(evaluator.confusion_matrix)

# fig, ax = plt.subplots(len(metrics), 1, figsize=(15, 5))
#
# labels = list(clfs.keys())
#
# for m, metric in enumerate(metrics):
#     ax[m].set_title(metric.__name__)
#     ax[m].set_ylim(0, 1)
#     for i, clf in enumerate(clfs):
#         ax[m].plot(evaluator.scores[i, :, m], label=labels[i])
#     ax[m].legend()
#
# plt.show()

labels = list(clfs.keys())
linestyles = ['-','-','-','-','-','-','-','-','-','-','--','--','--','--','--','--','--','--','--','--']
for m, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.suptitle(metric.__name__)
    # plt.ylim(0, 1)
    for i, clf in enumerate(clfs):
        plt.plot(gaussian_filter1d(evaluator.scores[i, :, m], 3), label=labels[i], linestyle=linestyles[i])
        # plt.plot(evaluator.scores[i, :, m], label=labels[i], linestyle=linestyles[i])
    plt.legend()
    plt.savefig("results/sl_%s.png" % metric.__name__)
    # plt.show()
