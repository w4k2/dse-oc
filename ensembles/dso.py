from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.neural_network import MLPClassifier

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.svm import OneClassSVM
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


class DSO(ClassifierMixin, BaseEnsemble):

    def __init__(self, n_estimators=10, undersampling=InstanceHardnessThreshold(), cluster_method=MiniBatchKMeans, cluster_metric=silhouette_score):
        self.base_estimator = OneClassSVM(nu=0.01)
        self.n_estimators = n_estimators
        self.undersampling = undersampling
        self.cluster_method = cluster_method
        self.cluster_metric = cluster_metric

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.metrics_array = []
            self.stored_X = []
            self.stored_y = []
            self.drift_detector = None

            self.ensemble_maj = []
            self.ensemble_min = []
            self.iter_maj = []
            self.iter_min = []

        # ________________________________________
        # Check if is more than one class

        if len(np.unique(y)) == 1:
            raise ValueError("Only one class in data chunk.")

        # ________________________________________
        # Check feature consistency

        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # ________________________________________
        # Check classes

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # ________________________________________
        # Find minority and majority names

        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        # ________________________________________
        # Drift detection

        if(self.drift_detector is not None):
            dd_pred = self.drift_detector.predict(X)
            score = geometric_mean_score(dd_pred, y)
            if score / np.mean(self.metrics_array) < 0.7:
                print("Det")
                self.drift_detector = None
                self.metrics_array = []
                self.ensemble_ = []
                self.stored_X = []
                self.stored_y = []

                # self.stored_X = [self.stored_X[-1]]
                # print(self.stored_X)
                # self.stored_y = [self.stored_y[-1]]

                self.ensemble_maj = []
                self.ensemble_min = []
                self.iter_maj = []
                self.iter_min = []
                # self.fit(X, y)
            else:
                self.metrics_array.append(score)

        # ________________________________________
        # Get stored data

        new_X, new_y = [], []

        for tmp_X, tmp_y in zip(self.stored_X, self.stored_y):
            new_X.extend(tmp_X)
            new_y.extend(tmp_y)

        new_X.extend(X)
        new_y.extend(y)

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # ________________________________________
        # Undersample and store new data

        und_X, und_y = self.undersampling.fit_resample(X, y)
        self.stored_X.append(und_X)
        self.stored_y.append(und_y)

        # ________________________________________
        # Split data

        minority, majority = self.minority_majority_split(new_X, new_y, self.minority_name, self.majority_name)

        # ________________________________________
        # Train minority classifier

        samples, n_of_clust = self._best_number_of_clusters(minority, 5)
        for i in range(n_of_clust):
            self.ensemble_min.append(clone(self.base_estimator).fit(samples[i]))
            self.iter_min.append(self.n_estimators)

        # ________________________________________
        # Train majority classifiers

        samples, n_of_clust = self._best_number_of_clusters(majority, 5)
        for i in range(n_of_clust):
            self.ensemble_maj.append(clone(self.base_estimator).fit(samples[i]))
            self.iter_maj.append(self.n_estimators)

        # ________________________________________
        # Prune stored data

        if len(self.stored_X) >= self.n_estimators:
            del self.stored_X[0]
            del self.stored_y[0]

        # ________________________________________
        # Prune minority

        to_delete = []
        for i, w in enumerate(self.iter_min):
            if w <= 0:
                to_delete.append(i)
            self.iter_min[i] -= 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_min[i]
            del self.ensemble_min[i]

        # ________________________________________
        # Prune majority

        to_delete = []
        for i, w in enumerate(self.iter_maj):
            if w <= 0:
                to_delete.append(i)
            self.iter_maj[i] -= 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_maj[i]
            del self.ensemble_maj[i]

        # ________________________________________
        # Train drift detector

        if self.drift_detector is None:
            self.drift_detector = MLPClassifier((10))
        self.drift_detector.partial_fit(new_X, new_y, np.unique(new_y))

        return self

    def _best_number_of_clusters(self, data, kmax=10):

        sil_values = []
        clusters = []

        for k in range(2, kmax+1):
            try:
                cluster_model = self.cluster_method(n_clusters=k)
                labels = cluster_model.fit_predict(data)
                clusters.append(labels)
                sil_values.append(self.cluster_metric(data, labels))
            except Exception:
                # print("EXC", k)
                break

        best_number = np.argmax(np.array(sil_values))
        n_of_clust = best_number+2
        samples = [[] for i in range(n_of_clust)]

        for i, x in enumerate(clusters[best_number]):
            samples[x].append(data[i].tolist())

        return samples, n_of_clust

    def predict(self, X):
        maj = np.argmax(self.predict_proba(X), axis=1)

        return maj

    def predict_proba(self, X):
        probas_min = np.max([clf.decision_function(X) for clf in self.ensemble_min], axis=0)
        probas_maj = np.max([clf.decision_function(X) for clf in self.ensemble_maj], axis=0)
        probas_ = np.stack((probas_maj, probas_min), axis=1)

        return probas_

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority : array-like, shape = [n_samples, n_features]
            Minority class samples.
        majority : array-like, shape = [n_samples, n_features]
            Majority class samples.
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns the name of minority and majority class

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority_name : object
            Name of minority class.
        majority_name : object
            Name of majority class.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name
