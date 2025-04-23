import numpy as np
from sklearn.cluster import MiniBatchKMeans

class MultiClassLLC:
    """
    Locality-constrained Linear Coding (LLC) for multi-class features.

    Parameters
    ----------
    knn : int, default=8
        Number of nearest atoms for LLC coding.
    beta : float, default=1e-3
        Regularization coefficient to stabilize coding.
    per_class_words : int, default=16
        Number of visual words (atoms) per class.
    kmeans_batch_size : int, default=100
        MiniBatchKMeans batch size during vocabulary training.
    kmeans_max_iter : int, default=100
        Maximum number of iterations for MiniBatchKMeans.
    kmeans_n_init : int, default=3
        Number of different initializations to run for MiniBatchKMeans.
    """
    def __init__(
        self,
        knn=8,
        beta=1e-3,
        per_class_words=16,
        kmeans_batch_size=512,
        kmeans_max_iter=1000,
        kmeans_n_init=4,
    ):
        self.knn = knn
        self.beta = beta
        self.per_class_words = per_class_words
        self.kmeans_batch_size = kmeans_batch_size
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_n_init = kmeans_n_init
        self.dictionaries = {}  # stores per-class codebooks

    def fit(self, feature_list, class_list):
        """
        Train one MiniBatchKMeans per class to learn visual vocabularies.

        Parameters
        ----------
        feature_list : list of np.ndarray, each shape (Ni, D)
            List of feature arrays for each class.
        class_list : list
            Corresponding class labels for each feature array.

        Returns
        -------
        self : object
            Fitted instance.
        """
        from collections import defaultdict
        feats_by_class = defaultdict(list)
        # Group features by class
        for feats, cls in zip(feature_list, class_list):
            feats_by_class[cls].append(feats)

        # Fit KMeans for each class
        for cls, flist in feats_by_class.items():
            all_feats = np.vstack(flist)
            kmeans = MiniBatchKMeans(
                n_clusters=self.per_class_words,
                batch_size=self.kmeans_batch_size,
                max_iter=self.kmeans_max_iter,
                n_init=self.kmeans_n_init,
                random_state=191,
            )
            kmeans.fit(all_feats)
            self.dictionaries[cls] = kmeans.cluster_centers_
        return self

    def transform(self, feature_list, class_list):
        """
        Encode each feature matrix using its class-specific dictionary.

        Parameters
        ----------
        feature_list : list of np.ndarray, each shape (Ni, D)
        class_list : list
            Corresponding class labels.

        Returns
        -------
        codes_list : list of np.ndarray
            List of code matrices, each shape (Ni, per_class_words).
        """
        codes_list = []
        for feats, cls in zip(feature_list, class_list):
            code = self._llc_encode(feats, self.dictionaries[cls])
            codes_list.append(code)
        return codes_list

    def _llc_encode(self, features, atoms):
        """
        Compute LLC codes for input features given visual atoms.

        Parameters
        ----------
        features : np.ndarray, shape (n_samples, D)
        atoms : np.ndarray, shape (K, D)

        Returns
        -------
        codes : np.ndarray, shape (n_samples, K)
        """
        n_samples, _ = features.shape
        K = atoms.shape[0]
        # 1) squared Euclidean distances
        f2 = np.sum(features**2, axis=1)[:, None]
        a2 = np.sum(atoms**2, axis=1)[None, :]
        dist2 = f2 - 2 * features.dot(atoms.T) + a2

        # 2) find knn nearest atoms
        neigh_idx = np.argsort(dist2, axis=1)[:, :self.knn]
        codes = np.zeros((n_samples, K))

        # 3) solve regularized least squares per sample
        I = np.eye(self.knn)
        for i in range(n_samples):
            z = atoms[neigh_idx[i]] - features[i]
            C = z.dot(z.T)
            C += self.beta * max(np.trace(C), 1e-12) * I
            ones = np.ones(self.knn)
            w, *_ = np.linalg.lstsq(C, ones, rcond=None)
            w /= (w.sum() + 1e-12)
            codes[i, neigh_idx[i]] = w
        return codes
