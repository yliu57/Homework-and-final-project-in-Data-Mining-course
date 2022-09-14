import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# K-Means algorithm Implementation
class Kmeans:
    def __init__(self, n_clusters=3, random_state=0, n_iters=1000):
        assert n_clusters >= 1, "must be valid"
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._iters = n_iters
        self._X = None
        self.cluster_centers_ = None
        # within cluster sum of squares
        self.wss_ = 0

    def distance(self, M, N):
        return (np.sum((M - N) ** 2, axis=1)) ** 0.5

    def distance_kmpp(self, M, N):  # kmpp = kmeans ++
        M = np.array(M)
        N = np.array(N)
        return np.sum((M - N) ** 2)

    def _generate_labels(self, center, X):
        return np.array([np.argmin(self.distance(center, item)) for item in X])

    def _generate_centers(self, labels, X):
        return np.array([np.average(X[labels == i], axis=0) for i in np.arange(self._n_clusters)])

    def _generate_wss(self, center, labels, X):
        # calculate the wss(within cluster sum of squares) for every point, for every item in X
        wss_sum = 0
        x_index = 0
        for label in labels:
            wss_sum += np.sum((X[x_index] - center[label]) ** 2)
            x_index += 1
        return wss_sum

    def fit_predict(self, X):
        k = self._n_clusters

        # set the random seed
        if self._random_state:
            random.seed(self._random_state)

        # Kmeans ++ algorithm ------------------------------------------------------------------------------
        # Generate the index of the first clustering center randomly
        center_index = [random.randint(0, X.shape[0]) for i in np.arange(1)]

        i = 0;
        original_point_selection_range = X.tolist()
        original_point_selection_range.pop(center_index[0])
        length = len(original_point_selection_range)
        # Get their specific coordinate attribute through the corresponding index of a randomly selected point
        center = X[center_index]
        #  item_sqd: item_square shortest distance
        item_sqd = []

        # Choose the k-1 clustering center(The first clustering center is chosen randomly)
        while i < k - 1:
            for item in original_point_selection_range:
                closest_distance = -1
                for centerpoint in center:
                    centerpoint1 = []
                    centerpoint1.append(centerpoint)
                    centerpoint = centerpoint1
                    distance = self.distance_kmpp(centerpoint, item)
                    if closest_distance == -1:
                        closest_distance = distance
                    else:
                        if distance < closest_distance:
                            closest_distance = distance

                item_sqd.append(closest_distance)

            sumd = 0  # sum of all points' distance which is the distance to the closest clustering point
            for d in item_sqd:
                sumd += d

            probability_proportional = []
            for item in item_sqd:
                probability_proportional.append(item / sumd)
            length_array = np.arange(length)
            new_point_choice = np.random.choice(length_array, 1, p=probability_proportional)

            new_array = []
            for num in original_point_selection_range[new_point_choice[0]]:
                new_array.append(num)
            center = center.tolist()
            center.append(new_array)
            center = np.array(center)
            original_point_selection_range.pop(new_point_choice[0])
            length = length - 1
            item_sqd = []

            i += 1

        n_iters = self._iters
        while n_iters > 0:

            # Record the coordinates of the center point of the previous iteration
            last_center = center

            # According to the previous batch of center points,
            # calculate the distance from each point to each center point and determine the class to which it belongs
            labels = self._generate_labels(last_center, X)
            self.labels_ = labels

            # compute WSS (within cluster sum of squares)
            wss = self._generate_wss(last_center, labels, X)
            self.wss_ = wss

            # new clustering center
            center = self._generate_centers(labels, X)

            # store the clustering center in classifier
            self.cluster_centers_ = center

            # Compare the last_centers and the new centers. If they are same, it means that the Kmeans loop is end.
            if (last_center == center).all():
                break

            n_iters = n_iters - 1

        return self


# Data preprocessing: read the data by pandas packages and normalize the data
print("Data preprocessing starts...")
features = pd.read_csv('./Image_Test.txt', header=None, sep=',')
features = np.array(features)
scaler = MinMaxScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)
scaled_features = np.array(scaled_features)

# Call the Kmeans method and store all the return results in the classifier
print("Kmeans classifier starts...")
k_array = []
wss_array = []
for i in range(2, 21, 2):
    clf = Kmeans(n_clusters=i)
    predict = clf.fit_predict(scaled_features)
    print("k of kmeans: ", i)
    print("wss(within cluster sum of squares): ", clf.wss_)
    k_array.append(i)
    wss_array.append(clf.wss_)
    # Output the classifier labels when k = 10 and make a distribution graph
    if i == 10:
        label = pd.DataFrame(clf.labels_)
        label.to_csv('./part2_label.txt', index=False, header=None)

        center = clf.cluster_centers_

        pca = PCA(n_components=2).fit(scaled_features)
        pca_2d_scaled_features = pca.transform(scaled_features)
        pca_2d_center = pca.transform(center)

        plt.scatter(pca_2d_scaled_features[:, 0], pca_2d_scaled_features[:, 1], c=clf.labels_)
        plt.grid()
        plt.scatter(pca_2d_center[:, 0], pca_2d_center[:, 1], marker="*", s=500, color="red")
        plt.title("The graph of 10000 data points' distribution and 10 clustering centers")
        plt.xlabel("The first column feature(total number of columns: 2, after PCA decomposition)")
        plt.ylabel("The second column feature(total number of columns: 2, after PCA decomposition)")
        plt.show()

# Use matplotlib to make the graph
print("Graph making starts...")
plt.figure(figsize=(15, 15))
plt.scatter(k_array, wss_array)
plt.plot(k_array, wss_array)
plt.grid()
plt.xlim(2, 20)
plt.ylim(340000, 500000)
plt.title("k - wss relation graph")
plt.xlabel("k of kmeans")
plt.ylabel("wss (within cluster sum of squares)")
plt.show()