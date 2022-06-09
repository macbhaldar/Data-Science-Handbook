from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# Importing the image and normalizing it.
def preprocess():
    img = np.array(Image.open('images/Penguins.jpg'))
    img = img/255
    points = np.reshape(
        img, (img.shape[0]*img.shape[1], img.shape[2]))
    return points, img

# Performing the K-means clustering on the image with different values of "K" and with random initialization of the center points of the clusters.


def k_means(X):
    K = [2, 5, 10, 15, 20]

    for k in K:
        for x in range(1, X+1):
            centers = np.random.rand(k, 3)
            for i in range(1, 11):
                points, img = preprocess()
                if (i == 0):
                    index = np.random.randint(k, size=(len(points)))
                else:
                    euclidean = []
                    for c in centers:
                        temp = np.linalg.norm((c - points), axis=1)
                        euclidean.append(temp)
                    index = np.argmin(euclidean, axis=0)

                Unique = np.unique(index)
                points_index_df = pd.DataFrame(
                    points, index=index, columns=None)
                mean_df = pd.DataFrame(columns=[0, 1, 2])

                for n in range(k):
                    temp = []
                    if n in Unique:
                        temp = points_index_df.loc[n, :]
                        if temp.shape == (3,):
                            temp = pd.DataFrame(
                                temp, columns=[n], index=None)
                            mean_df = mean_df.append(temp.T)
                        else:
                            temp = temp.mean(axis=0)
                            temp = pd.DataFrame(
                                temp, columns=[n], index=None)

                            mean_df = mean_df.append(temp.T)

                # Upadating the center of the clusters
                for j in range(k):
                    if (j in Unique):
                        centers[j] = mean_df.loc[j]

            for c in range(len(centers)):
                points_index_df.loc[c, 0] = centers[c, 0]
                points_index_df.loc[c, 1] = centers[c, 1]
                points_index_df.loc[c, 2] = centers[c, 2]

            result = points_index_df.to_numpy()

            shape = np.shape(img)
            result = np.reshape(result[0:shape[0]*shape[1], :], np.shape(img))
            plt.imshow(result)

            plt.title("For k = {} and random initialization = {}".format(k, x))
            plt.show()


if __name__ == "__main__":
    #points = preprocess()

    # Set value of "x" for number of times you want to perform clustering with random initialization.
    x = 15
    k_means(x)
