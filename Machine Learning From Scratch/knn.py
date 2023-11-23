import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import heapq

class KNN():
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.label_counts = defaultdict(int)
        for label in set(self.data['Label']):
              self.label_counts[label] = 0

    @staticmethod
    def euclidean_dist(coord1,coord2):
        x1, y1 = coord1[0], coord1[1]
        x2, y2 = coord2[0], coord2[1]

        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def predict(self, test_point):
        distances = []
        for index, row in self.data.iterrows():
            coord = row['Coordinates']
            label = row['Label']
            dist = self.euclidean_dist(test_point, coord)
            distances.append((dist, label))

        # Getting the k nearest neighbors
        k_nearest = heapq.nsmallest(self.k, distances)

        # Counting the labels of the k nearest neighbors
        label_counts = defaultdict(int)
        for dist, label in k_nearest:
            label_counts[label] += 1

        # Finding the most frequent label
        max_count = max(label_counts.values())
        predictions = [label for label, count in label_counts.items() if count == max_count]

        # Handling ties
        return np.random.choice(predictions) if len(predictions) > 1 else predictions[0]


# functions to generate synthetic data - points along a circle
def generate_data(N): # N examples
    df = defaultdict(int)
    for _ in range(N):
        x = np.random.random()
        p_x = np.random.uniform()
        # subtract 1 50% of the time to get negative values since x is only in [0,1)
        if p_x <= 0.5:
              x = x - 1
        y = math.sqrt(1-x**2)
        # also take negative of y 50% of the time
        p_y = np.random.uniform()
        if p_y <= 0.5:
              y = y * - 1
        # color based on which quadrant on the unit circle
        if x < 0 and y < 0:
            df[(x,y)] = 'Red'
        if x < 0 and y > 0:
              df[(x,y)] = 'Yellow'
        if x > 0 and y < 0:
              df[(x,y)] = 'Purple'
        if x > 0 and y > 0:
              df[(x,y)] = 'Green'

    df_pd = pd.DataFrame({
          'Coordinates' : df.keys(),
          'Label' : df.values()
    })
    return df_pd

def main():
    N = 1000
    df = generate_data(N)
    df['X'] = df['Coordinates'].apply(lambda x : x[0])
    df['Y'] = df['Coordinates'].apply(lambda x : x[1])
    knn = KNN(data=df,k=5)
    print(knn.predict((0.8,0.01)))

if __name__ == "__main__":
    main()