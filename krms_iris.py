# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:48:36 2018

@author: hp
"""

#############################################################################
# Full Imports

from __future__ import division
import math
import random
import time
from sklearn.datasets import load_iris

a=[]
total_iterations=0

start_time=time.time()
def main(Name):

    # How many points are in our dataset?
    #num_points = 50

    # For each of those points how many dimensions do they have?
    # Note: Plotting will only work in two or three dimensions
    dimensions = 3

    # Bounds for the values of those points in each dimension
   # lower = 0
    #upper = 200

    # The K in k-rms. How many clusters do we assume exist?
    #   - Must be less than num_points
    num_clusters = 3

    # When do we say the process has 'converged' and stop updating clusters?
    cutoff = .01#Scope of improvement.Making it automatic using DL algos.For faster speeds increase its value.
    
    # Generate some points to cluster
    # Note: If you want to use your own data, set points equal to it here.
    if Name!='' or Name!=None:
        import pandas as pd
        dataset = pd.read_csv(Name)
        train = dataset.iloc[:, 0:len(dataset.columns)-1].values
        Y = list(dataset.iloc[:, len(dataset.columns)].values)
        #from sklearn.preprocessing import MinMaxScaler
        from sklearn.manifold import TSNE
        train=TSNE(n_components=3).fit_transform(train)
    else:
        iris=load_iris()
        train=iris.data
        Y=iris.target
    
    points=[Point(i[:3]) for i in train]#Method for converting 2 arrays to(x,y) form

    # Cluster those data!
    iteration_count =5 #Scope of improvement using GRADIENT DELUDE algo.Decide the value automatically.
    best_clusters = iterative_krms(
        points,
        num_clusters,
        cutoff,
        iteration_count
    )
    z=[]
    # Print our best clusters
    List=dict()
    for i, c in enumerate(best_clusters):
        for p in c.points:
            List["Cluster:%s"%(i)]="Point :"+ str(p)
            #print( " Cluster: ", i, "\t Point :", p)
            z.append(i+1)
        
    
    from sklearn import metrics as sm
    #print("Accuracy:",sm.accuracy_score(Y,z))
    List["Accuracy:"]=str(sm.accuracy_score(Y,z))
    return List

#############################################################################
# K-rms Methods

def iterative_krms(points, num_clusters, cutoff, iteration_count):
    """
    K-rms isn't guaranteed to get the best answer the first time. It might
    get stuck in a "local minimum."

    Here we run krms() *iteration_count* times to increase the chance of
    getting a good answer.

    Returns the best set of clusters found.
    """
    print ("Running K-rms %d times to find best clusters ..." % iteration_count)
    candidate_clusters = []
    errors = []
    for _ in range(iteration_count):
        clusters = krms(points, num_clusters, cutoff)
        error = calculateError(clusters)
        candidate_clusters.append(clusters)
        errors.append(error)

    highest_error = max(errors)
    lowest_error = min(errors)
    print ("Lowest error found: %.2f (Highest: %.2f)" % (
        lowest_error,
        highest_error
    ))
    ind_of_lowest_error = errors.index(lowest_error)
    best_clusters = candidate_clusters[ind_of_lowest_error]

    return best_clusters

def krms(points, k, cutoff):

    # Pick out k random points to use as our initial centroids
    initial_centroids = random.sample(points, k)
#Scope of improvement using GRAD DELUDE algorithm or Genetic algorithms
    # Create k clusters using those centroids
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster([p]) for p in initial_centroids]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(1, clusterCount):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i
            # After finding the cluster the smallest distance away
            # set the point to belong to that cluster
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c.points) != 0]
        
        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            #print ("Converged after %s iterations" % loopCounter)
            a.append(loopCounter)
            break
    return clusters


#############################################################################
# Classes

class Point(object):
    '''
    A point in n dimensional space
    '''
    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster(object):
    '''
    A set of points and their centroid
    '''

    def __init__(self, points):
        '''
        points - A list of point objects
        '''

        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n:
                raise Exception("ERROR: inconsistent dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centre and the new after
        recalculating and storing the new centre.

        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        old_centroid = self.centroid
        self.points = points
        # Return early if we have no points, this cluster will get
        # cleaned up (removed) in the outer loop.
        if len(self.points) == 0:
            return 0

        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.sqrt(math.fsum(d**2 for d in dList )/numPoints)for dList in unzipped]#Change this part to rms to make Krms
       # centroid_coords = [math.sqrt(math.fsum(d**2 for d in dList )/numPoints)for dList in unzipped]
        return Point(centroid_coords)

    def getTotalDistance(self):
        '''
        Return the sum of all squared Euclidean distances between each point in 
        the cluster and the cluster's centroid.
        '''
        sumOfDistances = 0.0
        for p in self.points:
            sumOfDistances += getDistance(p, self.centroid)

        return sumOfDistances

#############################################################################
# Helper Methods

def getDistance(a, b):
    '''
    Squared Euclidean distance between two n-dimensional points.
    https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ERROR: non comparable points")

    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        #squareDifference = abs((a.coords[i]-b.coords[i]))#[Manhattan error]
        accumulatedDifference += squareDifference

    #return math.sqrt(accumulatedDifference/a.n)
    return (accumulatedDifference/a.n)#[For Manhattan error]

def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for _ in range(n)])
    return p

def calculateError(clusters):
    '''
    Return the average squared distance between each point and its cluster
    centroid.

    This is also known as the "distortion cost."
    '''
    accumulatedDistances = 0
    num_points = 0
    for cluster in clusters:
        num_points += len(cluster.points)
        accumulatedDistances += cluster.getTotalDistance()

    error = accumulatedDistances / num_points
    return error
