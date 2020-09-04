#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

CONVERGENCE_THS = 0.5

# Function to convert feature string to sample point(list of integers)
def to_vals(feature_str):
	pt = feature_str.strip("\n").split(",")
	pt = [float(p) for p in pt]
	return pt

#TODO: Read the input file and store it in the data structure
def read_data(path):
	"""
	Read the input file and store it in data_set.

	DO NOT CHANGE SIGNATURE OF THIS FUNCTION

	Args:
		path: path to the dataset

	Returns:
		data_set: a list of data points, each data point is itself a list of features:
			[
				[x_1, ..., x_n],
				...
				[x_1, ..., x_n]
			]
	"""
	File = open(path, "r")
	lines = File.readlines()
	lines = [to_vals(line) for line in lines]
	return lines


# TODO: Select k points randomly from your data set as starting centers.
def init_centers_random(data_set, k):
	"""
	Initialize centers by selecting k random data points in the data_set.
	
	DO NOT CHANGE SIGNATURE OF THIS FUNCTION

	Args:
		data_set: a list of data points, where each data point is a list of features.
		k: the number of mean/clusters.

	Returns:
		centers: a list of k elements: centers initialized using random k data points in your data_set.
				 Each center is a list of numerical values. i.e., 'vals' of a data point.
	"""
	num_pts = len(data_set)
	centers = []
	while(len(centers)<k):
		temp = random.randint(0, num_pts-1)
		if data_set[temp] in centers:
			continue
		else:
			centers.append(data_set[temp])
	return centers


# TODO: compute the euclidean distance from a data point to the center of a cluster
def dist(vals, center):
	"""
	Helper function: compute the euclidean distance from a data point to the center of a cluster

	Args:
		vals: a list of numbers (i.e. 'vals' of a data_point)
		center: a list of numbers, the center of a cluster.

	Returns:
		 d: the euclidean distance from a data point to the center of a cluster
	"""
	return sum((p-q)**2 for p, q in zip(vals, center)) ** .5

# TODO: return the index of the nearest cluster
def get_nearest_center(vals, centers):
	"""
	Assign a data point to the cluster associated with the nearest of the k center points.
	Return the index of the assigned cluster.

	Args:
		vals: a list of numbers (i.e. 'vals' of a data point)
		centers: a list of center points.

	Returns:
		c_idx: a number, the index of the center of the nearest cluster, to which the given data point is assigned to.
	"""
	c_idx = 0
	min_dist = dist(vals, centers[0])

	for idx, center in enumerate(centers):
		new_dist = dist(vals, center)
		if(new_dist < min_dist):
			min_dist = new_dist
			c_idx = idx
	return c_idx



# TODO: compute element-wise addition of two vectors.
def vect_add(x, y):
	"""
	Helper function for recalculate_centers: compute the element-wise addition of two lists.
	Args:
		x: a list of numerical values
		y: a list of numerical values

	Returns:
		s: a list: result of element-wise addition of x and y.
	"""
	return [sum(e) for e in zip(x, y)]


# TODO: averaging n vectors.
def vect_avg(s, n):
	"""
	Helper function for recalculate_centers: Averaging n lists.
	Args:
		s: a list of numerical values: the element-wise addition over n lists.
		n: a number, number of lists

	Returns:
		s: a list of numerical values: the averaging result of n lists.
	"""
	return [a/n for a in s]


# TODO: return the updated centers.
def recalculate_centers(clusters):
	"""
	Re-calculate the centers as the mean vector of each cluster.
	Args:
		 clusters: a list of clusters. Each cluster is a list of data_points assigned to that cluster.

	Returns:
		centers: a list of new centers as the mean vector of each cluster.
	"""
	centers = []
	for cluster in clusters:
		if(len(cluster)==0):
			centers.append([])
		sum_vect = cluster[0]
		for pt in cluster[1:]:
			sum_vect = vect_add(sum_vect, pt)
		centers.append(vect_avg(sum_vect, len(cluster)))
	return centers


# TODO: run kmean algorithm on data set until convergence or iteration limit.
def train_kmean(data_set, centers, iter_limit):

	num_iter=0
	while True:
		num_iter+=1
		# print("Runnning Iteration: ", num_iter)
		

		#Create Cluster array
		clusters = []
		for center in centers:
			clusters.append([])

		## Assign clusters
		for pt in data_set:
			c_idx = get_nearest_center(pt, centers)
			clusters[c_idx].append(pt)

		## Calculate Centers
		centers = recalculate_centers(clusters)

		## Check Convergence
		sss_new = sum_of_within_cluster_ss(clusters, centers)
		# print(sss_new)
		if(num_iter==1): 
			sss_old = sss_new
			continue

		diff = sss_old - sss_new
		print("Runnning Iteration: ", num_iter, "\tChange in Within_cluster_SS: ", diff)
		sss_old = sss_new

		if(diff < CONVERGENCE_THS):
			print("Convergence reached!!")
			return centers, clusters, num_iter
		if(num_iter==iter_limit):
			print("Iteration Limit reached!!")
			return centers, clusters, num_iter


	"""
	DO NOT CHANGE SIGNATURE OF THIS FUNCTION

	Args:
		data_set: a list of data points, where each data point is a list of features.
		centers: a list of initial centers.
		iter_limit: a number, iteration limit

	Returns:
		centers: a list of updates centers/mean vectors.
		clusters: a list of clusters. Each cluster is a list of data points.
		num_iterations: a number, num of iteration when converged.
	"""



# TODO: helper function: compute within group sum of squares
def within_cluster_ss(cluster, center):

	"""
	For each cluster, compute the sum of squares of euclidean distance
	from each data point in the cluster to the empirical mean of this cluster.
	Please note that the euclidean distance is squared in this function.

	Args:
		cluster: a list of data points.
		center: the center for the given cluster.

	Returns:
		ss: a number, the within cluster sum of squares.
	"""
	ss = 0
	for pt in cluster:
		ss += dist(pt, center) ** 2
	return ss


# TODO: compute sum of within group sum of squares
def sum_of_within_cluster_ss(clusters, centers):
	"""
	For total of k clusters, compute the sum of all k within_group_ss(cluster).

	DO NOT CHANGE SIGNATURE OF THIS FUNCTION

	Args:
		clusters: a list of clusters.
		centers: a list of centers of the given clusters.

	Returns:
		sss: a number, the sum of within cluster sum of squares for all clusters.
	"""
	sss = 0
	for cluster, center in zip(clusters, centers):
		sss += within_cluster_ss(cluster, center)
	return sss

def k_means_single(data_path, iter_limit, k):

	data_set = read_data(data_path)
	centers = init_centers_random(data_set, k)
	centers, clusters, num_iter = train_kmean(data_set, centers, iter_limit)
	print("\n")
	sss = sum_of_within_cluster_ss(clusters, centers)
	print("Value of within_cluster_SS achieved: ", sss)
	print()
	return sss

def k_means_multiple(data_path, iter_limit, k, path_to_figure):

	sss_list = []
	for k_val in k:
		sss_list.append(k_means_single(data_path, iter_limit, k_val))

	# Show plot
	plt.plot(k, sss_list)
	plt.xlabel('Values of k')
	plt.ylabel('Within Cluster Sum of Squares')
	plt.show()

	# save plot to file
	plt.savefig(path_to_figure)



if __name__ == '__main__':

	data_path = "wine.txt"  # Change the data_path according to the file that you want to cluster
	iter_limit = 20


	################### Use this to perform clustering for a single value of k ###################
	# k = 3
	# k_means_single(data_path, iter_limit, k)

	##################  Use this to perform clustering for a list of values of k ##################
	k = [i for i in range(1,20)]
	path_to_figure = "plot_1_19.png"
	k_means_multiple(data_path, iter_limit, k, path_to_figure)





