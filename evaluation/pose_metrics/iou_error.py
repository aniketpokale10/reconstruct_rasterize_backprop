'''
This code will calculate the 3D IOU Bounding Box error of predicted mesh wrt to the ground truth.
Command:
python3 iou_error.py <gt_bb_file> <pred_bb_file> 
	The format of the bb files is:
	chair_id img_name x_min x_max y_min y_max z_min z_max
'''

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import argparse

def get_bb(data):
	x_min = float(data.split(' ')[2])
	x_max = float(data.split(' ')[3])
	y_min = float(data.split(' ')[4])
	y_max = float(data.split(' ')[5])
	z_min = float(data.split(' ')[6])
	z_max = float(data.split(' ')[7])

	bbox = np.array([[x_min, y_min, z_min],
					 [x_min, y_max, z_min],
					 [x_min, y_max, z_max],
					 [x_min, y_min, z_max],
					 [x_max, y_min, z_min],
					 [x_max, y_max, z_min],
					 [x_max, y_max, z_max],
					 [x_max, y_min, z_max]])
	return bbox


def compute_3d_iou(bbox_3d_1, bbox_3d_2):
	# Computes IoU overlaps between two 3d bboxes.
		# bbox_3d_1, bbox_3d_1: [3, 8]
	def asymmetric_3d_iou(bbox_3d_1, bbox_3d_2):
		bbox_1_max = np.amax(bbox_3d_1, axis=0)
		bbox_1_min = np.amin(bbox_3d_1, axis=0)
		bbox_2_max = np.amax(bbox_3d_2, axis=0)
		bbox_2_min = np.amin(bbox_3d_2, axis=0)

		overlap_min = np.maximum(bbox_1_min, bbox_2_min)
		overlap_max = np.minimum(bbox_1_max, bbox_2_max)

		# intersections and union
		if np.amin(overlap_max - overlap_min) <0:
			intersections = 0
		else:
			intersections = np.prod(overlap_max - overlap_min)
		union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
		overlaps = intersections / union
		return overlaps

	if bbox_3d_1 is None or bbox_3d_2 is None:
		return -1
	return asymmetric_3d_iou(bbox_3d_1, bbox_3d_2)

# Plots the bb in 3D
def plot_bb(bb_gt, bb_nr):
	fig = plt.figure(frameon=False, figsize=(7, 7))
	fig.patch.set_facecolor('white')
	ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0), projection='3d')
	ax.clear()
	ax.plot([bb_gt[0,0], bb_gt[1,0]], [bb_gt[0,1], bb_gt[1,1]], [bb_gt[0,2], bb_gt[1,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[1,0], bb_gt[2,0]], [bb_gt[1,1], bb_gt[2,1]], [bb_gt[1,2], bb_gt[2,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[2,0], bb_gt[3,0]], [bb_gt[2,1], bb_gt[3,1]], [bb_gt[2,2], bb_gt[3,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[3,0], bb_gt[0,0]], [bb_gt[3,1], bb_gt[0,1]], [bb_gt[3,2], bb_gt[0,2]], linewidth=3, antialiased=True, color = 'r')

	ax.plot([bb_gt[4,0], bb_gt[5,0]], [bb_gt[4,1], bb_gt[5,1]], [bb_gt[4,2], bb_gt[5,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[5,0], bb_gt[6,0]], [bb_gt[5,1], bb_gt[6,1]], [bb_gt[5,2], bb_gt[6,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[6,0], bb_gt[7,0]], [bb_gt[6,1], bb_gt[7,1]], [bb_gt[6,2], bb_gt[7,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[7,0], bb_gt[4,0]], [bb_gt[7,1], bb_gt[4,1]], [bb_gt[7,2], bb_gt[4,2]], linewidth=3, antialiased=True, color = 'r')

	ax.plot([bb_gt[0,0], bb_gt[4,0]], [bb_gt[0,1], bb_gt[4,1]], [bb_gt[0,2], bb_gt[4,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[1,0], bb_gt[5,0]], [bb_gt[1,1], bb_gt[5,1]], [bb_gt[1,2], bb_gt[5,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[2,0], bb_gt[6,0]], [bb_gt[2,1], bb_gt[6,1]], [bb_gt[2,2], bb_gt[6,2]], linewidth=3, antialiased=True, color = 'r')
	ax.plot([bb_gt[3,0], bb_gt[7,0]], [bb_gt[3,1], bb_gt[7,1]], [bb_gt[3,2], bb_gt[7,2]], linewidth=3, antialiased=True, color = 'r')


	ax.plot([bb_nr[0,0], bb_nr[1,0]], [bb_nr[0,1], bb_nr[1,1]], [bb_nr[0,2], bb_nr[1,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[1,0], bb_nr[2,0]], [bb_nr[1,1], bb_nr[2,1]], [bb_nr[1,2], bb_nr[2,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[2,0], bb_nr[3,0]], [bb_nr[2,1], bb_nr[3,1]], [bb_nr[2,2], bb_nr[3,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[3,0], bb_nr[0,0]], [bb_nr[3,1], bb_nr[0,1]], [bb_nr[3,2], bb_nr[0,2]], linewidth=3, antialiased=True, color = 'b')

	ax.plot([bb_nr[4,0], bb_nr[5,0]], [bb_nr[4,1], bb_nr[5,1]], [bb_nr[4,2], bb_nr[5,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[5,0], bb_nr[6,0]], [bb_nr[5,1], bb_nr[6,1]], [bb_nr[5,2], bb_nr[6,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[6,0], bb_nr[7,0]], [bb_nr[6,1], bb_nr[7,1]], [bb_nr[6,2], bb_nr[7,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[7,0], bb_nr[4,0]], [bb_nr[7,1], bb_nr[4,1]], [bb_nr[7,2], bb_nr[4,2]], linewidth=3, antialiased=True, color = 'b')

	ax.plot([bb_nr[0,0], bb_nr[4,0]], [bb_nr[0,1], bb_nr[4,1]], [bb_nr[0,2], bb_nr[4,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[1,0], bb_nr[5,0]], [bb_nr[1,1], bb_nr[5,1]], [bb_nr[1,2], bb_nr[5,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[2,0], bb_nr[6,0]], [bb_nr[2,1], bb_nr[6,1]], [bb_nr[2,2], bb_nr[6,2]], linewidth=3, antialiased=True, color = 'b')
	ax.plot([bb_nr[3,0], bb_nr[7,0]], [bb_nr[3,1], bb_nr[7,1]], [bb_nr[3,2], bb_nr[7,2]], linewidth=3, antialiased=True, color = 'b')

	ax.plot([0.], [0.], [0.], markerfacecolor='w', markeredgecolor='w', marker='o', markersize=5, alpha=0.6)
	plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('gt_bb_file', help='Ground Truth Bounding Box File')
parser.add_argument('pred_bb_file', help='Predicted Bounding Box File')
args = parser.parse_args()


gt_file = args.gt_bb_file
nr_file = args.pred_bb_file

with open(gt_file, 'r') as fp:
	gt_data = fp.read().split('\n')

with open(nr_file, 'r') as fp:
	nr_data = fp.read().split('\n')


iou = []
for i in range(len(nr_data)):
	bb_gt = get_bb(gt_data[i])
	bb_nr = get_bb(nr_data[i])
	val  = compute_3d_iou(bb_gt, bb_nr)
	iou.append(val)

print ('Mean 3D IOU Bounding Box Error ' + str(np.array(np.mean(iou))))
