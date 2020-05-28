'''
This code will return the Rotation Matrix and translation vector for the ground truth and predicted dense mesh.
Command:
python3 save_R.py <gt_view_params> <pred_view_params_dir>
The gt_view_params file should be of the following format:
	chair id image_name azimuth_angle elevation_angle inplace_rotation radius
The pred_view_params_dir should be the path to directory containing predicted pose in npy file
'''


import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import math
from pathlib import Path

import torch
import torch.nn as nn
from skimage.io import imread, imsave, imshow 
from skimage.color import rgb2gray
import tqdm
import imageio
import torchgeometry as tgm
from scipy.spatial.transform import Rotation as R

import neural_renderer as nr
import sys
import argparse

# Given the radius, azimuth_angle, elevation_angle from viewpoint network return R and T
def get_initialization(d,az,el):

	az = az*np.pi/180
	el = el*np.pi/180
	x = d*np.cos(az)*np.cos(el);
	y = d*np.sin(az)*np.cos(el);
	z = d*np.sin(el); 
	c = np.array([[x,y,z]],dtype='float32')
	at = np.array([[0,0,0]],dtype='float32') 
	up = np.array([[0,0,1]],dtype='float32')
	z_axis = at-c; 
	z_axis = z_axis/np.linalg.norm(z_axis);
	x_axis = np.cross(z_axis,up);
	x_axis = x_axis/np.linalg.norm(x_axis);
	y_axis = np.cross(z_axis,x_axis);
	y_axis = y_axis/np.linalg.norm(y_axis);

	R = np.vstack((x_axis,y_axis,z_axis))
	R = R.T

	Rt = np.hstack((R,c.T))  
	T = np.vstack((Rt, np.array([[0,0,0,1]],dtype='float32')))
	T = np.linalg.inv(T) 

	return T

def find_az(x,y,z):
	az = math.atan(y/x)*180/math.pi

	if(x<0 and y>0):
		az = 180+az
	if(x>0 and y<0):
		az = 360+az
	if(x<0 and y<0):
		az = 180+az
	return az


def find_ele(x,y,z):
	d = math.sqrt(x*x+y*y+z*z)
	el = math.acos(math.sqrt(x*x+y*y) / d) * (180/np.pi)
	return el

parser = argparse.ArgumentParser()
parser.add_argument('gt_view_params', help='Ground Truth view parameters')
parser.add_argument('pred_view_params', help='Predicted View parameters directory in npy file')
args = parser.parse_args()

gt_params = args.gt_view_params
pred_params = args.pred_view_params

with open (gt_params, 'r') as fp:
	object_list = fp.read().split('\n')

# Rotation matrix and translation vector for ground truth mesh
for i in range(len(object_list)):
	md5 = object_list[i].split(' ')[0]
	img_number = object_list[i].split(' ')[1].split('.')[0]

	az = -1*float(object_list[i].split(' ')[2])
	ele = float(object_list[i].split(' ')[3])
	d = float(object_list[i].split(' ')[5])

	T_gt = get_initialization(d, az, ele)


# Rotation matrix and translation vector for predicted mesh
for i in range(len(object_list)):
	md5 = object_list[i].split(' ')[0]
	img_number = object_list[i].split(' ')[1].split('.')[0]

	pose_file = md5 + '_' + img_number + '.npy'
	view_file = os.path.join(pred_params, pose_file)

	RT_matrix_pred = (np.load(view_file).astype('float32'))
	RT_matrix = np.vstack((RT_matrix, np.array([[0,0,0,1]],dtype='float32')))