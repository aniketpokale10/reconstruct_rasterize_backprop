'''
This code will calculate the median azimuth, median elevation and median translation error for the 
predicted mesh and the view point network mesh with respect to the ground truth.
Azimuth and elevation errors are in degree
Command:
python3 pose_error.py <gt_view_parameters_file> <pred_numpy_file_directory> <view_point_network_parameter_file>
	The gt view parameter file should be of the following format:
	chair id image_name azimuth_angle elevation_angle inplace_rotation radius
'''

import numpy as np
import math
import os
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import mat2euler
import argparse


# Returns the translation vector given azimuth, elevation
def get_initialization(d,az,el):

	az = az*np.pi/180
	el = el*np.pi/180
	x = d*np.cos(az)*np.cos(el);
	y = -1*d*np.sin(az)*np.cos(el);
	z = d*np.sin(el); 
	return [x,y,z]

# Returns the azimuth given [x,y,z]
def find_az(x,y,z):
	az = math.atan(y/x)*180/math.pi

	if(x<0 and y>0):
		az = 180+az
	if(x>0 and y<0):
		az = 360+az
	if(x<0 and y<0):
		az = 180+az
	return az

# Returns the elevation given [x,y,z]
def find_ele(x,y,z):
	d = math.sqrt(x*x+y*y+z*z)
	el = math.acos(math.sqrt(x*x+y*y) / d) * (180/np.pi)
	return el


parser = argparse.ArgumentParser()
parser.add_argument('gt_file', help='View Parameter file ground truth')
parser.add_argument('pred_dir', help='Directory containing predicted view parameters in numpy file')
parser.add_argument('viewpoint_network_file', help='View Parameter file for View point network')

args = parser.parse_args()

gt_file = args.gt_file
pred_dir = args.pred_dir
vp_file = args.viewpoint_network_file

with open(gt_file, 'r') as m_file:
	reqd_image_data = m_file.read().split('\n')

# Pose error in the predictions by RRB
err_azi = []
err_ele = []
err_trans = []

for img_data in reqd_image_data:
	img_name = img_data.split(' ')[0]
	azi = float(img_data.split(' ')[2])
	ele = float(img_data.split(' ')[3])
	radius = float(img_data.split(' ')[5])
	# Azimuth for gt starts clockwise from x-axis
	azi_gt = 360 - azi
	ele_gt = ele
	translation_gt = get_initialization(radius, azi, ele)

	pred_file = img_data.split(' ')[0] + '_' + (img_data.split(' ')[1]).split('.')[0] + '.npy'
	file_path = os.path.join(pred_dir, pred_file)
	pred_data = np.load(file_path)
	T = np.vstack((pred_data, np.array([[0,0,0,1]], dtype = 'float32')))
	T = np.linalg.inv(T)
	trans = T[:3,3]
	translation_pred = trans

	azi_pred = find_az(trans[0], trans[1],trans[2])
	ele_pred = find_ele(trans[0], trans[1],trans[2])

	error = np.abs(azi_gt - azi_pred)
	err_azi.append(error)

	error = np.abs(ele_gt - ele_pred)
	err_ele.append(error)

	error = np.linalg.norm(translation_gt - translation_pred)
	err_trans.append(error)

print ('Predicted Mesh Error (RRB) wrt to ground truth')
print ('Median Azimuth Error ' + str(np.median(np.array(err_azi))))
print ('Median Elevation Error ' + str(np.median(np.array(err_ele))))
print ('Median Translation Error ' + str(np.median(np.array(err_trans))))


# Pose error in the predictions by Viewpoint Network Initialization
err_azi = []
err_ele = []
err_trans = []

with open(vp_file, 'r') as n_file:
	vp_data = n_file.read().split('\n')

for idx, img_data in enumerate(reqd_image_data):
	img_name = img_data.split(' ')[0]
	azi = float(img_data.split(' ')[2])
	ele = float(img_data.split(' ')[3])
	radius = float(img_data.split(' ')[5])

	## Azimuth for predicted for RforCNN starts counterclockwise from x-axis
	azi_gt = 360 - azi
	ele_gt = ele

	azi_pred = float(vp_data[idx].split(' ')[1])
	ele_pred = float(vp_data[idx].split(' ')[2])
	
	error = np.abs(azi_gt - azi_pred)
	err_azi.append(error)

	error = np.abs(ele_gt - ele_pred)
	err_ele.append(error)

print ('Mesh Error (View point Network) wrt to ground truth')
print ('Median Azimuth Error ' + str(np.median(np.array(err_azi))))
print ('Median Elevation Error ' + str(np.median(np.array(err_ele))))