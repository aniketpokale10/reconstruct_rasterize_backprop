'''
This code will take input images, initialization by the viewpoint network and list of images.
It will iteratively optimize for the pose of the object. 
It will generate a numpy file containing the 3X4 Pose Matrix for the object
Command:
python3 run_nr_batch_rfcnn.py <view_parameters_file> <rescaled_mesh_dir> <input_img_dir>
The view parameter file should be of the following format:
	image_name azimuth_angle elevation_angle inplace_rotation

'''

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
import math
import argparse


# Saving all the poses to a file
def save_all_poses_to_file(i,image):
	RT_matrix = np.load('data/poses/out_poses/'+image+'.npy')
	R = RT_matrix[:,:3]
	t = RT_matrix[:,3]
	f = open('data/poses/all_poses.txt', 'a+')
	f.write("%d\n" % i)
	for i in range(3):
		f.write("%5.3f %5.3f %5.3f\n" % (R[i][0], R[i][1], R[i][2]))
	f.write("%5.3f %5.3f %5.3f\n" % (t[0], t[1], t[2]))
	f.write("\n")
	f.close()


# Given the frame number, radius, azimuth_angle, elevation_angle from viewpoint network return R and T
def get_initialization(frame,d,az,el):

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
	T = np.linalg.inv(T) # here inverse is taken since in projection.py it does R*x + t

	np.save("data/poses/init_poses/RT_matrix_init_"+frame+".npy",T[0:3,:])


# Known Camera calibration matrix
K = np.array([[245.0, 0.0, 112.0],[0.0, 245.0, 112.0],[0.0, 0.0, 1.0]],dtype=np.float32) # for resolution of 224x224

parser = argparse.ArgumentParser()
parser.add_argument('view_params', help='View Parameter file from the Viewpoint network which acts as initialization')
parser.add_argument('mesh_dir', help='Path of the rescaled meshes by o-net')
parser.add_argument('img_dir', help='Input image path with white background')
args = parser.parse_args()

view_file_path = args.view_params
meshes_path = args.mesh_dir
ref_image_path = args.img_dir

with open(view_file_path) as f:
	chairdata = f.readlines()
f.close()

for idx,item in enumerate(chairdata):
		item = item.split(' ')
		image = item[0].split('.')[0]

		azimuth_deg,elevation_deg,theta_deg =  float(item[1]),float(item[2]),float(item[3])
		radius = 1.5

		get_initialization(image,radius,azimuth_deg,elevation_deg)

		chair_filename = os.path.join(meshes_path,image)+'.obj'
		ref_image = os.path.join(ref_image_path,image+'.png')
		cmd = "python example4.py -rp poses/init_poses/RT_matrix_init_"+image+".npy -ir "+ref_image+ \
		" -or data/out_gifs/example4_result"+image + ".gif -om 1 -io "+chair_filename
		os.system(cmd)
		os.system("mv data/poses/RT_matrix.npy data/poses/out_poses/"+image+".npy")

		save_all_poses_to_file(idx,image)