"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave, imshow 
from skimage.color import rgb2gray
import tqdm
import imageio
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
	def __init__(self, filename_obj,filename_ref=None, cam_ref_pose='RT_matrix_init.npy'):
		super(Model, self).__init__()
		# load .obj
		vertices, faces = nr.load_obj(filename_obj,normalization=False) 

		self.register_buffer('vertices', vertices[None, :, :])
		self.register_buffer('faces', faces[None, :, :])

		# create textures
		texture_size = 2
		textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
		self.register_buffer('textures', textures)

		# load reference image
		if filename_ref:
			image_ref = torch.from_numpy((imread(filename_ref) != 0).astype(np.float32))
			self.register_buffer('image_ref', image_ref)
			if self.image_ref.dim() > 2:
				self.image_ref = self.image_ref[:,:,0]

		camera_mode = 'projection'
		# camera parameters
		if camera_mode == 'look_at':
			self.camera_position = nn.Parameter(torch.from_numpy(np.array(nr.get_points_from_angles(0.96600807, -1.1112652,   0.2862135), dtype=np.float32)))
			# setup renderer
			renderer = nr.Renderer(camera_mode='look_at')
			renderer.eye = self.camera_position

		if camera_mode == 'projection':
			renderer = nr.Renderer(camera_mode='projection')
			renderer.K = np.array([[245.0, 0.0, 112.0],[0.0, 245.0, 112.0],[0.0, 0.0, 1.0]],dtype=np.float32) # for resolution of 224x224
			RT_matrix = (np.load(os.path.join(data_dir,cam_ref_pose))).astype('float32')


			rot_quat0 = (R.from_dcm(RT_matrix[:,:3])).as_quat() # as_quat() will return [xsin(t/2),zsin(t/2),zsin(t/2),cos(t/2) ]
			rot_quat = np.array([rot_quat0[3],rot_quat0[0],rot_quat0[1],rot_quat0[2]],dtype='float32') # converting the [cos(t/2),xsin(t/2),zsin(t/2),zsin(t/2)]
			
			renderer.R = nn.Parameter(torch.from_numpy(np.array(rot_quat,dtype='float32')))
			renderer.t = nn.Parameter(torch.from_numpy(RT_matrix[:,3]))

			renderer.orig_size = 224
			renderer.image_size = 224

			#Add huber loss function
		self.loss_function = nn.SmoothL1Loss(reduction='sum')


		self.renderer = renderer


	def rotation_regularizer(self):
		rot_angle_axis = tgm.quaternion_to_angle_axis(renderer.R[0]) 
		rot_matrix = tgm.angle_axis_to_rotation_matrix(rot_angle_axis)
		print("Inside rotation regularizer:")
		print(rot_matrix.shape)




	def forward(self):
		image = self.renderer(self.vertices, self.faces, mode='silhouettes')
		# loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)  #using MSE loss
		loss =10* self.loss_function(image, self.image_ref[None, :, :])   #using huber loss
		return loss


def make_gif(filename,filename_ref):
	count=0 # store the number of image files in the tmp directory
	for item in glob.glob('/tmp/_tmp_*.png'):
		count=count+1

	i=0
	with imageio.get_writer(filename, mode='I') as writer:
		for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
			writer.append_data(imread(filename))
			i=i+1
			if i==count: # only store the final image
				final_image = imread(filename)
			os.remove(filename)
	writer.close()

	imsave(os.path.join(data_dir,"example4_result_image.png"), final_image)

	# save optimizatin images to folder
	print(os.path.join(data_dir,Path(filename_ref).parent, '../optimized_outputs',filename_ref.split('/')[-1]))
	imsave(os.path.join(data_dir,Path(filename_ref).parent, '../optimized_outputs',filename_ref.split('/')[-1]), final_image)

def make_reference_image(filename_obj, filename_ref, cam_ref_pose):

	model = Model(filename_obj, None, cam_ref_pose)
	model.cuda()

	model.renderer.eye = nn.Parameter(torch.from_numpy(np.array([1.84, 1.079, 0.3687*2],dtype=np.float32)))

	images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
	image = images.detach().cpu().numpy()[0].transpose(1,2,0)
	imsave(filename_ref, image)

	save_cam_pose_to_file(model)



def save_cam_pose_to_file(model):
	r_quat = model.renderer.R.cpu().data.numpy() # in the form [cos(t/2), xsin(t/2), ysin(t/2), zsin(t/2)]
	r_obj = R.from_quat(np.array([r_quat[1],r_quat[2],r_quat[3],r_quat[0]],dtype='float32')) #converting to [xsin(t/2),ysin(t/2),zsin(t/2),cos(t/2)]
	rot_matrix = r_obj.as_dcm()
	t = model.renderer.t.cpu().data.numpy()
	t = t[None,:]

	RT = np.concatenate((rot_matrix,t.T),axis=1)
	np.save(os.path.join(data_dir, 'poses/RT_matrix.npy'), RT) # saving the current pose to a file to be read in the next interation



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'models/meshes/realsense_chair.obj'))  
	parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'realsense/realsense_bw_mask_ref.jpg'))
	parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result_real.gif'))
	parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
	parser.add_argument('-g', '--gpu', type=int, default=0)
	parser.add_argument('-rp', '--cam_ref_pose', type=str, default=os.path.join(data_dir, 'poses/RT_matrix_init_300.npy'))
	parser.add_argument('-om', '--optimization_method', type=int, default=1)
	args = parser.parse_args()


	filename_ref = os.path.join(data_dir,args.filename_ref)
	if args.make_reference_image:
		make_reference_image(args.filename_obj, filename_ref, args.cam_ref_pose)


	model = Model(args.filename_obj, filename_ref, args.cam_ref_pose)
	model.cuda()

	loss_thresh=80
	loss_thresh_max=200
	min_loss = 10000

	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	loop = tqdm.tqdm(range(200))
	for i in loop:

		if args.optimization_method == 1:  #optimization method=1 implies this the first frame of the sequence
			if i < 30:
				model.renderer.R.requires_grad=False
			else:
				model.renderer.R.requires_grad=True

		optimizer.zero_grad()
		loss = model()
		loss.backward()
		optimizer.step()
		images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
		image = images.detach().cpu().numpy()[0].transpose(1,2,0)
		imsave('/tmp/_tmp_%04d.png' % i, image)
		loop.set_description('Optimizing (loss %.4f)' % loss.data)

		if loss.item() < loss_thresh:
			break

	if loss.item() < loss_thresh:
		loss_thresh = loss_thresh - loss_thresh/4 # dynamically changing loss to account for change in views
	elif loss.item() < loss_thresh_max and loss.item() > loss_thresh:
		loss_thresh = loss_thresh + loss_thresh/4

	make_gif(args.filename_output,args.filename_ref)
	save_cam_pose_to_file(model) # save the output pose of the renderer to file to be used for next iteration in batch run

if __name__ == '__main__':
	main()
