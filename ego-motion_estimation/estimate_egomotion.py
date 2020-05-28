import os
import numpy as np
import torch
import neural_renderer as nr
import yaml

# read reference frame image names to be passed in successive iterations


def save_all_poses_to_file(i):
	RT_matrix = np.load(os.path.join(pose_path, "RT_matrix_combined.npy"))
	R = RT_matrix[:,:3]
	t = RT_matrix[:,3]
	# saving all the poses to a file
	f = open(os.path.join(pose_path, 'all_poses.txt'), 'a+')
	f.write("%d\n" % (i+1))
	for i in range(3):
		f.write("%5.3f %5.3f %5.3f\n" % (R[i][0], R[i][1], R[i][2]))
	f.write("%5.3f %5.3f %5.3f\n" % (t[0], t[1], t[2]))
	f.write("\n")
	f.close()



def get_initialization(frame,bbox,h,az,el,chairId):
	xmid = bbox[0][0]+bbox[0][2]/2; ymid=bbox[0][1]+bbox[0][3];
	b = np.array([[xmid,ymid,1]],dtype='float32')
	K = np.array([[245.0, 0.0, 112.0],[0.0, 245.0, 112.0],[0.0, 0.0, 1.0]],dtype='float32')
	n=np.array([[0,-1,0]],dtype='float32')
	X_bottom_gt = -1*(h*np.dot(np.linalg.inv(K),b.T))/(np.dot(n, np.dot(np.linalg.inv(K),b.T)))
	d = np.linalg.norm(X_bottom_gt)

	az = az*np.pi/180
	el = el*np.pi/180
	x = d*np.sin(az)*np.cos(el);
	y = -1*d*np.cos(az)*np.cos(el);
	z = d*np.sin(el); 
	c = np.array([[x,y,z]],dtype='float32')
	at = np.array([[0,0,h]],dtype='float32')
	up = np.array([[0,0,1]],dtype='float32')
	z_axis = at-c; 
	z_axis = z_axis/np.linalg.norm(z_axis);
	x_axis = np.cross(z_axis,up);
	x_axis = x_axis/np.linalg.norm(x_axis);
	y_axis = np.cross(z_axis,x_axis);
	y_axis = y_axis/np.linalg.norm(y_axis);

	R = np.vstack((x_axis,y_axis,z_axis))
	R = R.T

	print(R)
	print(X_bottom_gt)
	print(c)

	Rt = np.hstack((R,np.dot(-1*R,X_bottom_gt)))  #np.hstack((R,c.T))
	T = np.vstack((Rt, np.array([[0,0,0,1]],dtype='float32')))

	T = np.linalg.inv(T) 

	np.save(os.path.join(pose_path, "RT_matrix_init_"+str(chairId)+"_"+frame+".npy"),T[0:3,:])



class chair:
	# input arguments: Id of each chair, obj filename, reference image, azimuth and elevation angles for when the object is seen for the first time 
	def __init__(self,chairId=-1,filename_obj=None,ref_image=None,bbox=None,az=None,el=None,starting_frame=None):
		self.filename_obj = filename_obj
		self.chairId = chairId
		self.frameList = []
		self.seen = False
		if filename_obj:
			vertices, faces = nr.load_obj(filename_obj,normalization=None)
			self.vertices = vertices.detach().cpu().numpy()
			self.faces = faces.detach().cpu().numpy()
		else:
			self.vertices = np.empty((0,3),dtype='float32')
			self.faces=np.empty((0,3),dtype='float32')
		if ref_image is not None:
			self.ref_image = ref_image
		if az and el:
			self.az = az
			self.el = el
		if bbox is not None:
			self.bbox=bbox
		self.RT_matrix = None
		if starting_frame is not None:
			self.starting_frame = starting_frame


class frame:
	def __init__(self,id,R=None,t=None):
		if id: self.id = id
		if R and t:
			self.R = R
			self.t = t
			self.initialized = True
		else: self.initialized = False


def combine_objects_and_save(chair, currentFrame): # combine the objects seen in one frame and save as a single object
# need to transform the new object with respect to the object at origin and store their locations in vertices
	RT_matrix = np.load(os.path.join(pose_path, "RT_matrix_"+str(chair.chairId)+".npy"))
	R = RT_matrix[:,:3]
	t = RT_matrix[:,3]
	
	if chair.chairId == originChairId:
		RcurrentFrame = R
		tcurrentFrame = t
	else:
		RcurrentFrame = frameListGlobal[currentFrame].R
		tcurrentFrame = frameListGlobal[currentFrame].t

	R = np.matmul(RcurrentFrame.transpose(1,0),R) 
	t = np.matmul(RcurrentFrame.transpose(1,0),t) - np.matmul(RcurrentFrame.transpose(1,0),tcurrentFrame)
	
	t = np.array([t],dtype='float32')
	RT = np.concatenate((R,t.T),axis=1)	
	np.save(os.path.join(pose_path, 'chair%d_pose.npy'%chair.chairId), RT)


	vertices_new = np.matmul(chair.vertices, R.transpose(1,0)) + t

	# the combined object contains all the chairs visible in the current frame now
	combined_object.vertices = np.append(combined_object.vertices, vertices_new, axis=0)
	
	if combined_object.faces.shape != (0,3): new_starting_face_index = np.amax(combined_object.faces) + 1
	else: new_starting_face_index = 0
	combined_object.faces = np.append(combined_object.faces, chair.faces + new_starting_face_index, axis=0)

	nr.save_obj(os.path.join(data_path, "combined_object.obj"), combined_object.vertices,combined_object.faces)

	if combined_object.RT_matrix is None:
		combined_object.RT_matrix = RT_matrix # copy the origin object matrix, will occur only once

	print('inside combine_objects_and_save')
	print(combined_object.RT_matrix)

	if chair.chairId == originChairId:
		np.save(os.path.join(pose_path, 'RT_matrix_combined.npy'),combined_object.RT_matrix)




def solveFramePose(frameId,ref_image,initialization):
	cmd = "python example4.py -rp " + os.path.join(pose_path, "RT_matrix_combined.npy") + " -ir " + ref_image + " -om 1 -io "+combined_object.filename_obj+ \
	" -or example4_result_combined.gif"
	print(cmd)
	os.system(cmd)
	os.system("mv " + os.path.join(pose_path, "RT_matrix.npy") + " " + os.path.join(pose_path, "RT_matrix_combined.npy"))

	# frameListGlobal[i].initialized = True

	RT_matrix = np.load(os.path.join(pose_path, "RT_matrix_combined.npy"))
	R = RT_matrix[:,:3]
	t = RT_matrix[:,3]
	
	frameListGlobal[i].R = R
	frameListGlobal[i].t = t

	if initialization == False: # if this is an initialization for a frame, dont append the frame pose to the output file
		save_all_poses_to_file(i)
	elif initialization == True:
		print('inside solveFramePose initialization: RT_matrix:')
		print(RT_matrix)




# main code

# Note: the chair indices start at 0, the frame indices start at 1

# data to be added for testing-----------------------------------------------------------------
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

# all the frame ids = frame number - 1
total_frames= cfg["total_frames"]
h= cfg["height"]
originChairId= cfg["originChairId"] # index of the chair which is at the origin
starting_frame = cfg["starting_frame"]
data_path = cfg["data_path"]
model_path = cfg["model_path"]
pose_path = cfg["pose_path"]

chairList = []; # to store chair datastructure in this list



# dataset data--------------------------------------------------------------
dataset = cfg["dataset"]

chairs_dict = cfg["chairs"]
for i, chair_id in enumerate(chairs_dict):
	bbox = np.array([chairs_dict[chair_id]["bbox"]],dtype='float32')
	az = chairs_dict[chair_id]["az"]
	el = chairs_dict[chair_id]["el"]
	chair_init_frame = chairs_dict[chair_id]["chair_init_frame"]
	ref_image = os.path.join(data_path, dataset, '/for_init/', chairs_dict[chair_id]["ref_image"])
	chairList.append(chair(i,os.path.join(model_path, chairs_dict[chair_id]["obj_file"]),ref_image,bbox,az,el,chair_init_frame)) # initializing with chairIds
# ---------------------------------------------------------------------------


# Added frame visibility
for i in range(chairList[0].starting_frame,total_frames):
	chairList[0].frameList.append(i)
	chairList[1].frameList.append(i)



# intitialize FrameListGlobal
frameListGlobal = []
for i in range(0,total_frames): frameListGlobal.append(frame(i))



f = open( os.path.join(pose_path, 'all_poses.txt'), 'a+')
f.write('%d\n' % (total_frames - starting_frame)) # starting_frame+1
f.close()



# datastructure to store the combined object
combined_object = chair(-1)
combined_object.filename_obj = os.path.join(data_path, 'combined_object.obj')




#For testing------------
# cmd = "python example4.py -rp poses/RT_matrix_init_"+str(starting_frame+1)+".npy -mr 1 -or data/example4_result_"+str(starting_frame+1) + ".gif -om 1"
# print(cmd)
# os.system(cmd)
#-----------------------



for i in range(starting_frame,total_frames):   #(total_frames):

	ref_image =  os.path.join(data_path, dataset, '/masks_gt/%04d' % (i+1) + '.png')

	chairs_visible= False # when there is no chair visible in the frame

	for chair in chairList:

		if i in chair.frameList:
			
			chairs_visible = True

			if chair.seen == False: # should read the initial pose from pose regressor for the first iteration

				print('Initialization:')

				# the frame has to be initialized before getting the pose of the current chair wrt it, since in combine_objects_and_save() we need
				# the current frames pose wrt the origin/origin_chair, this is for all the chairs other than the origin chair
				if chair.chairId != originChairId:
					solveFramePose(i, os.path.join(data_path, dataset, '/for_init/for_initialization_frame%d' % (i+1) + '_chair'+str(chair.chairId+1)+'.png'),True)


				get_initialization(str(i+1),chair.bbox,h,chair.az,chair.el,chair.chairId)
				cmd = "python example4.py -rp " + os.path.join("RT_matrix_init_"+str(chair.chairId)+"_"+str(chair.starting_frame+1)+".npy") + " -ir " + chair.ref_image \
				 + " -or " os.path.join(data_path, "example4_result_"+str(chair.chairId)+"_"+str(chair.starting_frame+1)+".gif) -om 1 -io "+chair.filename_obj
				print(cmd)
				os.system(cmd)
				os.system("mv " + os.path.join(pose_path,"RT_matrix.npy") + " " + os.path.join(pose_path, "RT_matrix_"+str(chair.chairId)+".npy"))

				# save_all_poses_to_file(i,chair.chairId)
				chair.seen = True
				# After we have seen more than one chair in the same frame and initialized them for the first time, we need to store all the chairs seen
				# in the current frame as a single object and then load this new combined object in example4 for running it as multiobject
				combine_objects_and_save(chair,i)
				# centralize_combined_object_vertices()


				print('frame: %d' % i)
				print('chair: %d' % chair.chairId)


	if chairs_visible == True:	
		solveFramePose(i,ref_image,False)

f.close()
