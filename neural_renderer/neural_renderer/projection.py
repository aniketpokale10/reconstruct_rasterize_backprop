from __future__ import division
# from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
import torchgeometry as tgm



def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3

    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)



def eul2rotmat(rot_euler):
    phi = rot_euler[0][0][0]
    theta = rot_euler[0][0][1]
    psy = rot_euler[0][0][2]

    print(phi)


    R = torch.tensor([[torch.cos(phi)*torch.cos(theta)*torch.cos(psy) - torch.sin(phi)*torch.sin(psy),  -1*torch.cos(phi)*torch.cos(theta)*torch.sin(psy) - torch.sin(phi)*torch.cos(psy), torch.cos(phi)*torch.sin(theta)],
    [torch.sin(phi)*torch.cos(theta)*torch.cos(psy) + torch.cos(phi)*torch.sin(psy),  -1*torch.sin(phi)*torch.cos(theta)*torch.sin(psy) + torch.cos(phi)*torch.cos(psy), torch.sin(phi)*torch.sin(theta)],
    [-1*torch.sin(theta)*torch.cos(psy),  torch.sin(theta)*torch.sin(psy),  torch.cos(theta)]], device='cuda')

    print(R)
    print(R.shape)
    return R


def projection(vertices, K, rot_quat, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'

    batch_size = vertices.shape[0]
    device = vertices.device


    if isinstance(K,np.ndarray):
        K = torch.from_numpy(K).to(device)
    if isinstance(rot_quat,np.ndarray):
        rot_quat = torch.from_numpy(rot_quat).to(device)
    if isinstance(t,np.ndarray):
        t = torch.from_numpy(t).to(device)

    
    if K.ndimension() == 2:
        K = K[None, :].repeat(batch_size,1,1)
    if rot_quat.ndimension() == 1:
        rot_quat = rot_quat[None, :].repeat(batch_size,1,1)
    # rot_matrix = rot_matrix[None, :].repeat(batch_size,1,1)
    if t.ndimension() == 1:
        t = t[None, :].repeat(batch_size,1)


    np.savetxt('/home/aniket/neural_renderer/examples/test_mesh_points_before_transformation.txt',(vertices.detach().cpu().numpy()[0,:,:]) \
     ,delimiter=',') 


#For Quaternion notation---------
    # for i in range(vertices.shape[0]): #vertices.shape[0] is the batch_size    
    #     print(vertices.shape[1])
    #     print(rot_quat[i].shape)
    #     temp_rot = rot_quat[i].repeat(vertices.shape[1],1)
    #     vertices[i] = qrot(temp_rot.float(), vertices[i].float())

    # vertices = vertices + t  # instead of P*x we compute x'*P'

#--------------------------------


#For Euler notation----------------

    rot_angle_axis = tgm.quaternion_to_angle_axis(rot_quat[0])  # batch size is one, hence passing rot_quat[0]
    rot_matrix = tgm.angle_axis_to_rotation_matrix(rot_angle_axis)[:,:3,:3]
    rot_matrix = rot_matrix.to(device)

    rot_matrix = rot_matrix.repeat(batch_size,1,1)

    vertices = torch.matmul(vertices, rot_matrix.transpose(2,1)) + t  # instead of P*x we compute x'*P'
    

    # vertices = torch.matmul(vertices - t, rot_matrix)  # instead of P*x we compute x'*P'
    
    vertices_to_print = vertices.detach().cpu().numpy()[0,:,:]
    np.savetxt('/home/aniket/neural_renderer/examples/test_mesh_points.txt',(vertices_to_print),delimiter=',') 

    vertices = torch.matmul(vertices, K.transpose(1,2))  # instead of P*x we compute x'*P'
    
    np.savetxt('/home/aniket/neural_renderer/examples/test.txt',(vertices.detach().cpu().numpy()[0,:,:]),delimiter=',') 

    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    # r = torch.sqrt(x_ ** 2 + y_ ** 2)
    # x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    # y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    vertices = torch.stack([x_, y_, torch.ones_like(z)], dim=-1)


    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    projections = torch.stack([u,v,z],dim=-1)


    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)

    return vertices
