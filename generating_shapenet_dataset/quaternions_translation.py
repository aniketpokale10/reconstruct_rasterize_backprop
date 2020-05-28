# This code is used to convert the translation in (x,y,z) form and the quaternion angles to render the chair object in Blender.
# It takes the input file with the parameters - Azimuthal, Elevation, Inplace rotation, radius

import os
import sys
import math
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

def camPosToQuaternion(cx, cy, cz):
	camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
	cx = cx / camDist
	cy = cy / camDist
	cz = cz / camDist
	axis = (-cz, 0, cx)
	angle = math.acos(cy)
	a = math.sqrt(2) / 2
	b = math.sqrt(2) / 2
	w1 = axis[0]
	w2 = axis[1]
	w3 = axis[2]
	c = math.cos(angle / 2)
	d = math.sin(angle / 2)
	q1 = a * c - b * d * w1
	q2 = b * c + a * d * w1
	q3 = a * d * w2 + b * d * w3
	q4 = -b * d * w2 + a * d * w3
	return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
	c1 = math.cos(yaw / 2.0)
	c2 = math.cos(pitch / 2.0)
	c3 = math.cos(roll / 2.0)    
	s1 = math.sin(yaw / 2.0)
	s2 = math.sin(pitch / 2.0)
	s3 = math.sin(roll / 2.0)    
	q1 = c1 * c2 * c3 + s1 * s2 * s3
	q2 = c1 * c2 * s3 - s1 * s2 * c3
	q3 = c1 * s2 * c3 + s1 * c2 * s3
	q4 = s1 * c2 * c3 - c1 * s2 * s3
	return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
	q1a = 0
	q1b = 0
	q1c = math.sqrt(2) / 2
	q1d = math.sqrt(2) / 2
	camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
	cx = cx / camDist
	cy = cy / camDist
	cz = cz / camDist    
	t = math.sqrt(cx * cx + cy * cy) 
	tx = cx / t
	ty = cy / t
	yaw = math.acos(ty)
	if tx > 0:
		yaw = 2 * math.pi - yaw
	pitch = 0
	tmp = min(max(tx*cx + ty*cy, -1),1)
	#roll = math.acos(tx * cx + ty * cy)
	roll = math.acos(tmp)
	if cz < 0:
		roll = -roll    
	q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
	q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
	q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
	q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
	q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
	return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
	theta = theta / 180.0 * math.pi
	camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
	cx = -cx / camDist
	cy = -cy / camDist
	cz = -cz / camDist
	q1 = math.cos(theta * 0.5)
	q2 = -cx * math.sin(theta * 0.5)
	q3 = -cy * math.sin(theta * 0.5)
	q4 = -cz * math.sin(theta * 0.5)
	return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
	a = qx[0]
	b = qx[1]
	c = qx[2]
	d = qx[3]
	e = qy[0]
	f = qy[1]
	g = qy[2]
	h = qy[3]
	q1 = a * e - b * f - c * g - d * h
	q2 = a * f + b * e + c * h - d * g
	q3 = a * g - b * h + c * e + d * f
	q4 = a * h + b * g - c * f + d * e    
	return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
	phi = float(elevation_deg) / 180 * math.pi
	theta = float(azimuth_deg) / 180 * math.pi
	x = (dist * math.cos(theta) * math.cos(phi))
	y = (dist * math.sin(theta) * math.cos(phi))
	z = (dist * math.sin(phi))
	return (x, y, z)




shape_view_params_file = sys.argv[-1]

with open (os.path.join(BASE_DIR, shape_view_params_file), 'r') as fp:
	view_params = fp.read()

azimuth_deg = 360 - float(view_params.split(' ')[0])
elevation_deg = float(view_params.split(' ')[1])
theta_deg = float(view_params.split(' ')[2])
rho = float(view_params.split(' ')[3])

cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
q1 = camPosToQuaternion(cx, cy, cz)
q2 = camRotQuaternion(cx, cy, cz, theta_deg)
q = quaternionProduct(q2, q1)

print (cx)
print (cy)    
print (cz)
print (q[0])
print (q[1])
print (q[2])
print (q[3])
