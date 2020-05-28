import numpy as np
import sys
import os
import argparse
import associate

parser = argparse.ArgumentParser(description='''
This script removes the frames with sudden jerks ie outliers in the NR trajectory.. 
''')
parser.add_argument('file_name', help='file_name (format: timestamp tx ty tz qx qy qz qw)')
args = parser.parse_args()

file_contents = associate.read_file_list(args.file_name)
f = open('nr_quat.txt',"w")

N = len(file_contents)
prev = file_contents[0]
d = []
f.write(str(0) + ' ' + prev[0] + ' ' + prev[1] + ' ' + prev[2] + ' ' + prev[3] + ' ' + prev[4] + ' ' + prev[5] + ' ' + prev[6] + '\n')
for i in range(1,N):
	curr = file_contents[i]
	x_prev = float(prev[0])
	y_prev = float(prev[1])
	z_prev = float(prev[2])
	x_curr = float(curr[0])
	y_curr = float(curr[1])
	z_curr = float(curr[2])
	dist = (x_curr - x_prev)*(x_curr - x_prev) + (y_curr - y_prev)*(y_curr - y_prev) + (z_curr - z_prev)*(z_curr - z_prev)
	if (dist < 0.01):
		f.write(str(i) + ' ' + curr[0] + ' ' + curr[1] + ' ' + curr[2] + ' ' + curr[3] + ' ' + curr[4] + ' ' + curr[5] + ' ' + curr[6] + '\n')
	else:
		print (i)
	prev = file_contents[i]
	d.append(dist)
	# print (dist)

print (np.sort(np.asarray(d)))
f.close()