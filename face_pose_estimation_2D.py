import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import face_alignment
from skimage import io
import glob
import os
import csv
import time
import cv2

# face_num: face number in a picture
face_num=np.array([0,0,0,0,0])

# fail_num: the number of no face detected pictures
fail_num=0


f=open('fail_detect','w')
face_file=open('pose_estimation.csv','w')
headers=['Path','Yaw','Pitch','Roll','v0','v1','v2','v3','v4','v5','v6','v7','v8','FN']
f_csv=csv.DictWriter(face_file,headers)
f_csv.writeheader()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)

startT=time.time()
directories=glob.glob('lfw/*')
directories.sort()
for directory in directories:
	images=glob.glob(directory + '/*')
	for image in images:
		imgpath=image.split('/')
		input = io.imread(image)
		preds = fa.get_landmarks(input)
		if preds is None:
			fail_num=fail_num+1
			print(image+' no face detected.',fail_num)
			f.write(image+' no face detected. {}\n'.format(fail_num))
			continue
		nf=len(preds)
		face_num[nf-1]+=1
		preds=preds[0]
		M=np.array([[-65.5,-5,-20],[65.5,-5,-20],[-77.5,-6,-100],[77.5,-6,-100],[0,-48,21],[0,-75,10],[0,-133,0]])
		x=preds[:,0]
		y=-preds[:,1]
		s1,s2=x[27],y[27]
		x=x-s1
		y=y-s2
		point_num=np.array([36,45,0,16,30,51,8])
		xnew=x[point_num]
		ynew=y[point_num]
		MM=inv(np.dot(M.transpose(),M)).dot(M.transpose())
		a1=MM.dot(xnew)
		a2=MM.dot(ynew)
		A=np.vstack((a1,a2))
		U,s,V=np.linalg.svd(A,full_matrices=True)
		for i in range(2):
			if V[i,i]<0:
				V[i,:]=-V[i,:]
				U[:,i]=-U[:,i]
		if V[2,2]<0:
			V[2,:]=-V[2,:]
		U_full=np.array([[U[0,0],U[0,1],0],[U[1,0],U[1,1],0],[0,0,1]])
		new_V=U_full.dot(V)
		vx=new_V[0,2]
		vy=new_V[1,2]
		vz=new_V[2,2]
		rx=new_V[0,1]
		vtheta=np.arcsin(vy)*180/np.pi
		xzsq=np.sqrt(vx*vx+vz*vz)
		htheta=np.arcsin(vx/xzsq)*180/np.pi
		rtheta=np.arcsin(rx)*180/np.pi
		strv=str(np.int(vtheta))
		strh=str(np.int(htheta))
		strr=str(np.int(rtheta))
		dic=[{'Path':image,'Yaw':htheta,'Pitch':vtheta,'Roll':rtheta,'v0':new_V[0,0],'v1':new_V[0,1],'v2':new_V[0,2],'v3':new_V[1,0],'v4':new_V[1,1],'v5':new_V[1,2],'v6':new_V[2,0],'v7':new_V[2,1],'v8':new_V[2,2],'FN':nf}]
		f_csv.writerows(dic)
		d=np.arange(5)
		fig=plt.figure()
		my_dpi=fig.get_dpi()
		fig.set_size_inches(2.5,2.5)
		ax=fig.gca(projection='3d')
		ax.view_init(0,0)
		ax.plot(new_V[2,0]*d,new_V[0,0]*d,new_V[1,0]*d)
		ax.plot(new_V[2,1]*d,new_V[0,1]*d,new_V[1,1]*d)
		ax.plot(new_V[2,2]*d,new_V[0,2]*d,new_V[1,2]*d)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_title(strh+', '+strv+', '+strr)
		pat1='pose/'+imgpath[2]
		fig.savefig(pat1,dpi=100)
		img1=cv2.imread(pat1)
		img2=cv2.imread(image)
		img=np.zeros(250*500*3,dtype='uint8').reshape(250,500,3)
		img[0:250,0:250,:]=img1
		img[0:250,250:500,:]=img2
		pat2='face_pose/'+imgpath[2]
		cv2.imwrite(pat2,img)
		plt.close(fig)
		cv2.destroyAllWindows()
		if sum(face_num)%100==0:
			endT=time.time()
			print(sum(face_num)//100,endT-startT,imgpath[2])
print(face_num)
#print(image,htheta,end=' ')
#print(vtheta,rtheta)
# new_M=new_V.dot(M.T)
# mx=M[:,0]
# my=M[:,1]
# mz=M[:,2]
# nmx=new_M[0,:]
# nmy=new_M[1,:]
# nmz=new_M[2,:]
# fig=plt.figure()
# ax=fig.gca(projection='3d')
# ax.scatter(nmx,nmy,nmz)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
