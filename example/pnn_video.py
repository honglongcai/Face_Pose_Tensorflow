import tensorflow as tf
import numpy as np
import cv2
import dlib

sess=tf.Session()


x=tf.placeholder(tf.float32,shape=[None,40*40])

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv1=weight_variable([3,3,1,15])
b_conv1=bias_variable([15])

x_image=tf.reshape(x,[-1,40,40,1])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([3,3,15,30])
b_conv2=bias_variable([30])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#h_pool2=max_pool_2x2(h_conv2)
h_pool2=h_conv2

W_fc1=weight_variable([20*20*30,512])
b_fc1=bias_variable([512])

h_pool2_flat=tf.reshape(h_pool2,[-1,20*20*30])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([512,9])
b_fc2=bias_variable([9])

y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

saver=tf.train.Saver()
saver.restore(sess,'pnn1530_2/hehe.ckpt')

ang=tf.argmax(y_conv,1)

angle=[-60,-45,-30,-15,0,15,30,45,60]

detector=dlib.get_frontal_face_detector()
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('./honglong1530_3.mp4',fourcc,15.0,(1280,720),isColor=True)
font=cv2.FONT_HERSHEY_SIMPLEX

cap=cv2.VideoCapture('./honglong2.mov')
while cap.isOpened():
	ret,imgcv=cap.read()
	if ret==False:
		print('out')
		break
	ds=detector(imgcv,1)
	if len(ds)==0:
		print('None')
		continue
	for i,d in enumerate(ds):
		l,t,r,b=d.left(),d.top(),d.right(),d.bottom()
		cv2.rectangle(imgcv,(l,t),(r,b),(255,0,0),2)
		gimg=cv2.cvtColor(imgcv[t:b,l:r,:],cv2.COLOR_BGR2GRAY)
		gimg1=cv2.resize(gimg,(40,40))
		gimg2=gimg1.reshape(1,40*40)
		an=sess.run(ang,feed_dict={x:gimg2,keep_prob:1.})
		an1=str(angle[int(an)])
		cv2.putText(imgcv,an1,(r,b),font,1.2,(255,0,0),2)
		out.write(imgcv)
#cv2.destroyAllWindows()
cap.release()
out.release()	



