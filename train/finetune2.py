import numpy as np
import cv2
import tensorflow as tf
import csv
import glob

'''
def fp(f):
	if f < -50.:
		p=0
	elif f < -35.:
		p=1
	elif f < -20.:
		p=2
	elif f < -10.:
		p=3
	elif f < 10.:
		p=4
	elif f < 20.:
		p=5
	elif f < 35.:
		p=6
	elif f < 50.:
		p=7
	else:
		p=8
	return p  

tit='net4040/'
f=open('face_pose_net7876.csv')
rows=csv.reader(f)
X=np.zeros((15000,40*40),dtype='uint8')
Y=np.zeros((15000,9))
i=0
num=np.zeros(9)

for row in rows:
	rsp=row[0].split('/')
	imgpath=tit+rsp[1]
	print(imgpath)
	img=cv2.imread(imgpath,0)
	if img is None:
		continue
	X[i]=img.reshape(40*40,)
	f=fp(float(row[1]))
	Y[i,f]=1
	i+=1
	num[f]+=1
print(num)

X_train=X[0:11000]
Y_train=Y[0:11000]
X_test=X[11000:12335]
Y_test=Y[11000:12335]
'''
nt=6500
idr1=np.arange(nt)
pnum=0
directories=glob.glob('../data/images/*')
print(directories)
X=np.zeros((10000,40*40),dtype='uint8')
Y=np.zeros((10000,9))
for directory in directories:
	images=glob.glob(directory+'/*')
	for image in images:
		c=int(directory.split('/')[-1])
		X[pnum,:]=cv2.imread(image,0).reshape(40*40,)
		Y[pnum,c]=1.
		pnum+=1
print(pnum)
idr=np.arange(pnum)
np.random.shuffle(idr)
X_train=X[idr[:nt],:]
X_test=X[idr[nt:],:]
Y_train=Y[idr[:nt],:]
Y_test=Y[idr[nt:],:]
print(idr)
print(X_train.shape,Y_train.shape)

x=tf.placeholder(tf.float32,shape=[None,40*40])
y_=tf.placeholder(tf.float32,shape=[None,9])

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

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = modeling
	for i in range(10000):
		np.random.shuffle(idr1)
		batch=idr1[:64]
		if i%1000==0:
			train_accuracy=accuracy.eval(feed_dict={x:X_train[batch],y_:Y_train[batch],keep_prob:1.0})
			ty=sess.run(tf.argmax(Y_train[batch],1))
			py=sess.run(tf.argmax(y_conv,1),feed_dict={x:X_train[batch],keep_prob:1.0})
			#for ij in range(50):
			#	print(ty[ij],py[ij])
			print('step {}, ta{}'.format(i,train_accuracy))
		train_step.run(feed_dict={x:X_train[batch],y_:Y_train[batch],keep_prob:0.5})
	print('test a {}'.format(accuracy.eval(feed_dict={x:X_test,y_:Y_test,keep_prob:1.0})))
	saver.save(sess,'models/test.ckpt')


