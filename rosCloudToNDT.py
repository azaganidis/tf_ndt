import numpy as np
import tensorflow as tf
import argparse
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import time
from ndt import *
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

def NDT_to_MSG(C,M):
    npoints= M.shape[0]
    C=C.reshape([-1,9])
    data=np.concatenate((C,M),axis=1)
    msg_data=Float32MultiArray(data=data.flatten())
    return msg_data

class PclToNDT():
    def __init__(self, args):
        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options ))
            self.inCloud=tf.placeholder(tf.float32, shape=(None, 4))
            self.ndt=NDT(self.inCloud, 0.75)
            #with tf.device('/device:GPU:0'):
            self.sess.run(tf.global_variables_initializer())
        rospy.Subscriber("/semantic", PointCloud2, self.callback)
        self.pD = rospy.Publisher('NDTs',Float32MultiArray,queue_size=1)
        self.publisher  = rospy.Publisher('semantic', PointCloud2, queue_size=10)
        print "READY"

    def PublishCloud(self, data):
        fields=[ pc2.PointField('x',0,pc2.PointField.FLOAT32,1),
                pc2.PointField('y',4,pc2.PointField.FLOAT32,1),
                pc2.PointField('z',8,pc2.PointField.FLOAT32,1),
                pc2.PointField('intensity',12,pc2.PointField.FLOAT32,1)]
        a= PointCloud2
        header=Header()
        header.stamp = rospy.Time.now()
        header.frame_id="velodyne"
        msP = pc2.create_cloud(header,fields, data)  
        self.publisher.publish(msP)

    def callback(self, msg):
        data_out = pc2.read_points(msg, skip_nans=True)
        a=np.array(list(data_out))
        a=a[:,:4]
        #a[:,3]=a[:,3]/255
        feed = {self.inCloud:a}
        M,C=self.sess.run([self.ndt.Means, self.ndt.Covariances], feed)
        self.pD.publish(NDT_to_MSG(C,M))
        #self.PublishCloud(result)
    def main(self):
        rospy.spin()


def main():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
        rospy.init_node('semantics', anonymous=True)
	pNDT= PclToNDT(args)
        pNDT.main()


if __name__ == '__main__':
	main()
	
