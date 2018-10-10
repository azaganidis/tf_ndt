import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
from tensorflow.python.client import timeline

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from ndt import *
from registration import RegistrationNDT
def split_semantic(incloud, val):
    ValueEqual = tf.equal(incloud[:,3], val)
    NumPoints=tf.count_nonzero(ValueEqual)
    with tf.control_dependencies([NumPoints]):
        incloud=tf.cond(tf.greater(NumPoints,0), lambda:incloud, lambda: tf.constant([[0.0,0.0,0.0,.0]]))
        ValueEqual=tf.cond(tf.greater(NumPoints,0), lambda:ValueEqual, lambda: tf.constant([True]))
        return tf.boolean_mask(incloud, ValueEqual) 

class RosSemantics():
    def __init__(self, args):
        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            #self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            gpu_options = tf.GPUOptions(allow_growth=True)
            #self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
            #self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options ))
            self.sess=tf.Session()
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

            self.inCloud=tf.placeholder(tf.float32, shape=(None, 4))
            self.static=tf.placeholder(tf.float32, shape=(None,4))
            self.registration=RegistrationNDT()
            '''
            for i in range(15):
                filt=split_semantic(self.inCloud,i)
                filt_static=split_semantic(self.static,i)
                ndt1=NDT(filt, 1.0)
                ndt2=NDT(filt_static, 1.0)
                self.registration.add_pair(ndt1, ndt2)
            '''
            self.registration.add_semantic(self.static,self.inCloud)
            #ndt1=NDT(self.inCloud, 4.0)
            #ndt2=NDT(self.static, 4.0)
            #self.registration.add_pair(ndt1, ndt2)
            self.previusCloud=np.empty((0,4))
            #with tf.device('/device:GPU:0'):
            self.registration.get_train_op()
            self.sess.run(tf.global_variables_initializer())
        rospy.Subscriber("/semantic", PointCloud2, self.callback)
        self.publisher  = rospy.Publisher('semantic', PointCloud2, queue_size=10)
        self.count=False
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
        train_writer = tf.summary.FileWriter('/home/anestis/workspace/ndt_gpu/timing', self.sess.graph)

        if self.count :
            start = time.time()
            feed = {self.inCloud:a, self.static:self.previusCloud}
            start_time=time.time()
            self.sess.run(self.registration.reset_transform)
            #T, _, transform, loss=self.sess.run([self.registration.Transform, self.registration.train_op, self.registration.PARAMS, self.registration.loss], feed, options=self.options, run_metadata=self.run_metadata)
            for step in range(50):
                #_, transform, loss=self.sess.run([self.registration.train_op, self.registration.PARAMS, self.registration.loss], feed, options=self.options, run_metadata=self.run_metadata)
                #loss=self.sess.run([self.registration.loss], feed, options=self.options, run_metadata=self.run_metadata)
                #T, _, transform, loss=self.sess.run([self.registration.Transform, self.registration.train_op, self.registration.PARAMS, self.registration.loss], feed, options=self.options)
                #g,G,h,H,T, _, transform, loss=self.sess.run([self.registration.gradient, self.registration.G, self.registration.hessian, self.registration.H,self.registration.Transform, self.registration.train_op, self.registration.PARAMS, self.registration.loss], feed)
                T, _, transform, loss=self.sess.run([self.registration.Transform, self.registration.train_op, self.registration.PARAMS, self.registration.loss], feed,options=self.options, run_metadata=self.run_metadata)
                print loss, transform[3:]
                #np.set_printoptions(precision=1, suppress=True)
                #print h
                #print H
                train_writer.add_run_metadata(self.run_metadata, 'iter%d' %step)
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_01.json', 'w') as f:
                f.write(chrome_trace)
            end_time=time.time()
            print "TIME", end_time-start_time
        self.previusCloud=a
        self.count=True
        #self.PublishCloud(result)
    def main(self):
        rospy.spin()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=8, help='size of mini batch')
	parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters')
        parser.add_argument('--model', default='my_auto', help='Model name [default: my]')
	#parser.add_argument('--input_dim', type=int, default=[2048, 7, 1],
	#                    help='dim of input')
	#parser.add_argument('--maxn', type=int, default=2048,
	#                    help='max number of point cloud size')
	args = parser.parse_args()
        rospy.init_node('semantics', anonymous=True)
	Semantics = RosSemantics(args)
        Semantics.main()


if __name__ == '__main__':
	main()
