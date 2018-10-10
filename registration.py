import tensorflow as tf
from gradient_computation import *
class RegistrationNDT():
    def n_nearest(self, Distances, CSum, n_neighbors):
        dist=tf.norm(Distances, axis=2)
        values, nearest=tf.nn.top_k(-dist, k=n_neighbors)
        #nearest=tf.Print(nearest, [tf.reduce_mean(values)])
        ind=tf.tile(tf.expand_dims(tf.range(tf.shape(dist)[0]),1), (1,n_neighbors))
        ind=tf.concat([tf.reshape(ind,(-1,1)), tf.reshape(nearest,(-1,1))],1)
        with tf.device('/device:GPU:0'):
            Distances= tf.gather_nd(Distances, ind)
            return Distances, tf.gather_nd(CSum,ind)
    def init_transform(self):
        self.PARAMS=tf.Variable([0,0,0,0,0,0],dtype=tf.float32, trainable=False)
        c=tf.cos(self.PARAMS[3:])
        s=tf.sin(self.PARAMS[3:])
        R=tf.stack([ tf.stack([c[1]*c[2], -c[1]*s[2], s[1]]),
            tf.stack([c[0]*s[2]+c[2]*s[0]*s[1], c[0]*c[2]-s[0]*s[1]*s[2], -c[1]*s[0]]),
            tf.stack([s[0]*s[2]-c[0]*c[2]*s[1], c[2]*s[0]+c[0]*s[1]*s[2], c[0]*c[1]])])
        Rotation=R
        Translation=self.PARAMS[:3]
        self.Transform=tf.concat([tf.concat([Rotation, tf.expand_dims(Translation,1)], axis=1), tf.constant([0,0,0,1], shape=(1,4),dtype=tf.float32)], axis=0)
        self.Rotation=Rotation
        self.Translation= Translation

    def add_pair_lamd(self,static, moving, n_neighbors):
        lfd1=1
        lfd2=0.05
        RotationTiled= tf.tile(tf.expand_dims(self.Rotation,0), (moving.NCELLS, 1,1))
        TransformedMeans=tf.matmul(RotationTiled, tf.expand_dims(moving.Means,-1))
        TransformedMeans=tf.squeeze(TransformedMeans, -1)+self.Translation
        TransformedCovars=tf.matmul(tf.matmul(RotationTiled,moving.Covariances, transpose_a=True),RotationTiled)
        Distances=tf.expand_dims(TransformedMeans,1)-tf.expand_dims(static.Means,0)
        CSum=tf.expand_dims(TransformedCovars,1)+tf.expand_dims(static.Covariances,0)
        MCov = tf.reshape(tf.tile(tf.expand_dims(moving.Covariances,1),(1,2,1,1)), [-1,3,3])
        Distances,CSum= self.n_nearest(Distances, CSum, n_neighbors)
        #Instead of inverse, cholesky decomposition
        #CInv = tf.matrix_inverse(CSum)
        CInv=tf.cholesky_solve(CSum, tf.tile(tf.expand_dims(tf.eye(3),0),(tf.shape(CSum)[0],1,1)))
        m_ij=tf.expand_dims(Distances,2)
        l=tf.matmul(tf.matmul(m_ij, CInv, transpose_a=True), m_ij)
        likelihood=tf.exp(-lfd2*l/2)
        loss=-lfd1*tf.reduce_sum(likelihood)
        G,H=gradients(m_ij,CInv, MCov, likelihood,lfd2,self.PARAMS[3:])
        return self.loss+loss,self.gradient-G,self.hessian-H
        #self.loss=self.loss+loss
        #GRADIENT COMPUTATION
        #self.gradient=self.gradient - G
        #self.hessian=self.hessian - H

    def add_pair(self,static, moving, n_neighbors=2):
        NM=tf.greater_equal(tf.shape(moving.Means)[0], n_neighbors)
        NS=tf.greater_equal(tf.shape(static.Means)[0], n_neighbors)
        HasDistributions=tf.logical_and(NM,NS)
        self.loss, self.gradient, self.hessian=tf.cond(HasDistributions, lambda: self.add_pair_lamd(static,moving, n_neighbors), lambda: (self.loss, self.gradient, self.hessian))

        
    def __init__(self):
        self.init_transform()
        self.loss=tf.constant(0, dtype=tf.float32)
        self.gradient=tf.zeros((6,1),dtype=tf.float32)
        self.hessian=tf.zeros((6,6),dtype=tf.float32)
        self.reset_transform=tf.variables_initializer([self.PARAMS])
    def get_train_op(self):
        Gradient=self.gradient
        Hessian=self.hessian
        #Hessian =tf.Print(Hessian ,[Hessian ],summarize=36)
        self.G=-tf.transpose(tf.gradients(self.loss, self.PARAMS))
        self.H=-tf.hessians(self.loss, self.PARAMS)[0]
        #Gradient=self.G
        #Hessian=self.H
        Hessian=regularize_Hessian(Hessian,Gradient)
        X=tf.cholesky_solve(tf.cholesky(Hessian), Gradient)
        #X=tf.matmul(tf.matrix_inverse(Hessian), Gradient)

        Diff_update = tf.squeeze(X)
        self.train_op=tf.assign_add(self.PARAMS, 0.9*Diff_update)
