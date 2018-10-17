import tensorflow as tf
class NDT():
    def rescaleCovariance(self):
        EVAL_FACTOR=1000
        e,v=tf.self_adjoint_eig(self.Covariances)
        eS=e*EVAL_FACTOR
        emax=tf.tile(tf.expand_dims(tf.reduce_max(e, axis=1),-1),(1,3))
        e=tf.where(emax>eS,emax/EVAL_FACTOR,e)
        #e=tf.stack([e0,e1,e2])
        print e
        Lam=tf.matrix_diag(e)
        print Lam, v
        self.Covariances=tf.matmul(tf.matmul(v,Lam), v, transpose_b=True)


    def __init__(self,inputCloud, resolution, min_num_points=3):
        maxCoord = tf.reduce_max(inputCloud[:,:3], axis=0)
        minCoord = tf.reduce_min(inputCloud[:,:3], axis=0)
        ranges=(maxCoord-minCoord)
        nCells = tf.to_int32(ranges/resolution+1) #CHECK THIS!!!!!
        self.resolution=resolution

        cond = tf.to_int32(tf.floordiv(inputCloud[:,:3]-minCoord, resolution))
        Indexes= cond[:,0]+cond[:,1]*nCells[0]+ cond[:,2]*nCells[0]*nCells[1]
        tNCELLS=nCells[0]*nCells[1]*nCells[2]
        CellFreq = tf.unsorted_segment_sum(tf.ones_like(inputCloud[:,0]), Indexes,tNCELLS) 
        #REMOVE CELLS WITH FEW POINTS
        MoreThanNpoints=tf.greater(CellFreq,min_num_points)
        ValidCells=tf.where(MoreThanNpoints)
        tNCELLS_new=tf.shape(ValidCells)[0]
        ind_ = tf.to_int32(ValidCells)
        upd_ = tf.range(1,tNCELLS_new+1)
        shp_ = tf.expand_dims(tNCELLS,0)
        k=tf.scatter_nd(ind_, upd_, shp_)
        Indexes=tf.gather(tf.to_int64(k),Indexes)
        ValidPoints=tf.squeeze(tf.where(tf.not_equal(Indexes,0)),-1)
        inputCloud=tf.gather(inputCloud, ValidPoints)
        Indexes=tf.gather(Indexes, ValidPoints)-1
        CellFreq=tf.boolean_mask(CellFreq,MoreThanNpoints)
        tNCELLS=tNCELLS_new
        #END REMOVE
        sumCell = tf.unsorted_segment_sum(inputCloud[:,:3], Indexes,tNCELLS) 
        CellMeans = sumCell/tf.expand_dims(CellFreq,-1)
        CellPointMeans = tf.gather(CellMeans, Indexes)
        CellDiffs = inputCloud[:,:3] - CellPointMeans
     
        Covariances=tf.matmul(tf.expand_dims(CellDiffs,-1), tf.expand_dims(CellDiffs,1))
        Covariances=tf.unsorted_segment_sum(Covariances, Indexes, tNCELLS)
        Covariances=Covariances/(tf.reshape(CellFreq,(-1,1,1))+1)

        self.Covariances=Covariances
        #self.rescaleCovariance()
        self.Means=CellMeans
        self.NCELLS=tNCELLS

