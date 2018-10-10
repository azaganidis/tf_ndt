import tensorflow as tf
def He(x,fi):
    #x : [?,3,1]
    x=tf.reshape(x,(-1,3))
    NUM=tf.shape(x)[0]
    x=tf.transpose(x)
    #x : [3,?]
    sx=tf.sin(fi[0])
    sy=tf.sin(fi[1])
    sz=tf.sin(fi[2])
    cx=tf.cos(fi[0])
    cy=tf.cos(fi[1])
    cz=tf.cos(fi[2])
    x1=x[0,...]
    x2=x[1,...]
    x3=x[2,...]
    N0 = tf.stack([tf.zeros(NUM),tf.zeros(NUM), tf.zeros(NUM)])
    a=tf.stack([
        tf.zeros(NUM),
        x1*(-cx*sz-sx*sy*cz)+x2*(-cx*cz+sx*sy*sz)+x3*(sx*cy),
        x1*(-sx*sz+cx*sy*cz)+x2*(-cx*sy*sz-sx*cz)+x3*(-cx*cy)
    ])
    b=tf.stack([
        tf.zeros(NUM),
        x1*(cx*cy*cz)+x2*(-cx*cy*sz)+x3*(cx*sy),
        x1*(sx*cy*cz)+x2*(-sx*cy*sz)+x3*(sx*sy)
    ])
    c=tf.stack([
        tf.zeros(NUM),
        x1*(-sx*cz-cx*sy*sz)+x2*(-sx*sz-cx*sy*cz),
        x1*(cx*cz-sx*sy*sz)+x2*(-sx*sy*cz-cx*sz)
    ])
    d=tf.stack([
        x1*(-cy*cz)+x2*(cy*sz)+x3*(-sy),
        x1*(-sx*sy*cz)+x2*(sx*sy*sz)+x3*(sx*cy),
        x1*(cx*sy*cz)+x2*(-cx*sy*sz)+x3*(-cx*cy)
    ])
    e=tf.stack([
        x1*(sy*sz)+x2*(sy*cz),
        x1*(-sx*cy*sz)+x2*(-sx*cy*cz),
        x1*(cx*cy*sz)+x2*(cx*cy*cz)
    ])
    f=tf.stack([
        x1*(-cy*cz)+x2*(cy*sz),
        x1*(-cx*sz-sx*sy*cz)+x2*(-cx*cz+sx*sy*sz),
        x1*(-sx*sz+cx*sy*cz)+x2*(-cx*sy*sz-sx*cz)
    ])
    H=tf.stack([
        [N0, N0, N0, N0, N0, N0],
        [N0, N0, N0, N0, N0, N0],
        [N0, N0, N0, N0, N0, N0],
        [N0, N0, N0, a , b , c ],
        [N0, N0, N0, b , d , e ],
        [N0, N0, N0, c , e , f ]])
    H=tf.expand_dims(tf.transpose(H, perm=[3,0,1,2]),-1)
    return H

def ZHe(C1):
    NUM=tf.shape(C1)[0]
    C1=tf.transpose(C1, perm=[1,2,0])
    N0 = tf.zeros(NUM)
    b0 = tf.stack([
        [N0, N0, N0],
        [N0, N0, N0],
        [N0, N0, N0]])
    b1=tf.stack([
        [N0,-C1[0,1],-C1[0,2]],
        [-C1[0,1],2*C1[2,2]-2*C1[1,1],-4*C1[1,2]],
        [-C1[0,2],-4*C1[1,2],2*C1[1,1]-2*C1[2,2]]])
    b2=tf.stack([
        [N0,C1[0,0]-C1[2,2],C1[1,2]],
        [C1[0,0]-C1[2,2],2*C1[0,1],2*C1[0,2]],
        [C1[1,2],2*C1[0,2],-2*C1[0,1]]])
    b3=tf.stack([
        [N0,C1[1,2],C1[0,0]-C1[1,1]],
        [C1[1,2],-2*C1[0,2],2*C1[0,1]],
        [C1[0,0]-C1[1,1],2*C1[0,1],2*C1[0,2]]])
    b4=tf.stack([
        [2*C1[2,2] - 2*C1[0,0], -C1[0,1],-4*C1[0,2]],
        [-C1[0,1],N0,-C1[1,2]],
        [-4*C1[0,2], -C1[1,2], 2*C1[0,0] - 2*C1[2,2]]])
    b5=tf.stack([
        [-2*C1[1,2],C1[0,2],2*C1[0,1]],
        [C1[0,2],N0,C1[1,1] - C1[0,0]],
        [2*C1[0,1], C1[1,1] - C1[0,0], 2*C1[1,2]]])
    b6=tf.stack([
        [2*C1[1,1] - 2*C1[0,0], -4*C1[0,1], -C1[0,2]],
        [-4*C1[0,1], 2*C1[0,0] - 2*C1[1,1], -C1[1,2]],
        [-C1[0,2], -C1[1,2], N0]])
    ZHest=tf.stack([
        [b0,b0,b0,b0,b0,b0],
        [b0,b0,b0,b0,b0,b0], 
        [b0,b0,b0,b0,b0,b0],
        [b0,b0,b0,b1,b2,b3],
        [b0,b0,b0,b2,b4,b5],
        [b0,b0,b0,b3,b5,b6],])
    ZHest=tf.transpose(ZHest, perm=[4,0,1,2,3])
    return ZHest


def Ze_FAST(C):
    #C : [?,3,3]
    NUM=tf.shape(C)[0]
    C=tf.transpose(C, perm=[1,2,0])
    Z=tf.zeros([3,18])
    Z0=tf.stack([
        tf.stack([tf.zeros(NUM),tf.zeros(NUM),tf.zeros(NUM)]),
        tf.stack([tf.zeros(NUM),tf.zeros(NUM),tf.zeros(NUM)]),
        tf.stack([tf.zeros(NUM),tf.zeros(NUM),tf.zeros(NUM)])])
    Z1=tf.stack([
        tf.stack([tf.zeros(NUM),-C[0,2,:],C[0,1,:]]),
        tf.stack([-C[0,2,:],-2*C[1,2,:],-C[2,2,:]+C[1,1,:]]),
        tf.stack([C[0,1,:],-C[2,2,:]+C[1,1,:],2*C[1,2,:]])])
    Z2=tf.stack([
        tf.stack([2*C[0,2,:], C[1,2,:], -C[0,0,:]+C[2,2,:]]),
        tf.stack([C[1,2,:],tf.zeros(NUM),-C[0,1,:]]),
        tf.stack([-C[0,0,:]+C[2,2,:],-C[0,1,:], -2*C[0,2,:]])])
    Z3=tf.stack([
        tf.stack([-2*C[0,1,:], -C[1,1,:]+C[0,0,:], -C[1,2,:]]),
        tf.stack([-C[1,1,:]+C[0,0,:], 2*C[0,1,:], C[0,2,:]]),
        tf.stack([-C[1,2,:], C[0,2,:],tf.zeros(NUM)])])
    Ze=tf.transpose(tf.stack([Z0,Z0,Z0,Z1,Z2,Z3]), perm=[3,0,1,2])
    return Ze
def Je_FAST(x,fi):
    #x : [?,3,1]
    x=tf.reshape(x,(-1,3))
    NUM=tf.shape(x)[0]
    x=tf.transpose(x)
    #x : [3,?]
    sx=tf.sin(fi[0])
    sy=tf.sin(fi[1])
    sz=tf.sin(fi[2])
    cx=tf.cos(fi[0])
    cy=tf.cos(fi[1])
    cz=tf.cos(fi[2])
    x1=x[0,...]
    x2=x[1,...]
    x3=x[2,...]
    a=x1*(-sx*sz+cx*sy*cz)+x2*(-sx*cz-cx*sy*sz)+x3*(-cx*cy)
    b=x1*(cx*sz+sx*sy*cz)+x2*(-sx*sy*sz+cx*cz)+x3*(-sx*cy)
    c=x1*(-sy*cz)+x2*(sy*sz)+x3*(cy)
    d=x1*(sx*cy*cz)+x2*(-sx*cy*sz)+x3*(sx*sy)
    e=x1*(-cx*cy*cz)+x2*(cx*cy*sz)+x3*(-cx*sy)
    f=x1*(-cy*sz)+x2*(-cy*cz)
    g=x1*(cx*cz-sx*sy*sz)+x2*(-cx*sz-sx*sy*cz)
    h=x1*(sx*cz+cx*sy*sz)+x2*(cx*sy*cz-sx*sz)
    Je= tf.stack([
        tf.stack([tf.ones(NUM),tf.zeros(NUM),tf.zeros(NUM),tf.zeros(NUM),c,f]),
        tf.stack([tf.zeros(NUM),tf.ones(NUM),tf.zeros(NUM),a, d, g]),
        tf.stack([tf.zeros(NUM),tf.zeros(NUM),tf.ones(NUM),b, e, h])])
    Je=tf.transpose(Je, perm=[2,0,1])
    return Je

def gradients(x,B,C, likelihood,lfd2, angle):
    J=Je_FAST(x, angle)
    Z=Ze_FAST(C)
    Zh=ZHe(C)
    H=He(x,angle)
    xtB = tf.matmul(x,B, transpose_a=True)
    xtBJ=tf.matmul(xtB, J)
    xtBZ=tf.matmul(tf.tile(tf.expand_dims(xtB,1), (1,6,1,1)) , Z)
    Bx=tf.tile(tf.expand_dims(tf.matmul(B,x),1), (1,6,1,1))
    xtBZBx=tf.squeeze(tf.matmul(xtBZ, Bx), [2,3])
    q=2*xtBJ-tf.expand_dims(xtBZBx,1)
    factor = -(lfd2/2)*likelihood
    Q= tf.transpose(factor*q, perm=[0,2,1])

    #HESSIAN
    xtB_Tiled=tf.tile(tf.reshape(xtB,(-1,1,1,1,3)),(1,6,6,1,1)) #(?,6,6,1,3)
    B_Tiled=tf.tile(tf.reshape(B,(-1,1,1,3,3)),(1,6,6,1,1)) #(?,6,6,3,3)
    x_Tiled=tf.tile(tf.reshape(x,(-1,1,1,3,1)),(1,6,6,1,1)) #(?,6,6,3,1)
    xtBZh=tf.matmul(xtB_Tiled,Zh) # (?,6,6,1,3)
    xtBZhB=tf.matmul(xtBZh,B_Tiled)# (?,6,6,1,3)
    JTBJ=tf.matmul(tf.matmul(J,B, transpose_a=True),J) #CHECKED (?,6,6)
    xtBH=tf.reshape(tf.matmul(xtB_Tiled,H),(-1,6,6)) #CHECKED (?,6,6)
    xtBZhBx=tf.reshape(tf.matmul(xtBZhB,x_Tiled),(-1,6,6))# CHECKED (?,6,6)
    xtBZB= tf.matmul(xtBZ, tf.tile(tf.reshape(B,(-1,1,3,3)),(1,6,1,1)))# (?,6,1,3)
    xtBZBJ=tf.matmul(tf.squeeze(xtBZB,2),J) #OK MAYBE? (?,6,6)

    #WRONG
    #xtBZBZ=tf.matmul(tf.tile(tf.expand_dims(xtBZB,2),(1,1,6,1,1)), tf.tile(tf.expand_dims(Z,1),(1,6,1,1,1)))
    #xtBZBZB=tf.matmul(xtBZBZ,B_Tiled)
    #xtBZBZBx=tf.squeeze(tf.matmul(xtBZBZB,x_Tiled),(3,4))
    #END WRONG

    Z_=tf.tile(tf.expand_dims(Z, 1),(1,6,1,1,1))
    B_=tf.tile(tf.expand_dims(tf.expand_dims(B, 1),1),(1,6,6,1,1))
    BZ_=tf.matmul(B_,Z_)
    BZBZ_=tf.matmul(BZ_,BZ_)
    BZBZB_=tf.matmul(BZBZ_,B_)
    xtBZBZBx=tf.matmul(tf.matmul(x_Tiled, BZBZB_,transpose_a=True), x_Tiled)
    xtBZBZBx=tf.squeeze(xtBZBZBx,(3,4))

    Hessian=factor*(2*JTBJ+2*xtBH -xtBZhBx -2*tf.transpose(xtBZBJ,perm=[0,2,1])-2*xtBZBJ +xtBZBZBx +tf.transpose(xtBZBZBx,perm=[0,2,1])-lfd2*tf.matmul(Q,Q,transpose_b=True)/2 );
    Q=tf.reduce_sum(Q, 0)
    Hessian=tf.reduce_sum(Hessian, 0)
    return Q,Hessian

def regularize_Hessian(H,G):
    e,v=tf.self_adjoint_eig(H)
    minCoeff=tf.reduce_min(e)
    maxCoeff=tf.reduce_max(e)
    regularizer=tf.norm(G)
    regularizer=tf.cond(regularizer+minCoeff>0, lambda: regularizer,lambda: 0.001*maxCoeff-minCoeff )
    e=e+regularizer
    Lam=tf.diag(e)
    H=tf.matmul(tf.matmul(v,Lam), v, transpose_b=True)
    return H


