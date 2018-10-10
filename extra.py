
    def init_transform_QUAT(self):
        self.PARAMS=tf.Variable([0,0,0,1,0,0,0],dtype=tf.float32, constraint=
                lambda x:tf.concat([tf.nn.l2_normalize(x[:4]), x[4:]], 0))
        RotationQuatN=self.PARAMS[:4]
        qz=RotationQuatN[0]
        qy=RotationQuatN[1]
        qx=RotationQuatN[2]
        qw=RotationQuatN[3]

        R1L1=tf.stack([qw, qz, -qy, qx])
        R1L2=tf.stack([-qz, qw, qx, qy])
        R1L3=tf.stack([qy, -qx, qw, qz])
        R1L4=tf.stack([-qx, -qy, -qz, qw])
        ############
        R2L1=tf.stack([qw, qz, -qy, -qx])
        R2L2=tf.stack([-qz, qw, qx, -qy])
        R2L3=tf.stack([qy, -qx, qw, -qz])
        R2L4=tf.stack([qx, qy, qz, qw])
        ######
        R1=tf.stack([R1L1,R1L2,R1L3,R1L4])
        R2=tf.stack([R2L1,R2L2,R2L3,R2L4])
        Rotation=tf.matmul(R1,R2)
        Translation=self.PARAMS[4:]
        transf_ = tf.concat([Translation,[Rotation[3,3]]],-1)
        self.Transform=tf.concat([Rotation[:,:3], tf.expand_dims(transf_,1)], axis=1)
        self.Rotation=Rotation[:3,:3]
        self.Translation= Translation
    def Je_PROPER(x, Rparam):
        s=tf.sin(Rparam)
        c=tf.cos(Rparam)
        sx=s[0]; sy=s[1]; sz=s[2]
        cx=c[0]; cy=c[1]; cz=c[2]
        x1=x(0); x2=x(1); x3=x(2)
        a=x1*(-sx*sz+cx*sy*cz)+x2*(-sx*cz-cx*sy*sz)+x3*(-cx*cy)
        b=x1*(cx*sz + sx*sy*cz) + x2*(−sx*sy*sz + cx*cz) + x3*(−sx*cy)
        c=x1*(−sy*cz) + x2*(sy*sz) + x3*(cy)
        d=x1*(sx*cy*cz) + x2*(−sx*cy*sz) + x3*(sx*sy)
        e=x1*(−cx*cy*cz) + x2*(cx*cy*sz) + x3*(−cx*sy)
        f=x1*(−cy*sz) + x2*(−cy*cz)
        g=x1*(cx*cz − sx*sy*sz) + x2*(−cx*sz − sx*sy*cz)
        h=x1*(sx*cz + cx*sy*sz) + x2*(cx*sy*cz − sx*sz)
        Je= tf.stack([
            tf.stack([1,0,0,0,c,f]),
            tf.stack([0,1,0,a,d,g]),
            tf.stack([0,0,1,b,e,h])])
        return Je


