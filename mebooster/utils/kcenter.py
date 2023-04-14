import numpy as np

def pairwise_distances(A, B): #800,10; 200;10
    
    na = np.sum(np.square(A), 1) #800,1
    nb = np.sum(np.square(B), 1) #200,1
    
    na = np.reshape(na, [-1, 1])
    nb = np.reshape(nb, [1, -1])
    # print("na: ", na.shape)
    # print("nb:", nb.shape)
    D = np.sqrt(np.maximum(na - 2 * np.matmul(A, np.transpose(B)) + nb, 0.0))#tf.matmul(A,B,False,True) transpose_a=false, transpose_b=true
    return D

class KCenter(object):
    def __init__(self):
    
        self.A = []
        self.B = []

        # D = []#pairwise_distances(self.A, self.B)

        # D_min = np.min(D, axis=1)
        self.D_min_max = [] #np.reduce_max(D_min)
        self.D_min_argmax =[] #np.argmax(D_min)

    def cal_D_min_max(self, A, B):
        self.A = A
        self.B = B
        D = pairwise_distances(self.A, self.B)
        D_min = np.min(D, axis=1)
        self.D_min_max = np.max(D_min) #scalar
        self.D_min_argmax = np.argmax(D_min) #scalar
        return self.D_min_max, self.D_min_argmax
