# Created by ay27 at 16/11/8
import numpy as np


class Tensor:
    def __init__(self, data):
        if not (isinstance(data, np.ndarray) or isinstance(data, list)):
            raise ValueError('data nust be ndarray or list')
        self.data = np.array(data)
        self.shape = self.data.shape

    def vectorization(self):
        return self.data.reshape(-1)

    def t2mat(self, rdims, cdims):
        if isinstance(rdims, list) and isinstance(cdims, list):
            indies = rdims + cdims
            rsize = np.prod([self.shape[i] for i in rdims])
            csize = np.prod([self.shape[i] for i in cdims])
        elif isinstance(rdims, int):
            indies = [rdims]
            indies.extend(cdims)
            rsize = self.shape[rdims]
            csize = np.prod([self.shape[i] for i in cdims])
        else:
            raise ValueError('rdims and cdims must be int or list')
        tmp = np.reshape(np.transpose(self.data, indies), (rsize, csize))
        return tmp

    def __cmp__(self, other):
        t1 = self.data.reshape(-1)
        t2 = other.data.reshape(-1)
        if len(t1) != len(t2):
            return False
        for ii in range(len(t1)):
            if t1[ii] != t2[ii]:
                return False
        return True

