from scipy.ndimage import imread
from scipy.misc import imresize,imshow
from vis_utils.tools import *
import numpy as np



class Distribution(object):

    def __init__(self,image_path,detail=300,noise=1e-2):
        self.image = imread(image_path)
        self.image = imresize(self.image,(detail,detail))
        self.noise=noise
        self.colors = [None, None, None]
        self.classes = [[],[],[]]
        self.h = int(self.image.shape[0])
        self.w = int(self.image.shape[1])
        for i in range(self.h):
            for j in range(self.w):
                if ceq(self.image[i][j],WHITE):
                    continue
                else:
                    found_color = False
                    pixel = np.array([0,0,0,1.0])
                    cl = np.argmax(self.image[i][j][:-1])
                    pixel[cl]=1.0
                    self.image[i][j] = pixel
                    for k,color in enumerate(self.colors):
                        if color is None:
                            continue
                        if ceq(self.image[i][j],color):
                            self.classes[k].append((i/detail-0.5,j/detail-0.5))
                            found_color = True
                            break
                    if not found_color:
                        self.colors[cl]=self.image[i][j]
                        self.classes[cl].append((i/detail-0.5,j/detail-0.5))


    @property
    def no_cls(self):
        return len(self.colors)

    @property
    def no_per_cls(self):
        return [len(x) for x in classes]


    def sample_cls(self,cls_id, size=1):
        if type(size)!=int:
            size = size*len(classes[cls_id])
        ind = np.random.choice(len(self.classes[cls_id]),size,replace=True)
        data = np.array([self.classes[cls_id][i] for i in ind])
        data += np.random.normal(0,self.noise,data.shape)
        labels = np.zeros([size,self.no_cls])
        labels[:,cls_id] = 1
        return data,labels

    def sample(self,size,per_cls=None):
        if per_cls is None:
            per_cls = [1/self.no_cls]*self.no_cls
        ret_samples = []
        ret_labels = []
        for i in range(self.no_cls):
            samples, labels = self.sample_cls(i,int(size*per_cls[i])) 
            ret_samples.extend(samples)
            ret_labels.extend(labels)
        return np.array(ret_samples),np.array(ret_labels)



