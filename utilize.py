# public available function
from numpy import array, delete
import cPickle as pickle
# from sklearn import linear_model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, struct
from pylab import *
from array import array as pyarray
from numpy import *
from PIL import Image

def normlization(image):
    '''divide each element of a image by 255, if its scale is in [0,255]'''
    im = image/255.0
    return im

# def data_readf():
#     '''Read MIMIC-III data'''
#     with open('./MIMIC-III/patient_vectors.pkl', 'rb') as f:
#         MIMIC_ICD9 = pickle.load(f) # dictionary, each one is a list
#     MIMIC_data = []
#     for key, value in MIMIC_ICD9.iteritems(): # dictionary to numpy array
#         MIMIC_data.append(value)
#     MIMIC_data = array(MIMIC_data)
#     num_data = (MIMIC_data.shape)[0] # data number
#     dim_data = (MIMIC_data.shape)[1] # data dimension
#
#     return MIMIC_data, num_data, dim_data
#
# # MIMIC_data, num_data, dim_data = data_readf()
# # print MIMIC_data.shape, num_data, dim_data
#
# def split(matrix, col):
#     '''split matrix into feature and target (col th column of matrix), matrix \in R^{N*D}, f_r \in R^{N*(D-1)} , t_r \in R^{N*1}'''
#     t_r = matrix[:,col] # shape: (len(t_r),)
#     f_r = delete(matrix, col, 1)
#     return f_r, t_r
#
# def match(l1,l2):
#     '''# count the matched position in 2 lists'''
#     if len(l1) != len(l2):
#         raise Exception('Two lists must have same length!')
#     count = 0
#     for i in range(len(l1)):
#         if l1[i] == l2[i]:
#             count = count + 1
#     return count
#
# def dwp(r, g, te):
#     '''Dimension-wise prediction, r for real, g for generated, t for test, all without separated feature and target, all are numpy array'''
#     rv = []
#     gv = []
#     for i in range(len(r[0])):
#         f_r, t_r = split(r, i) # separate feature and target
#         f_g, t_g = split(g, i)
#         f_te, t_te = split(te, i)
#         model_r = linear_model.LogisticRegression()
#         model_r.fit(f_r, t_r)
#         label_r = model_r.predict(f_te)
#         model_g = linear_model.LogisticRegression()
#         model_g.fit(f_g, t_g)
#         label_g = model_r.predict(f_te)
#         print label_r, label_g, t_te
#         rv.append(match(label_r, t_te)/(len(t_te)+10**(-10)))
#         gv.append(match(label_g, t_te)/(len(t_te)+10**(-10)))
#
#     return rv, gv # return 2 vectors, both with length dim_data, classification error rate
#
# r = array([[0.8,0.1,0.4,0.1], [0.2,0.3,0.5,0.6], [0.7,0.3,0.1,0.5], [0.9,0.5,0.6,0.11]])
# g = array([[0.1,0.3,0.2,0.4], [0.12,0.3,0.51,0.8], [0.23,0.13,0.5,0.2], [0.22,0.5,0.12,0.5]])
# te = array([[0.1,0.3,0.12,0.6], [0.2,0.3,0.4,0.7], [0.3,0.3,0.6,0.8], [0.2,0.5,0.9,0.03]])
# rv, gv = dwp(r, g, te)
# print rv, gv
# plt.plot(rv, gv)
# plt.savefig('./u.png')
#
# def scale_transform(self, image):
#     '''this function transform the scale of generated image (0, largest pixel value) to (0,255) linearly'''
#     im = array(image)
#     Max = amax(im)
#     for i in range(len(im)):
#         im[i] = (im[i] / Max) * 255
#     return im


# def im_avg(im):
#     '''compress image from rbg to grayscale, input should be numpy array'''
#     return average(im, axis=2).reshape(28,28,1)

def loaddata_face(path):
    # for file in os.listdir(path):
    #     print file
    im_name = array([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    N = len(im_name) # count files in directory, the file names in original total: "000001.jpg" to "202599.jpg"
    image_n = zeros(shape=(N, 64, 64, 3)) # normalized image
    for i in range(N):
        jpgfile = Image.open(path + im_name[i])
        # print asarray(jpgfile.getdata(),dtype=float64).shape
        # print jpgfile.size
        # image_n[i] = im_avg(normlization(asarray(jpgfile.getdata(),dtype=float64).reshape((jpgfile.size[1],jpgfile.size[0],(asarray(jpgfile.getdata(),dtype=float64).shape)[1]))))
        image_n[i] = normlization(asarray(jpgfile.getdata(),dtype=float64).reshape((jpgfile.size[1],jpgfile.size[0],(asarray(jpgfile.getdata(),dtype=float64).shape)[1]))) # image is averaged
    return image_n

# path = "./face/CelebA/img_align_celeba_10000_1st_r_28/"
# im = loaddata_face(path)

def loaddata_face_batch(dataset, batch_size):
    '''random select batch from '''
    res = dataset[random.choice(len(dataset), batch_size)]
    # res = res.reshape((batch_size, 784))  # type(res[0][0]): numpy.float64
    return res

# batch_size = 2
# res = loaddata_face_batch(im, batch_size)

# load data and labels into matrix of specific digit
def loaddata(digits, dataset='training',
             path='.'):  # digits should among 0-9, dataset should be 'training' or 'testing', path is where you store your dataset file

    # get the path of dataset
    if dataset is 'training':
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    else:
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    # if this is a label file
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack('>II', flbl.read(
        8))  # read the header information in the label file, '>II' means using big-endian, read 8 characters.
    lbl = pyarray("b", flbl.read())  # 'b' for signed char
    flbl.close()

    # if this is a image file
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))  # read the header information in the image file.
    img = pyarray('B', fimg.read())  # 'B' for unsigned char
    fimg.close()

    # extract the labels conrresponding to the digits we want
    ind = [k for k in range(size) if str(lbl[k]) in digits]  # list that contain the labels
    N = len(ind)  # number of labels

    # store and return the result
    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols])  # every row is an image. every row: array([784 data, , , ,...])
        labels[i] = lbl[ind[i]]
    labels = [v[0] for v in labels] # array to int

    return images, labels


# images:
# #array([[0, 0, 0, ..., 0, 0, 0],
# #       [0, 0, 0, ..., 0, 0, 0],
# #       [0, 0, 0, ..., 0, 0, 0],
# #       ...,
# #       [0, 0, 0, ..., 0, 0, 0],
# #       [0, 0, 0, ..., 0, 0, 0],
# #       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
# the type of each element is not float !
#
# images[0]:
# #array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)
#
# labels:
# #array([[ 1],
# #       [ 1],
# #       [ 1],
# #       ...,
# #       [ 1],
# #       [-1],
# #       [ 1]], dtype=int8)
#
# #labels[1] looks like: array([1], dtype=int8)
# #labels[3] looks like: array([-1], dtype=int8)
# #labels[1]*labels[3] is array([-1], dtype=int8)
# #labels[1]*2 is array([2], dtype=int8)
#
# average norm of MNIST data (scale: 0 to 255) is 2349.74572748