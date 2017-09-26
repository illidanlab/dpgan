# public available function
from numpy import array, delete
import cPickle as pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib
matplotlib.use('agg')
import os, struct
from pylab import *
from array import array as pyarray
from numpy import *
from PIL import Image
from sklearn.preprocessing import binarize
from sklearn.metrics import f1_score

def normlization(image):
    '''divide each element of a image by 255, if its scale is in [0,255]'''
    im = image/255.0
    return im

# def age_filter(data):
#     '''remove certain data points by certain property'''
#
#     return data_new

def select_code(data, top): # top=64, smallest one is 128, final data shape: (36228, 64)
    '''select top "top" of feature (by frequency) appears in data (binarized) and remove data (in row) that don't have at least one of these features'''
    s = data.sum(axis=0) # count frequency of each feature
    a = array(range(len(s))) # index
    c = [x for _, x in sorted(zip(s, a), reverse=True)][:top]  # c contains indices correspondent to top ICD9 codes, is sorted according to frequency (from largest to smallest)
    a = zeros(len(s))
    a[c] = 1 # to one hot vector, a vector whose indices in c are 1 and all the other are 0
    data_selected = []  # store selected data
    for i in range(len(data)):
        if dot(data[i], a) == 0:  # if dot product is 0, this means the data vector don't have at least one of these features
            pass
        else:
            data_selected.append(data[i])
    return sorted(c), array(data_selected) #

def data_readf(top):
    '''Read MIMIC-III data'''
    with open('/home/xieliyan/Dropbox/GPU/Data/MIMIC-III/patient_vectors.pkl', 'rb') as f: # Original MIMIC-III data is in GPU1
        MIMIC_ICD9 = pickle.load(f) # dictionary, each one is a list
    MIMIC_data = []
    for key, value in MIMIC_ICD9.iteritems(): # dictionary to numpy array
        if mean(value) == 0: # skip all zero vectors, each patiens should have as least one disease of course
            continue
        MIMIC_data.append(value) # amax(MIMIC_data): 540,
        # if len(MIMIC_data) == 100:
        #     print "Break out to prevent out of memory issue"
        #     break
    # MIMIC_data = age_filter(MIMIC_data) # remove those patients with age 18 or younger
    MIMIC_data = binarize(array(MIMIC_data)) # binarize, non zero -> 1, average(MIMIC_data): , type(MIMIC_data[][]): <type 'numpy.int64'>
    index, MIMIC_data = select_code(MIMIC_data, top) # select top "top" codes and remove the patients that don't have at least one of these codes, see "applying deep learning to icd-9 multi-label classification from medical records"
    MIMIC_data = MIMIC_data[:, index] # keep only those coordinates (features) correspondent to top ICD9 codes
    num_data = (MIMIC_data.shape)[0] # data number
    dim_data = (MIMIC_data.shape)[1] # data dimension
    return MIMIC_data, num_data, dim_data # (46520, 942) 46520 942 for whole dataset

# MIMIC_data, num_data, dim_data = data_readf(top)
# print MIMIC_data.shape, num_data, dim_data

def load_MIMICIII(dataType, _VALIDATION_RATIO, top):
    MIMIC_data, num_data, dim_data = data_readf(top)
    if dataType == 'binary':
        MIMIC_data = clip(MIMIC_data, 0, 1)
    trainX, testX = train_test_split(MIMIC_data, test_size=_VALIDATION_RATIO, random_state=0)
    return trainX, testX, dim_data

# trainX, testX, num_data, dim_data = load_MIMICIII('binary', 0.1, 30)
# print trainX.shape, testX.shape, num_data, dim_data

def split(matrix, col):
    '''split matrix into feature and target (col th column of matrix), matrix \in R^{N*D}, f_r \in R^{N*(D-1)} , t_r \in R^{N*1}'''
    t_r = matrix[:,col] # shape: (len(t_r),)
    f_r = delete(matrix, col, 1)
    return f_r, t_r

def match(l1,l2):
    '''# count the matched position in 2 lists'''
    if len(l1) != len(l2):
        raise Exception('Two lists must have same length!')
    count = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            count = count + 1
    return count

def dwp(r, g, te):
    '''Dimension-wise prediction, r for real, g for generated, t for test, all without separated feature and target, all are numpy array'''
    rv = []
    gv = []
    for i in range(len(r[0])):
        print i
        f_r, t_r = split(r, i) # separate feature and target
        f_g, t_g = split(g, i)
        f_te, t_te = split(te, i) # these 6 are all numpy array
        if (unique(t_r).size == 1) or (unique(t_g).size == 1): # if only those coordinates correspondent to top codes are kept, no coordinate should be skipped, if those patients that doesn't contain top ICD9 codes were removed, more coordinates will be skipped
            print "skip this coordinate"
            continue

        # reg = linear_model.LinearRegression() # least square error
        # reg.fit(f_r, t_r)
        # target_r = reg.predict(f_te)
        # reg = linear_model.LinearRegression()
        # reg.fit(f_g, t_g)
        # target_g = reg.predict(f_te)
        # rv.append(square(linalg.norm(target_r-t_te)))
        # gv.append(square(linalg.norm(target_g-t_te)))

        model_r = linear_model.LogisticRegression() # logistic regression, if labels are all 0, this will cause: ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
        model_r.fit(f_r, t_r)
        label_r = model_r.predict(f_te)
        model_g = linear_model.LogisticRegression()
        model_g.fit(f_g, t_g)
        label_g = model_r.predict(f_te)
        # print label_r
        # print mean(model_r.coef_), count_nonzero(model_r.coef_), mean(model_g.coef_), count_nonzero(model_g.coef_) # statistics of classifiers
        rv.append(match(label_r, t_te)/(len(t_te)+10**(-10))) # simply match
        gv.append(match(label_g, t_te)/(len(t_te)+10**(-10)))
        # rv.append(f1_score(t_te, label_r)) # F1 score
        # gv.append(f1_score(t_te, label_g))
    return rv, gv

# r = array([[0.8,0.1,0.4,0.1], [0.2,0.3,0.5,0.6], [0.7,0.3,0.1,0.5], [0.9,0.5,0.6,0.11]])
# g = array([[0.1,0.3,0.2,0.4], [0.12,0.3,0.51,0.8], [0.23,0.13,0.5,0.2], [0.22,0.5,0.12,0.5]])
# te = array([[0.1,0.3,0.12,0.6], [0.2,0.3,0.4,0.7], [0.3,0.3,0.6,0.8], [0.2,0.5,0.9,0.03]])
# rv, gv = dwp(r, g, te)
# print rv, gv
# plt.plot(rv, gv)
# plt.savefig('./u.png')

# # test dwp using MIMIC-III data
# trainX, testX, _ = load_MIMICIII(dataType, _VALIDATION_RATIO, top)  # load whole dataset and split into training and testing set
# rv, gv = dwp(trainX, trainX, testX)
# plt.scatter(rv, gv)
# plt.title('Scatter plot of dimension-wise MSE')
# plt.xlabel('Real')
# plt.ylabel('Generated')
# plt.savefig('./result/genefinalfig/dwp.jpg')

# def scale_transform(self, image):
#     '''this function transform the scale of generated image (0, largest pixel value) to (0,255) linearly'''
#     im = array(image)
#     Max = amax(im)
#     for i in range(len(im)):
#         im[i] = (im[i] / Max) * 255
#     return im


# def im_avg(im):
#     '''compress image from rbg to grayscale, input should be numpy array'''
#     return average(im, axis=2).reshape(64,64,1)

# def loaddata_face(path):
#     # for file in os.listdir(path):
#     #     print file
#     im_name = array([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
#     N = len(im_name) # count files in directory
#     # N = 10
#     image_n = zeros(shape=(N, 64, 64, 1)) # normalized image
#     for i in range(N):
#         jpgfile = Image.open(path + im_name[i])
#         # print asarray(jpgfile.getdata(),dtype=float64).shape
#         # print jpgfile.size
#         # image_n[i] = im_avg(asarray(jpgfile.getdata(),dtype=float64).reshape((jpgfile.size[1],jpgfile.size[0],(asarray(jpgfile.getdata(),dtype=float64).shape)[1])))
#         image_n[i] = normlization(asarray(jpgfile.getdata(),dtype=float64).reshape((jpgfile.size[1],jpgfile.size[0],1))) # image is averaged
#     return image_n

# path = "./face/CelebA/img_align_celeba_50k_1st_r_64_64_1/"
# im = loaddata_face(path)

# def loaddata_face_batch(dataset, batch_size):
#     '''random select batch from whole dataset'''
#     res = dataset[random.choice(len(dataset), batch_size)]
#     # res = res.reshape((batch_size, 784))  # type(res[0][0]): numpy.float64
#     return res

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