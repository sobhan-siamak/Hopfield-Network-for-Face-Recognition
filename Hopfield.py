


#  @copy by sobhan siamak



import numpy as np
import pandas as pd
import PIL
from PIL import Image
import sklearn
import glob
import matplotlib.pyplot as plt
import math

# c = 0
dict = {'subject01':1, 'subject02':2, 'subject03':3, 'subject04':4, 'subject05':5, 'subject06':6, 'subject07':7}
def LoadData():
    imageNames = glob.glob("TrainData/*.*")
    img = []
    lbls = []
    c = 0
    for i in imageNames:
        s = i.split('\\')[-1]
        s = s.split('.')[0]
        lbls += [dict[s]]
        im = Image.open(i)
        # if c == 6:
        #     im.show()
        # im2 = im.load()
        im = im.resize((32,32))
        m,n = im.size
        im2 = np.array(im)
        im2 = np.reshape(im2, (32*32, 1))
        im2 = np.squeeze(im2)
        img += [im2]
        c+=1

    return np.array(img), np.array(lbls), c


images, labels, c = LoadData()


print("shape of image is:", np.shape(images))


#reading Test Data
# cnt = 0
dict = {'subject01':1, 'subject02':2, 'subject03':3, 'subject04':4, 'subject05':5, 'subject06':6, 'subject07':7}
def LoadTestData():
    imgNames = glob.glob("TestData/*.*")
    imgt = []
    lblst = []
    cnt = 0
    for i in imgNames:
        s = i.split('\\')[-1]
        s = s.split('.')[0]
        lblst += [dict[s]]
        im = Image.open(i)
        # if cnt == 0:
        #     im.show()
        # im2 = im.load()
        im = im.resize((32, 32))
        m, n = im.size
        im2 = np.array(im)
        im2 = np.reshape(im2, (32 * 32, 1))
        im2 = np.squeeze(im2)
        imgt += [im2]
        cnt += 1
    return  np.array(imgt), np.array(lblst), cnt


imagest, labelst, cnt = LoadTestData()

print("shape of test image is:", np.shape(imagest))



#decimal to 8 bit binary
def D2B(numb):
    # return bin(numb).replace("0b", "")
    return format(numb, '08b')

#binary to decimal
def B2D(bnumb):
    bnumb = list(bnumb)
    decimal = 0
    m = len(bnumb)
    for i in range(m):
        digit = bnumb.pop()
        if digit == 1:
            decimal += pow(2, i)
    return decimal
# def B2D(bnumb):
#     b = [str(i) for i in bnumb]
#     st = ""
#     st2 = st.join(b)
#     dnumb = int(st2, 2)
#     return dnumb


def Rshape(vector):
    if len(vector) == 1024:
        Rimage = np.reshape(vector, (32,32))

    return Rimage




def Vec2Mat(img):
    bits = 8
    m = np.size(img)
    face = np.zeros((m, bits))
    for i in range(m):
        bi = D2B(img[i])
        bi = list(bi)
        bi = [int(j) for j in bi]
        face[i] = bi

    return face

#convert train data to 7 list with each size 1024*8
listmat = [[] for i in range(c)]
# print(images)
for f in range(c):
    li = Vec2Mat(images[f])
    listmat[f] = li

#convert test data to 70 list with each size 1024*8
testlistmat = [[] for i in range(cnt)]
for ft in range(cnt):
    tli = Vec2Mat(imagest[ft])
    testlistmat[ft] = tli



#Start Hopfield Network
def energy(weight, input):
    en = -0.5 * (np.dot(np.dot(input, weight), np.transpose([input])))
    en = np.squeeze(en)
    return en

def act_func(weight, input):
    output = np.dot(weight, input)
    output = np.sign(output)
    output = np.where(output < 0,0,output)
    return output

def weight(input):
    m,n = np.shape(input)
    # print(m,n)
    # print(input[0])
    w = 0
    for i in range(m):
        w1 = np.outer(input[i], input[i])
        # print(w1)
        # print(np.shape(w1))
        w1 = w1 - np.diag(np.diag(w1))
        # w1 /= n
        w =+ w1
    # w = np.outer(input, input)# w is symmetric
    # w = w - np.diag(np.diag(w))# zeros of main diag
    # w /= len(input)# Normalize w
    w /= n#Normalize weigth matrix
    return w
#start training
print("listmat elements are:")
print(listmat[4][:,0])
def trainWeight():
    m, n, p = np.shape(listmat)#7,1024,8
    w = []
    hop1 = [listmat[i][:,0] for i in range(m)]
    w1 = weight(hop1)
    w.append(w1)
    # print(w)
    # print(np.shape(w))
    hop2 = [listmat[i][:,1] for i in range(m)]
    w2 = weight(hop2)
    w.append(w2)
    hop3 = [listmat[i][:,2] for i in range(m)]
    w3 = weight(hop3)
    w.append(w3)
    hop4 = [listmat[i][:,3] for i in range(m)]
    w4 = weight(hop4)
    w.append(w4)
    hop5 = [listmat[i][:,4] for i in range(m)]
    w5 = weight(hop5)
    w.append(w5)
    hop6 = [listmat[i][:,5] for i in range(m)]
    w6 = weight(hop6)
    w.append(w6)
    hop7 = [listmat[i][:,6] for i in range(m)]
    w7 = weight(hop7)
    w.append(w7)
    hop8 = [listmat[i][:,7] for i in range(m)]
    w8 = weight(hop8)
    w.append(w8)


    return w

w = trainWeight()
# print(np.shape(w[0]))
# print(w[7])
print(len(w))


# we = np.dot(w[0],testlistmat[0][:,0])
# print("we is",we)

def HopfieldTrain(w,TestSample):
    result = []
    m = len(w)##### m = 8
    for i in range(m):
        sample = TestSample[:, i]
        while True:
            sampletset = act_func(w[i], sample)
            en1 = energy(w[i], sampletset)
            en2 = energy(w[i], sample)
            if np.all(sampletset) == np.all(sample):
                result.append([sampletset])
                break
            if en1 >= en2:
                result.append([sampletset])
                break
            sample = sampletset
    return result




#return vector with 1024 size of integer values
def convert(binary):
    m, n = np.shape(binary)#m=1024, n=8
    pixel = []
    for i in range(m):
        pixel.append(B2D(binary[i]))
    return pixel

def distance(img1, img2):
    dist = np.sqrt(np.sum([(xi - yi) ** 2 for xi, yi in zip(img1, img2)]))
    return dist

def visual(re):
    plt.figure()
    plt.imshow(re)
    plt.show()



def finalResult(tlistmat, w, counter):
    img = np.transpose(HopfieldTrain(w, tlistmat))
    img = np.squeeze(img)
    pixels = convert(img)
    dist = []
    for i in range(7):
        di = distance(pixels, images[i])
        dist.append(di)
    mn = np.argmin(dist)
    pixels2 = Rshape(images[mn])
    pixel2 = Rshape(images[counter])
    return pixel2









# for i in range(cnt):
#    re = finalResult(testlistmat[i], w, (i // 10))

re = finalResult(testlistmat[69], w, (69//10))
visual(re)






# def accuracy(labelst):


print("######## w elements are:########")
print(np.shape(w[0]))
print(w[0])


# print(testlistmat[0][:,0])

# imgresult = np.transpose(HopfieldTrain(w,testlistmat[69]))
# # imgresult = HopfieldTrain(w,testlistmat[4])
# imgresult = np.squeeze(imgresult)
# # print(imgresult)
# print(np.shape(testlistmat[0]))
# print(np.shape(imgresult))
# print(testlistmat[0][1009])
# print(imgresult[1009])
# print("################")
# print(np.shape(imgresult))
#
# # pix = convert(imgresult)
# # pix2 = Rshape(pix)
# # pix3 = np.asarray(pix2)
# # print(pix2)
# plt.figure()
# # # plt.imshow(pix3, interpolation="none")
# # plt.imshow(pix2)
# # plt.show()
# #
# # pixe = convert(testlistmat[69])
# # pixe = Rshape(pixe)
# # plt.imshow(pixe)
# # plt.show()
# #
# p = convert(listmat[6])
# p = Rshape(p)
# plt.imshow(p)
# plt.show()
#
#
# # print(Rshape(imagest[0]))
#
#




























"""

print(len(listmat))
print(np.shape(listmat))
print(np.shape(testlistmat))


m,n,p = np.shape(listmat)
print("m is",m,"\nn is",n,"\np is", p)

a = [listmat[i][:,0] for i in range(m)]
a = np.squeeze(a)
print(a)
















ab = np.squeeze(listmat)
aa = listmat[0]
print("element is:", ab[0][:,4])
# print(aa)

xy = ab[0][:,1]
z = np.outer(xy,xy)
print(z)


print(ab)


at = np.squeeze(testlistmat)

print("test list mat is:", at[1][34])
print("cnt is:", cnt)

# bd = B2D(at[0][0])
# print(bd)




bd = B2D(at[1][34])
print(bd)
"""