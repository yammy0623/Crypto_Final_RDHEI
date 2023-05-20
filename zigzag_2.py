import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import random
random.seed()

def zigzag_create(M_pixel, N_pixel):
    n = 0
    zigzag = [[] for i in range(M_pixel + N_pixel -1)]
    for i in range(M_pixel):
        for j in range(N_pixel):
            index_sum = i + j
            if(index_sum % 2 == 0):
                zigzag[index_sum].insert(0, n)
            else:
                zigzag[index_sum].append(n)   
            n += 1
    i = 0
    j = 0
    scramble_index = np.zeros((M_pixel, N_pixel))
    for line in zigzag:
        for cont in line:
            scramble_index[i][j] = cont
            if(j == scramble_index.shape[1] - 1):
                i += 1
                j = 0
            else:
                j += 1

    return scramble_index

def img2block(img, piece, block_size):
    block = np.zeros((piece, piece, block_size, block_size))
    i, j = np.meshgrid(range(0, M, block_size), range(0, N, block_size), indexing='ij')
    i = i.ravel() 
    j = j.ravel()
    for k in range(len(i)):
        block[int(i[k]/block_size), int(j[k]/block_size)] = img[i[k]:i[k]+block_size, j[k]:j[k]+block_size]

    return block

def block2img(block, block_size, rotate):
    img_combine = np.zeros(img_size)
    ii, jj = np.meshgrid(range(0, M, block_size), range(0, N, block_size), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        if rotate == 90:
            img_combine[i:i+block_size, j:j+block_size] = np.rot90(block[int(i/block_size),int(j/block_size)]) # block rotate
        elif rotate == 270:
                img_combine[i:i+block_size, j:j+block_size] = np.rot90(np.rot90(np.rot90(block[int(i/block_size),int(j/block_size)]))) # block rotate
        else:
                img_combine[i:i+block_size, j:j+block_size] = block[int(i/block_size),int(j/block_size)]             
    return img_combine



if __name__ == "__main__":   
    img_size = (64,64)
    block_size = 16
    piece = int(img_size[0]/block_size)
    M = img_size[0]
    N = img_size[1]
    M_pixel = int(img_size[0]/block_size)
    N_pixel = int(img_size[1]/block_size)

    img = cv2.imread('300px-Lenna.jpg')
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    fig = plt.figure()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:,:,::-1] # BGR to RGB => for matplotlib
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img)

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(img_gray, cmap='gray')

    # 將圖片切成block
    img_gray_block = img2block(img_gray, piece, block_size)
    # zigzag pattern
    scramble_index =  zigzag_create(M_pixel, N_pixel)

    # rearrange image based on zigzag index map
    img_gray_zigzag_block = np.zeros(img_gray_block.shape)
    ii, jj = np.meshgrid(range(M_pixel), range(N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        new_i = int(scramble_index[i][j] / piece)
        new_j = int(scramble_index[i][j] % piece)
        img_gray_zigzag_block[i][j] = img_gray_block[new_i][new_j]


    # 將圖片拼回來
    # image reshape
    img_gray_combine = block2img(img_gray_zigzag_block, block_size,0)
    img_gray_rotate90_combine = block2img(img_gray_zigzag_block, block_size,90)

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(img_gray_combine, cmap='gray')
    plt.show()

    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(img_gray_rotate90_combine, cmap='gray')
    plt.show()


    # 照r排列
    # Key Generation
    P_sum = 0
    for i in range(M):
        for j in range(N):
            P_sum = P_sum + img_gray[i,j]

    a = 3.9 # control parameter
    N0 = 1000 # iteration number
    Y0 = P_sum/(M*N*255)
    Yn = a*Y0*(1-Y0) # Y1
    S = []
    Key2 = []
    for i in range(1, N0 + M*N):    
        Yn = a*Yn*(1-Yn)
        if i >= N0:
            S.append(Yn)
            Key2.append(math.floor(S[i-N0]*1e14)%256)

    img_diff = np.zeros(img_gray_rotate90_combine.shape)    
    for i in range(M):
        for j in range(N):
            # img_bin = [int(x) for x in bin(img_gray_rotate90_combine[i][j].astype('int32'))[2:]]
            # Key2_bin = [int(x) for x in bin(Key2[i*M + j])[2:]]
            # XOR = [a_^b_ for a_, b_ in zip(img_bin, Key2_bin)]
            # img_diff[i][j] = sum(val*(2**idx) for idx, val in enumerate(reversed(XOR)))
            img_diff[i][j] = int(img_gray_rotate90_combine[i][j])^Key2[i*M + j]

    fig2 = plt.figure()
    f1 = fig2.add_subplot(1,1,1)
    
    plt.imshow(img_diff, cmap='gray')
    plt.show()
    plt.hist(img_diff)

    # Secret Sharing
    # Galois Filed GF(2^8)
    img_diff_block = img2block(img_diff, piece, block_size)
    ii, jj = np.meshgrid(range(0,  M_pixel), range(0, N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    # 對每個block 做sharing
    sharing_num = 4 # 產生幾個sharing
    x_set = []
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        x_i = random.sample(range(256), sharing_num) # 產生四個x作為變數


# Recover

