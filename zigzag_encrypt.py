import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import galois
import time
import json
import pprint
random.seed()

def img2block(img, block_size):
    img_size = img.shape
    M = img_size[0]
    N = img_size[1]
    M_pixel = int(img_size[0]/block_size)
    N_pixel = int(img_size[1]/block_size)    
    block = np.zeros((M_pixel, N_pixel, block_size, block_size))
    i, j = np.meshgrid(range(0, M, block_size), range(0, N, block_size), indexing='ij')
    i = i.ravel() 
    j = j.ravel()
    for k in range(len(i)):
        block[int(i[k]/block_size), int(j[k]/block_size)] = img[i[k]:i[k]+block_size, j[k]:j[k]+block_size]
    return block

def block2img(img_size, block, block_size, rotate):
    M = img_size[0]
    N = img_size[1]    
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

def zigzag_scramble(M_pixel, N_pixel, img_gray_block):
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
    
    # rearrange image based on zigzag index map
    img_gray_zigzag_block = np.zeros(img_gray_block.shape)
    ii, jj = np.meshgrid(range(M_pixel), range(N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        new_i = int(scramble_index[i][j] / M_pixel)
        new_j = int(scramble_index[i][j] % N_pixel)
        img_gray_zigzag_block[i][j] = img_gray_block[new_i][new_j]

    return scramble_index, img_gray_zigzag_block
   
def key_based(M_pixel, N_pixel, img_gray_block):
    index=0
    merge_blocks=np.zeros((M_pixel*2,N_pixel*2), dtype=int)
    scramble_block=np.zeros((M_pixel*N_pixel,2,2), dtype=int)
    blocks=np.zeros((M_pixel*N_pixel,2,2), dtype=int)
    pixel_scramble_order=np.zeros((M_pixel*N_pixel,1,4), dtype=int)
    block_scramble_order=random.sample(range(0,M_pixel*N_pixel),M_pixel*N_pixel)
    for i in range(M_pixel):
        for j in range(N_pixel):
            scramble_block[block_scramble_order[index]]=img_gray_block[index]
            pixel_scramble_order[block_scramble_order[index]]=random.sample(range(0,4),4)
            e=scramble_block[block_scramble_order[index]].flatten()
            m=e[pixel_scramble_order[block_scramble_order[index]]]
            blocks[block_scramble_order[index]]=np.reshape(m,[2,2])
            index=index+1
    
    a=0
    for i in range(M_pixel):
        for j in range(int(N_pixel/2)):
            add=np.hstack((blocks[a], blocks[a+1]))
            merge_blocks[i*2:(i+1)*2,j*2*2:2*(j+1)*2]=add
            a=a+2
            
    return block_scramble_order, pixel_scramble_order, merge_blocks



def logisticMap(img_gray,M,N,a,N0):
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
    return Key2

def shamir_secret_sharing(img_diff, block_size, sharing_num):
    gf2 = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
    M_pixel = int(img_diff.shape[0]/block_size)
    N_pixel = int(img_diff.shape[1]/block_size)
    img_scrambled_block = img2block(img_diff, block_size)
    img_sharing_block = np.zeros((sharing_num, M_pixel, N_pixel, block_size, block_size))

    ii, jj = np.meshgrid(range(M_pixel), range(N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    all_a= []
    all_x = []
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        a = random.sample(range(1,int(255/(sharing_num^2))), 2)
        x = random.sample(range(1, sharing_num+1), sharing_num) # 產生四個share
        all_a.append(a)
        all_x.append(x)
    
        s_array = img_scrambled_block[i][j].flatten()
        
        block_result = []
        for s in s_array:
            s = int(s)
            fx = gf2(s) + gf2(a[0])*gf2(x) + gf2(a[1])*gf2(x)**2
            block_result.append(fx)
            
        # 一個block產出4個share
        block_result = np.array(block_result)
        block_result = block_result.T.reshape(sharing_num,2,2) # 4,2,2
        for share_idx in range(sharing_num):
            img_sharing_block[share_idx][i][j] = block_result[share_idx]

    all_x = np.array(all_x)
    # print(all_x.shape)
    return  img_sharing_block, all_x

def embedded_data(share, additional_data, M_pixel, N_pixel):
    gf2 = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
    share_block = img2block(share, 2) # M*N*2*2
    # Calculate MSB
    share_block=np.reshape(share_block,[M_pixel*N_pixel,2,2])
    # All_MSB = share_block[:,0,0].flatten()
    # All_MSB = (All_MSB / 128).astype('int32').astype(str)
    # print("all MSB shape", All_MSB.shape)
    # All_MSB_bin = ''.join(All_MSB) # 將所有MSB併起來
    # print("MSB bin = ", All_MSB_bin)
    # Calculate D_max
    All_MSB=np.zeros(M_pixel*N_pixel)
    for i in range(M_pixel*N_pixel):
        All_MSB[i]=share_block[i,0,0]/128
    All_MSB=All_MSB.astype(int).astype(str)
    All_MSB_bin=''.join(All_MSB)
    # print(All_MSB_bin)
    D_all = np.zeros((M_pixel*N_pixel,3))
    share_block = share_block.astype('int32')
    # for i in range(1,4):
    D_all[:,0] = gf2(share_block[:,0,0]) - gf2(share_block[:,0,1])
    D_all[:,1] = gf2(share_block[:,0,0]) - gf2(share_block[:,1,0])
    D_all[:,2] = gf2(share_block[:,0,0]) - gf2(share_block[:,1,1])

    D_max = np.max(D_all, axis = 1)
    # print("d max shape", D_max.shape)
    # Calculate threshold
    add_data_bin = format(additional_data, 'b')
    # print(add_data_bin)
    histDmax, histBin = np.histogram(D_max, bins = range(256))
    # print(histBin)
    payload_length = M_pixel*N_pixel + len(add_data_bin)
    for u1 in range(2,7):
        threshold = 2**u1 - 1
        u2 = 8 - u1
        ES_amount = sum(histDmax[:threshold+1])
        if payload_length/(3*u2) <= ES_amount:
            u1choose = u1
            break # u1輸出
    total_bit=3*u2*ES_amount
    zeros=total_bit-payload_length
    zeros=np.zeros((zeros),dtype=int).astype(str)
    zeros=''.join(zeros)
    payload=All_MSB_bin + zeros + add_data_bin
    
    ES_embedded=np.zeros((2,2), dtype=int)
    NS_embedded=np.zeros((2,2), dtype=int)
    share_embed=np.zeros((M_pixel*N_pixel,2,2), dtype=int)
    blocks_merge=np.zeros((M_pixel*2,N_pixel*2), dtype=int)
    D_max=D_max.flatten()
    share_block=np.reshape(share_block,[M_pixel*N_pixel,2,2])
    cal0=D_all[:,0].flatten().astype(int)
    cal1=D_all[:,1].flatten().astype(int)
    cal2=D_all[:,2].flatten().astype(int)
    b=1
    h=1
    for i in range(M_pixel*N_pixel):
        if D_max[i] <= threshold: # ES block
            if share_block[i,0,0]<128: # ES's RP's MSB=0
                ES_embedded[0,0]=share_block[i,0,0]
            else:
                ES_embedded[0,0]=share_block[i,0,0]-128
            a=(b-1)*u2*3
            ES_embedded[0,1]= int((format(cal0[i], 'b') + payload[a:(a+u2)]),2)
            ES_embedded[1,0]= int((format(cal1[i], 'b') + payload[(a+u2):(a+(2*u2))]),2)
            ES_embedded[1,1]= int((format(cal2[i], 'b') + payload[(a+(2*u2)):(a+(3*u2))]),2)
            share_embed[i]=ES_embedded
            # print(i)
            b=b+1
            h=h+1
        else: # NS block
            NS_embedded=share_block[i]
            if share_block[i,0,0]<128:
                NS_embedded[0,0]=share_block[i,0,0]+128
            else:
                NS_embedded[0,0]=share_block[i,0,0]
            share_embed[i]=NS_embedded
    
    a=0
    for i in range(M_pixel):
        for j in range(int(N_pixel/2)):
            add=np.hstack((share_embed[a], share_embed[a+1]))
            blocks_merge[i*2:(i+1)*2,j*2*2:2*(j+1)*2]=add
            a=a+2
    return blocks_merge,threshold

def encryptmain(file_name, share_num, additional_data, scramble_type):
    img_size = (300,300) # 375,454
    scramble_block_size = 2
    block_size = scramble_block_size
    M = img_size[0]
    N = img_size[1]
    M_pixel = int(img_size[0]/block_size)
    N_pixel = int(img_size[1]/block_size)
    
    img = cv2.imread(file_name)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''''''''''zigzag'''''''''''
    if scramble_type==0:
        print("---- Image scrambling: ZigZag ----")
        img_gray_block = img2block(img_gray, block_size)
        # zigzag pattern
        Key1, img_gray_zigzag_block =  zigzag_scramble(M_pixel, N_pixel, img_gray_block)
        # image reshape
        img_gray_combine = block2img(img_size, img_gray_zigzag_block, block_size,0)
        img_gray_rotate90_combine = block2img(img_size,img_gray_zigzag_block, block_size,90)
        '''Key Generation'''
        Key2 = logisticMap(img_gray,M,N,3.9,10000)
        img_diff = np.zeros(img_gray_rotate90_combine.shape)    
        for i in range(M):
            for j in range(N):
                img_diff[i][j] = int(img_gray_rotate90_combine[i][j])^Key2[i*M + j]
    
    '''''''''''keybased'''''''''''
    if scramble_type==1:
        print("---- Image scrambling: Key-based ----")
        img_gray_block = img2block(img_gray, block_size)
        img_gray_block=np.reshape(img_gray_block,[M_pixel*N_pixel,2,2])
        # keybased
        Key1, Key2, img_diff =  key_based(M_pixel, N_pixel, img_gray_block)
        # print(img_diff.shape)
        cv2.imwrite('scramble.png',img_diff)

    '''shamir secret sharing'''
    print("---- Shamir Secret Sharing ----")
    block_size= 2
    img_sharing_block, all_x = shamir_secret_sharing(img_diff, block_size, share_num)
    for i in range(share_num):
        share = block2img(img_size, img_sharing_block[i], block_size, 0)
        share_name = file_name[:-4] + '_share' + str(i+1) + file_name[-4:]
        # if scramble_type==0:
        #    share_name = file_name[:-4] + '_share' + str(i+1) + 'z' + file_name[-4:]
        # if scramble_type==1:
        #    share_name = file_name[:-4] + '_share' + str(i+1)  + 'k' + file_name[-4:]
    
    '''data embedding'''
    print("---- Data Embedding ----")
    print("Your secret message: ", additional_data)
    # additional_data=random.sample(range(0,2**20),1)
    # additional_data=788000

    for i in range(share_num):
        share = block2img(img_size, img_sharing_block[i], block_size, 0)
        # marked_encrtpted_share, threshold=embedded_data(share, additional_data[0], M_pixel, N_pixel)
        marked_encrtpted_share, threshold = embedded_data(share, additional_data, M_pixel, N_pixel)
        share_name = file_name[:-4] + '_share' + str(i+1) + file_name[-4:]
        cv2.imwrite(share_name, marked_encrtpted_share)

    print('Additional_data',additional_data)

    print('----- key saving -----')
    for i in range(share_num):
        shareX = all_x.T[i].T   
        if scramble_type==0:
           jsonDict = {"key1": Key1.tolist(), "key2": Key2, "shareX": shareX.tolist()}
        if scramble_type==1:
           jsonDict = {"key1": Key1, "key2": Key2.tolist(), "shareX": shareX.tolist(), "key3": threshold}
        with open('Key4share' + str(i+1) +'.json', 'w') as f:
            json.dump(jsonDict, f, indent = 2)

if __name__ == "__main__": 
    file_name = '512_lena.png'
    share_num = 4
    print('start') 
    scramble_type = 1 #int(input('Which scramble type you want ? (0 : ZigZag, 1 : Key-based) : '))
    additional_data = int(input('Your secret (no more than 1048576): '))
    start  = time.time()
    encryptmain(file_name, share_num, additional_data, scramble_type)
    end = time.time()
    print('total time: ', format(end-start))   
    
    # block_size = scramble_block_size
    # file_name = '512_Lena'


    # img = cv2.imread('512_Lena.png')
    # img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img[:,:,::-1] # BGR to RGB => for matplotlib

    # # Plot: Original Image & Histogram
    # # fig = plt.figure()
    # # ax1 = fig.add_subplot(2,2,1).imshow(img)
    # # ax2 = fig.add_subplot(2,2,2).imshow(img_gray, cmap='gray')
    # # cv2.imwrite(file_name+'_origin.jpg', img)
    # # cv2.imwrite(file_name+'_gray.jpg', img_gray)
    # # -----
    # # plt.hist(img_gray.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    # # plt.title('Original Histogram')
    # # plt.savefig(file_name+'_hist.jpg')
    # # -----
    
    # '''zigzag'''
    # print("---- Image scrambling: ZigZag ----")
    # # 將圖片切成block
    # img_gray_block = img2block(img_gray, M_pixel, N_pixel, block_size)
    # # zigzag pattern
    # Key1, img_gray_zigzag_block =  zigzag_scramble(M_pixel, N_pixel, img_gray_block)
        
    # # image reshape
    # img_gray_combine = block2img(img_gray_zigzag_block, block_size,0)
    # img_gray_rotate90_combine = block2img(img_gray_zigzag_block, block_size,90)

    # #----
    # # fig = plt.figure()
    # # ax3 = fig.add_subplot(2,2,3)
    # # ax3.imshow(img_gray_combine, cmap='gray')
    # # ax4 = fig.add_subplot(2,2,4)
    # # ax4.imshow(img_gray_rotate90_combine, cmap='gray')
    # # cv2.imwrite(file_name+'_zigzag.jpg', img_gray_combine)
    # # cv2.imwrite(file_name+'_zigzag90.jpg', img_gray_rotate90_combine)
    # #---

    # '''Key Generation'''
    # # 照r排列
    # # Key Generation
    # Key2 = logisticMap(M,N,3.9,10000)
    # img_diff = np.zeros(img_gray_rotate90_combine.shape)    
    # for i in range(M):
    #     for j in range(N):
    #         img_diff[i][j] = int(img_gray_rotate90_combine[i][j])^Key2[i*M + j]  

    # # ---
    # # diff = plt.figure()
    # # d1 = diff.add_subplot(2,2,1).imshow(img_diff, cmap='gray')
    # # d2 = diff.add_subplot(2,2,2).hist(img_diff.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)  
    # # cv2.imwrite(file_name+'_logissticmap.jpg', img_diff)
    # # ---
    # # plt.hist(img_diff.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    # # plt.title('ZigZag Pattern Scramble Histogram')
    # # plt.savefig(file_name+'Scramble_hist.jpg')

    # '''shamir secret sharing'''
    # print("---- Shamir Secret Sharing ----")
    # block_size= 2
    # sharing_num = 4
    # img_sharing_block, all_x = shamir_secret_sharing(img_diff, block_size, sharing_num)

    # if sharing_num == 4:
    #     share1 = block2img(img_sharing_block[0], block_size,0)
    #     share2 = block2img(img_sharing_block[1], block_size,0)
    #     share3 = block2img(img_sharing_block[2], block_size,0)
    #     share4 = block2img(img_sharing_block[3], block_size,0)
    #     sharefig = plt.figure()
    #     s1 = sharefig.add_subplot(2,2,1).imshow(share1, cmap = 'gray')
    #     plt.title('Share1')
    #     s2 = sharefig.add_subplot(2,2,2).imshow(share2, cmap = 'gray')
    #     plt.title('Share2')
    #     s3 = sharefig.add_subplot(2,2,3).imshow(share3, cmap = 'gray')
    #     plt.title('Share3')
    #     s4 = sharefig.add_subplot(2,2,4).imshow(share4, cmap = 'gray')
    #     plt.title('Share4')
    #     plt.tight_layout()

    #     cv2.imwrite(file_name+'_share1.jpg', share1)
    #     cv2.imwrite(file_name+'_share2.jpg', share2)
    #     cv2.imwrite(file_name+'_share3.jpg', share3)
    #     cv2.imwrite(file_name+'_share4.jpg', share4)
    #     sharefig.savefig(file_name+'_Share.jpg')

    #     hist = plt.figure()
    #     h1 = hist.add_subplot(2,2,1).hist(share1.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    #     plt.title('Share1 Histogram')
    #     h2 = hist.add_subplot(2,2,2).hist(share2.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    #     plt.title('Share2 Histogram')
    #     h3 = hist.add_subplot(2,2,3).hist(share3.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    #     plt.title('Share3 Histogram')
    #     h4 = hist.add_subplot(2,2,4).hist(share4.flatten(), bins=255, lw=0.5, ec="black", fc="green", alpha=0.5)
    #     plt.title('Share4 Histogram')
    #     plt.tight_layout() 
    #     hist.savefig(file_name+'Share1_hist.jpg')
    

    # # Save the key
    # print('----- key saving -----')
    # # print(type(Key1), type(Key2), type(all_x))
    # jsonDict = {"key1":Key1.tolist(), "key2": Key2, "shareX": all_x.tolist()}
    # with open('imgencrpytKey.json', 'w') as f:
    #     json.dump(jsonDict, f, indent = 4)
