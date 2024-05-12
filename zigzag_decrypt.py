import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import random
import galois
import time
import json
random.seed()

def img2block(img, M, N, M_pixel, N_pixel, block_size):
    block = np.zeros((M_pixel, N_pixel, block_size, block_size))
    i, j = np.meshgrid(range(0, M, block_size), range(0, N, block_size), indexing='ij')
    i = i.ravel() 
    j = j.ravel()
    for k in range(len(i)):
        block[int(i[k]/block_size), int(j[k]/block_size)] = img[i[k]:i[k]+block_size, j[k]:j[k]+block_size]

    return block

def block2img(M, N, img_size, block, block_size, rotate):
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

def extracting_data(all_share, threshold, M_pixel, N_pixel):
    gf2 = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
    u1=int(np.sqrt(threshold+1))
    u2=8-u1
    extracted_block=np.zeros((3,M_pixel*N_pixel,2,2), dtype=int)
    merge_images=np.zeros((3,M_pixel*2,N_pixel*2), dtype=int)
    cal=np.zeros((3,1), dtype=int)
    for i in range(3):
        marked=all_share[i]
        marked=np.reshape(marked,(M_pixel*N_pixel,2,2))
        b=1
        total=''
        for j in range(M_pixel*N_pixel):
            if marked[j,0,0]<128 : # ES block
                # first=length.format(int(marked[j,0,1] % (2**(u2-1))))
                # second=length.format(int(marked[j,1,0] % (2**(u2-1))))
                # third=length.format(int(marked[j,1,1] % (2**(u2-1))))
                first=format(int(marked[j,0,1]),'#008b')
                second=format(int(marked[j,1,0]),'#008b')
                third=format(int(marked[j,1,1]),'#008b')
                first=first[-u2:]
                second=second[-u2:]
                third=third[-u2:]
                bit=first + second  + third
                total=total+bit
                b=b+1

        extracted_data = total[-20:]

        for k in range(M_pixel*N_pixel):
            if marked[k,0,0]<128 : # ES block
                # difference[0]=length.format(int(marked[k,0,1] / (2**(u2-1))))
                # difference[1]=length.format(int(marked[k,1,0] / (2**(u2-1))))
                # difference[2]=length.format(int(marked[k,1,1] / (2**(u2-1))))
                difference_1=format(int(marked[k,0,1]),'08b')
                difference_2=format(int(marked[k,1,0]),'08b')
                difference_3=format(int(marked[k,1,1]),'08b')
                # RP_bin=format(marked[k,0,0].astype(int),'#008b')
                RP_bin=int(marked[k,0,0])+int(total[k])*128
                # RP_bin='{0:08b}'.format(RP_bin)
                # print(RP_bin)
                # RP_recover=total[k]+RP_bin[3:]
                extracted_block[i,k,0,0] = RP_bin
                
                cal[0] = gf2(extracted_block[i,k,0,0]) - gf2(int(difference_1[:2],2))
                cal[1] = gf2(extracted_block[i,k,0,0]) - gf2(int(difference_2[:2],2))
                cal[2] = gf2(extracted_block[i,k,0,0]) - gf2(int(difference_3[:2],2))
                # print(int(difference_1[2:(u1+2)],2))
                extracted_block[i,k,0,1]=cal[0]
                extracted_block[i,k,1,0]=cal[1]
                extracted_block[i,k,1,1]=cal[2]
                
            else : # NS block
                RP_bin=format(marked[k,0,0].astype(int),'#008b')
                RP_recover=total[k]+RP_bin[3:]
                extracted_block[i,k,0,0]=int(RP_recover,2)
                extracted_block[i,k,0,1]=marked[k,0,1]
                extracted_block[i,k,1,0]=marked[k,1,0]
                extracted_block[i,k,1,1]=marked[k,1,1]

        a=0
        for m in range(M_pixel):
            for n in range(int(N_pixel/2)):
                add=np.hstack((extracted_block[i,a,:,:], extracted_block[i,(a+1),:,:]))
                merge_images[i,m*2:(m+1)*2,n*2*2:2*(n+1)*2]=add
                a=a+2
    return merge_images, extracted_data
               


def shamir_secret_recover(All_share_block, img_size, block_size, all_x, M_pixel, N_pixel):
    gf2 = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
    all_x = gf2(all_x)
    All_shareY = gf2(All_share_block) # 只需要三個share (3, 4, 4, 75, 75)
    # print(All_share_block.shape)
    
    xx = gf2(0)
    sharing_recover_block = np.zeros((M_pixel, N_pixel, block_size, block_size))
    ii, jj = np.meshgrid(range(M_pixel), range(N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()    
    # 針對每個block去還原
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        x = all_x[k]
        shareY = All_shareY[:,i,j].reshape(3,4).T # (3,2,2) -> (4,4)
        
        block_result = []
        for Y in shareY: # 4個pixel
            s = (Y[0]*(xx-x[1])*(xx-x[2])/((x[0]-x[1])*(x[0]-x[2]))) +\
            (Y[1]*(xx-x[0]*(xx-x[2]))/((x[1]-x[0])*(x[1]-x[2]))) +\
            (Y[2]*(xx-x[0])*(xx-x[1])/((x[2]-x[0])*(x[2]-x[1])))
            block_result.append(s)
            
        # 3個share求出一個block
        block_result = np.array(block_result).T.reshape(2,2) # (2,2) => block
        sharing_recover_block[i][j] = block_result
    
    M = img_size[0]
    N = img_size[1]
    sharing_recover = block2img(M, N, img_size, sharing_recover_block, block_size, 0)
    plt.figure()
    plt.imshow(sharing_recover, cmap='gray')    
    return sharing_recover

def image_recover_zigzag(sharing_recover, block_size, scramble_index, Key2):
    img_size = sharing_recover.shape
    M = sharing_recover.shape[0]
    N = sharing_recover.shape[1]
    M_pixel = int(M/block_size)
    N_pixel = int(N/block_size)

    # fig = plt.figure()
    img_recover = np.zeros(sharing_recover.shape)
    for i in range(M):
        for j in range(N):
            img_recover[i][j] = int(sharing_recover[i][j]) ^Key2[i*M + j]
    # im1 = fig.add_subplot(1,2,1)
    # im1.imshow(img_recover, cmap='gray')

    img_recover_block = img2block(img_recover, M, N, M_pixel, N_pixel, block_size)
    img_rotate_recover = block2img(M, N, img_size, img_recover_block, block_size, 270)
    # im2 = fig.add_subplot(1,2,2)
    # im2.imshow(img_rotate_recover, cmap='gray')
    
    # rezigzag pattern
    rezigzag_index = np.argsort(scramble_index.reshape(-1)).reshape(M_pixel, N_pixel)
    img_recover_block = img2block(img_rotate_recover, M, N, M_pixel, N_pixel, block_size)

    # rearrange image based on zigzag index map
    img_reigzag_block = np.zeros(img_recover_block.shape)
    ii, jj = np.meshgrid(range(M_pixel), range(N_pixel), indexing='ij')
    ii = ii.ravel() 
    jj = jj.ravel()
    for k in range(len(ii)):
        i = ii[k]
        j = jj[k]
        new_i = int(rezigzag_index[i][j] / M_pixel)
        new_j = int(rezigzag_index[i][j] % N_pixel)
        img_reigzag_block[i][j] = img_recover_block[new_i][new_j]
    img_recover_complete = block2img(M, N, img_size, img_reigzag_block, block_size, 0)
    # plt.figure()
    # plt.imshow(img_recover_complete, cmap = 'gray')  
    return img_recover_complete  

def image_recover_keybased(M, N, M_pixel, N_pixel, share, Key1, Key2):
    block_size=2
    index=0
    blocks=np.zeros((M_pixel*N_pixel,2,2),dtype=int)
    merge_blocks=np.zeros((M,N))
    unscramble_image=np.zeros((M_pixel*N_pixel,2,2), dtype=int)
    for i in range(M_pixel):
        for j in range(N_pixel):
            blocks[index]=share[i*block_size:block_size*(i+1),j*block_size:block_size*(j+1)]
            index=index+1
    recover_block_order=np.argsort(Key1)

    for i in range(M_pixel*N_pixel):
        new_pixel_order=np.argsort(Key2[i][0])
        g=np.reshape(blocks[i],(4))
        h=g[new_pixel_order]
        recover_shape=np.reshape(h,(2,2))
        unscramble_image[recover_block_order[i]]=recover_shape
    
    a=0
    for i in range(M_pixel):
        for j in range(int(N_pixel/2)):
            add=np.hstack((unscramble_image[a], unscramble_image[a+1]))
            merge_blocks[i*2:(i+1)*2,j*2*2:2*(j+1)*2]=add
            a=a+2
    return merge_blocks
        
def decryptmain(keyname, share_name, scramble_type):
    print("The keys you have: ", keyname)
    print("---- read the key ----")
    print(keyname)
    print(share_name)
    all_x = []
    for i in range(3):
        with open(keyname[i],'r') as f:
            key = json.load(f)
            if scramble_type == 0 :
               Key1 = np.array(key["key1"])
               Key2 = key["key2"]
            if scramble_type == 1 : 
               Key1 =  np.array(key["key1"])
               Key2 = np.array(key["key2"])

            threshold = key["key3"]
            sharex = key["shareX"]
            all_x.append(sharex)
    all_x = np.array(all_x).T
    # print(all_x.shape)
    block_num = Key1.shape[0]
    print("The shares you have: ", share_name)
    print('---- load sharing ----')
    share1 = cv2.imread(share_name[0], cv2.COLOR_BGR2GRAY)
    share2 = cv2.imread(share_name[1], cv2.COLOR_BGR2GRAY)
    share3 = cv2.imread(share_name[2], cv2.COLOR_BGR2GRAY)
    img_size = share1.shape
    # img_size = (300,300) # 375,454
    # share1 = cv2.resize(share1, img_size, interpolation=cv2.INTER_AREA)
    # share2 = cv2.resize(share2, img_size, interpolation=cv2.INTER_AREA)
    # share3 = cv2.resize(share3, img_size, interpolation=cv2.INTER_AREA)
    block_size = 2    
    scrambel_block_size = int(img_size[0]/ block_num)
    M = share1.shape[0]
    N = share1.shape[1]
    M_pixel = int(img_size[0]/block_size)
    N_pixel = int(img_size[1]/block_size)
    share1_block = img2block(share1, M, N, M_pixel, N_pixel, block_size)
    share2_block = img2block(share2, M, N, M_pixel, N_pixel, block_size)
    share3_block = img2block(share3, M, N, M_pixel, N_pixel, block_size)
    All_share_block = [share1_block, share2_block, share3_block]
    # All_share_block = np.array(All_share_block).astype(np.int32)    
    # All_share_block = np.array(All_share_block)
    print('---- extracting data ----')
    extrated_images,additional_data= extracting_data(All_share_block, threshold, M_pixel, N_pixel)
    # print('extracted data',int(additional_data,2))
    print('Your secret message: ', int(additional_data,2))
    # print('Your secret message: ', additional_data)

    print('---- secret sharing recover ----')
    share1_block = img2block(extrated_images[0], M, N, M_pixel, N_pixel, block_size)
    share2_block = img2block(extrated_images[1], M, N, M_pixel, N_pixel, block_size)
    share3_block = img2block(extrated_images[2], M, N, M_pixel, N_pixel, block_size)
    All_share_block = [share1_block, share2_block, share3_block]
    All_share_block = np.array(All_share_block).astype(np.int32) 
    sharing_recover = shamir_secret_recover(All_share_block, img_size, block_size, all_x, M_pixel, N_pixel)
    cv2.imwrite('sharing.png',sharing_recover)

    print('---- image recover ----')
    if scramble_type == 0:
       img_recover_complete = image_recover_zigzag(sharing_recover, scrambel_block_size, Key1, Key2)
    if scramble_type == 1:
       img_recover_complete = image_recover_keybased(M, N, M_pixel, N_pixel, sharing_recover, Key1, Key2)
    
    saving_name = share_name[0][:-11]+"_recover" + share_name[0][-4:]
    print(saving_name, " saving...")
    cv2.imwrite(saving_name, img_recover_complete)
    print('Finished!')

if __name__ == "__main__":   
    print("---- img recover -----")
    start  = time.time()
    index = [2,3,4]
    keyname = []
    share_name = []
    for i in range(3):
        keyname.append("Key4share" + str(index[i]) + ".json")
        share_name.append("512_lena_share" + str(index[i]) + ".png")
    scramble_type = 1 #int(input('Which scramble_type you use ? (0 : Zigzag, 1 : Key-based) '))
    decryptmain(keyname, share_name, scramble_type)
    
    end = time.time()
    print('total time: ', format(end-start))

    # print("---- read the key ----")
    # with open('imgencrpytKey.json','r') as f:
    #     key = json.load(f)
    #     Key1 = np.array(key["key1"])
    #     Key2 = key["key2"]
    #     all_x = np.array(key["shareX"])
    # block_num = Key1.shape[0]

    # print('---- load sharing ----')
    # file_name = 'brain'
    # share1 = cv2.imread(file_name + '_share1.jpg', cv2.COLOR_BGR2GRAY)
    # share2 = cv2.imread(file_name + '_share2.jpg', cv2.COLOR_BGR2GRAY)
    # share3 = cv2.imread(file_name + '_share3.jpg', cv2.COLOR_BGR2GRAY)
    # img_size = (300,300) # 375,454
    # share1 = cv2.resize(share1, img_size, interpolation=cv2.INTER_AREA)
    # share2 = cv2.resize(share2, img_size, interpolation=cv2.INTER_AREA)
    # share3 = cv2.resize(share3, img_size, interpolation=cv2.INTER_AREA)
    # block_size = 2
    # scrambel_block_size = int(img_size[0]/ block_num)
    # M = share1.shape[0]
    # N = share1.shape[1]
    # M_pixel = int(img_size[0]/block_size)
    # N_pixel = int(img_size[1]/block_size)
    # share1_block = img2block(share1, M_pixel, N_pixel, block_size)
    # share2_block = img2block(share2, M_pixel, N_pixel, block_size)
    # share3_block = img2block(share3, M_pixel, N_pixel, block_size)
    # All_share_block = [share1_block, share2_block, share3_block]
    # All_share_block = np.array(All_share_block).astype(np.int32)

    # print('---- secret sharing recover ----')
    # sharing_recover = shamir_secret_recover(All_share_block, block_size, all_x, M_pixel, N_pixel)
    
    # print('---- image recover ----')
    # img_recover_complete = image_recover(sharing_recover, scrambel_block_size, Key1)
    
    # cv2.imwrite(file_name+'_recover.jpg', img_recover_complete)
