from zigzag_encrypt import encryptmain
from zigzag_decrypt import decryptmain
from client import sender
from server import receiver
import socket

identity = input('What is your identity?(0: sender, 1: receiver) ')
# method = int(input('What encrypt method do you choose?(0 : ZigZag, 1 : Key-based) '))
method = 1
if identity == '0':
    ##############先產生share##############
    file_name = input('Input file name: ')
    share_num = int(input('How many shares do you want to create?(more than 3) '))
    embed_data = int(input('Your secret (no more than 1048576): '))
    print("\n")
    print('Preparing secret sharing...')
    encryptmain(file_name, share_num, embed_data, method)
    print("Shares are created!!")
    ##############傳送share################
    print("Choose the shares to receivers")
    share_name = []
    keyname = []
    for i in range(3):
        shareN = input('Input share index: ')
        share_name.append(file_name[:-4] + '_share' + shareN + file_name[-4:])
        keyname.append("Key4share" + shareN + ".json")
    print("Connect to server...")
    addr = input("Enter Server's HOST,PORT: ").split(",")
    sender(addr, share_name, keyname)
else:
    ##############先接收share#############
    receiver()
    ##############還原img################
    share_name = []
    key_name = []
    for i in range(3):
        share_name.append('receive_share' + str(i+1) + '.png')
        key_name.append('Key4share' + str(i+1) + '.json')
    
    
    
    decryptmain(key_name, share_name, method)