import socket
import time
# Sender

def sender(addr, share_name, keyname):
    #############connect###############
    socket02 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET:默認IPv4, SOCK_STREAM:TCP
    HOST = addr[0]
    PORT = int(addr[1])
    try:
        socket02.connect((HOST,PORT)) # 客戶端指定要串接的IP位址跟Port號(連線到server)
        print("Connected!")
        ##################################
        # 開始傳輸
        print('Start sending data')
        #########連續傳輸會有問題 因為一起傳所以第一筆資料會把所有資料都吃掉!!(應該需要交替等待 傳完之後才能傳下一個)
        for i in range(3):
            imgFile = open(share_name[i], "rb") 
            while True:
                imgData = imgFile.readline(512)
                socket02.send(imgData)
                if not imgData: # 讀完檔案結束迴圈
                    time.sleep(0.1)
                    socket02.send(b'end\n')  # 發送 \n做為結束提示
                    break  
            imgFile.close()

            print(share_name[i] + " is sent.") 
            response = socket02.recv(4)
            print(response)
            if response == b'get!': # 收到接收完成才往下
                continue
            ###################################
        '''等待同步'''
        print("Waiting...")
        time.sleep(2)
        '''等待同步'''
        for i in range(3):
            jsonFile = open(keyname[i], 'rb')
            while True:
                jsonData = jsonFile.readline(512)  # 接收遠端主機傳來的數據
                socket02.send(jsonData)
                if not jsonData:
                    time.sleep(0.005)
                    socket02.send(b'end\n')  # 發送 \n做為結束提示
                    break  # 讀完檔案結束迴圈
            jsonFile.close()
            print('Key4share' + str(i+1) + ' is sent.')

            response = socket02.recv(512)
            print(response)
            if response == b'get!': # 收到接收完成才往下
                continue     
        print('Transmit end')
        ##################################
        socket02.close()  # 關閉
        print('client close')
    except:
        print("Failure. Please reconnect.")
        return 

if __name__ == '__main__':
    addr = input("enter HOST,PORT: ").split(",")
    index = [1,2,4]
    keyname = []
    share_name = []
    for i in range(3):
        keyname.append("Key4share" + str(index[i]) + ".json")
        share_name.append("512_lena_share" + str(index[i]) + ".png")
    sender(addr, share_name, keyname)