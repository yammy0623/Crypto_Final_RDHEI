# https://gist.github.com/endless3cross3/dabc1ef3783967c73a2ee0ed4c50a3c1
import socket
import time
# Reciever
def receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect( ( "www.google.com", 80 ) )
    #把這個PORT設定為可以重複用
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    (host, port) = s.getsockname()
    s.close()

    print("Server's ip : {}  Port : {}".format(host, port))
    print("Copy this to client : {},{}".format(host, port))
    address = (host, port)
    socket01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET:默認IPv4, SOCK_STREAM:TCP
    socket01.bind(address)  # 讓這個socket要綁到位址(ip/port)
    socket01.listen(1)  # listen(backlog) (等待建立連線，1是指最多可接受一個socket串接) 
    print('Socket Startup')
    try: 
        conn, addr = socket01.accept()  
        # conn是新的套接字對象，可以用來接收和發送數據。address是連接客戶端的地址
        print('Connected by', addr)
        ##################################################
        # 開始接收
        print('Start receiving data')
        for i in range(3):
            imgFile = open('receive_share' + str(i+1) + '.png', 'wb')  # 開始寫入圖片檔
            while True:
                imgData = conn.recv(512)  # 接收遠端主機傳來的數據
                if imgData == b'end\n':
                    print(imgData)
                    conn.send(b'get!') # 告知已接收完成
                    break  # 讀完檔案結束迴圈
                imgFile.write(imgData)
            imgFile.close()
            time.sleep(0.05)
            print('Share' + str(i+1) + ' is received.')
            ##################################
        '''等待同步'''
        print("Waiting...")
        time.sleep(2)
        '''等待同步'''
        for i in range(3):
            jsonFile = open('Key4share' + str(i+1) + '.json', 'wb')
            while True:
                jsonData = conn.recv(512)  # 接收遠端主機傳來的數據
                if jsonData == b'end\n':
                    conn.send(b'get!') # 告知已接收完成
                    break  # 讀完檔案結束迴圈
                jsonFile.write(jsonData)
            jsonFile.close()
            print('Key4share' + str(i+1) + ' is received.')
        ##################################################
        print('Receive end')
        conn.close()  # 關閉
        print('Connection close')
        socket01.close()
        print('server close')
    except:
        print("Failure. Please reconnect.")

if __name__ == '__main__': 
    receiver()