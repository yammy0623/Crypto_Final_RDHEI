# Reversible Data Hiding in Encrypted Image via Secret Sharing Based on GF(2^8)
###### Experiment result records in https://hackmd.io/@vekMh5uNRK-ZfmBnFj1BgQ/Sk3qFHEPn

# Environment
```
python                    3.8.0
numpy                     1.24.3
opencv-python             4.7.0.72
pillow                    9.4.0
matplotlib                3.7.1
socket programing (TCP)
```
# Execution
In our program, sender and receiver are the main two characters. 
- Sender takes charge of image encryption, Shamir's secret sharing and data hiding. 
The program creates shares and keys and transmit to receiver with the TCP transportation.
- Receiver collects three shares and three keys to extract the secret message and recover the image.

## As sender
```
python main.py
```
1. Determine the identity of user first (Sender/ Receiver)
```
What is your identity?(0: sender, 1: receiver) 0
```
2. Input the file name
```
Input file name: brain.png
```
3. Determine the amount of shares to create
```
How many shares do you want to create?(more than 3) 5
```
4. Insert the secret
```
Your secret (no more than 1048576): 11234
```
5. Shares are created
```
Preparing secret sharing...
---- Image scrambling: Key-based ----
---- Shamir Secret Sharing ----
---- Data Embedding ----
Your secret message:  11234
Additional_data 11234
----- key saving -----
Shares are created!!
```
Five shares are created.


Five keys are created.

6. Choose any three shares to transmit


```
Choose the shares to receivers
Input share index: 1
Input share index: 3
Input share index: 5
```

6. Connect with the receiver and send data automatically.
```
Connect to server...
Enter Server's HOST,PORT: 192.168.0.109,4081
Connected!
```
```
Start sending data
brain_share1.png is sent.
b'get!'
brain_share3.png is sent.
b'get!'
brain_share5.png is sent.
b'get!'
Waiting...
Key4share1 is sent.
b'get!'
Key4share2 is sent.
b'get!'
Key4share3 is sent.
b'get!'
Transmit end
client close
```

## As a receiver
```
python main.py
```
1. Determine the identity of user first (Sender/ Receiver)
```
What is your identity?(0: sender, 1: receiver) 1
```
2. Generate the IP and port to sender
```
Server's ip : 192.168.0.109  Port : 4081
Copy this to client : 192.168.0.109,4081
```
3. Collect the data from sender
```
Socket Startup
Connected by ('192.168.0.109', 4082)
Start receiving data
b'end\n'
Share1 is received.
b'end\n'
Share2 is received.
b'end\n'
Share3 is received.
Waiting...
Key4share1 is received.
Key4share2 is received.
Key4share3 is received.
Receive end
Connection close
server close
```
Three shares are collected.


Three keys are collected.

3. Data Extraction and Image recovery
```
The keys you have:  ['Key4share1.json', 'Key4share2.json', 'Key4share3.json']
---- read the key ----
['Key4share1.json', 'Key4share2.json', 'Key4share3.json']
['receive_share1.png', 'receive_share2.png', 'receive_share3.png']
The shares you have:  ['receive_share1.png', 'receive_share2.png', 'receive_share3.png']
---- load sharing ----
---- extracting data ----
Your secret message:  11234
---- secret sharing recover ----
```
Recover Image

