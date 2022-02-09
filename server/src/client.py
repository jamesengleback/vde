#!/home/u0/miniconda3/envs/enz/bin/python
import sys
import socket
import os
import os.path as osp
import time
import json
import shutil

def client(msg, port, host='localhost'):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.connect((host,port))
        client.send(msg.encode())
        response = client.recv(1024).decode()
        return response

            
def main(args):
    sequences = ['ANDSAJKLNTGUAILGBDASJKL', 'NGJFSIKOLHBAYUIEOHBATKUGFL']
    for i in sequences:
        r = client(i, 8000)
        print(r)


if __name__ == '__main__':
    main(sys.argv[1:])
