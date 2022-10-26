import time
import bluetooth as bt
import threading
from datetime import datetime as t
    
class BlueTooth:
    def __init__(self):
        self.dev_list = []
        self.find_thread = threading.Thread(target=self.find_dev)
        self.data_buf = ""
        self.socket = bt.BluetoothSocket(bt.RFCOMM)

    def find_dev(self):
        devs = bt.discover_devices(lookup_names=True)
        for (addr,name) in devs:
            if addr not in self.dev_list:
                print("[NAME]:" + str(name))
                print("[MAC]:" + str(addr))
                self.dev_list.append([name, addr])
        time.sleep(5)

    def find(self):
        self.find_thread.start()

    def connect_by_name(self, name):
        for [n,addr] in  self.dev_list:
            if str(n)==str(name):
                self.connect_by_addr(addr)
                break

    def connect_by_addr(self, addr):
        self.socket.connect((addr, 1))
        print("successfully connected to "+addr)

    def receive(self):
        ret = ""
        data = self.socket.recv(1024)
        self.data_buf += data.decode()
        if '\n' in data.decode() or '\r' in data.decode():
            print(t.now().strftime("%H:%M:%S")+"->" + self.data_buf.strip() )
            ret = ''.join(ch for ch in self.data_buf if ch.isalnum())
            self.data_buf = ""
        return ret

if __name__ == "__main__":
    b = BlueTooth()
    b.connect_by_addr("98:D3:71:FE:5C:23")
    while True:
        b.receive()