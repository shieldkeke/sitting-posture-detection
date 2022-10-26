from drawer import Drawer
from blue_tooth import BlueTooth
from dataset import pressureDataPosture
import random
if __name__ == "__main__":
    
    from train import Config
    c = Config('predict')
    b = BlueTooth()
    d = Drawer(c, 10, 15)# row * col
    
    USE_BLUE = False
    if USE_BLUE:
        b.connect_by_addr("98:D3:71:FE:5C:23")
        while True:
            ret = b.receive()
            if ret != "":
                d.draw(ret)
    else:
        datas = pressureDataPosture(c).getData()
        random.shuffle(datas)
        for data in datas:
            d.draw(data, True)