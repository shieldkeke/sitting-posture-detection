from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import torch

class pressureDataPosture(Dataset):
        
    def __init__(self, config, path = 'data/'):
        
        self.files = ['sit.txt', 'right_up.txt', 'left_up.txt']
        self.datas = []
        self.inputs = []
        self.labels = []
        self.mats = []
        self.hots = []
        self.classes = ['sit', 'right_up', 'left_up']
        self.config = config
        self.row = 10
        self.col = 15

        img_trans = [
            transforms.Resize((150, 150), cv2.INTER_AREA),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.trans = transforms.Compose(img_trans)

        print("loading data...")


        for idx, file in enumerate(self.files):
            print("opening: " + path + file)
            f = open(path + file ,encoding = "utf-8")
            rawData = f.readlines()
            for data in rawData:
                mat = np.zeros([self.row, self.col,3])
                mat1 = np.zeros([self.row, self.col,3])
                hot = np.zeros([self.row, self.col,1])
                for i in range(self.row):
                    for j in range(self.col):
                        mat[i][j] = self.color(int(data[i * self.col + j]))
                        mat1[i][j] = self.color(int(data[i * self.col + j]), True)
                        hot[i][j] = min(int(data[i * self.col + j]), 5) * 255 // 5 

                mat = mat[:, :, ::-1]
                self.datas.append(data)
                self.inputs.append(self.getInput(mat1))
                self.mats.append(mat)
                self.hots.append(hot)
                self.labels.append(torch.LongTensor([idx]).to(self.config.device))

            f.close()

    def color(self, k, for_train = False):
        if not for_train:
            # co1 = [70/255, 130/255, 180/255]  #dark blue
            co2 = [135/255, 206/255, 235/255] #light blue
            co1 = [3/255, 41/255, 81/255]
        else:
            co2 = [255, 255, 255]
            co1 = [0, 0, 0]
        scales = 9
        return [(k*co1[i]+(scales-k)*co2[i])/scales for i in range(3) ]

    def getInput(self, img):
        img = Image.fromarray(img.astype(np.uint8))
        img = self.trans(img).to(self.config.device)
        return img

    def show(self):
        
        for idx, img in enumerate(self.mats):
            img = cv2.resize(img, [750, 500],  interpolation=cv2.INTER_AREA)
            hot = cv2.resize(self.hots[idx].astype(np.uint8), [750, 500],  interpolation=cv2.INTER_AREA)
            hot = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
            
            cv2.imshow('img', img)
            cv2.imshow('hot', hot)
            cv2.waitKey(0)
    
    def getData(self):
        return self.datas

    def __getitem__(self, index):
        return [self.inputs[index], self.labels[index]]

    def __len__(self):
        return len(self.datas)

if __name__ == '__main__':
    from train import Config
    opt = Config('test')
    d = pressureDataPosture(opt)
    print(len(d))
    d.show()
