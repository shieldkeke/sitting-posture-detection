import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import time

class Drawer:
    def __init__(self, opt, row = 2, col = 2) -> None:
        self.row = row
        self.col = col
        self.mat = []
        self.mat_input = []
        self.hot = []
        self.config = opt
        self.data = np.zeros([self.row*self.col, 1, 3])
        self.model = torch.load('w.pt', map_location='cpu')
        # self.model = torch.load('w.pt')
        self.model.eval().to(opt.device)
        self.classes = ['sit', 'right_up', 'left_up']
        img_trans = [
            transforms.Resize((150, 150), cv2.INTER_AREA),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.trans = transforms.Compose(img_trans)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        

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

    def draw(self, data, IF_WAIT = False, IF_SAVE = False, path = 'right_up.txt'):

        if len(data)<self.row * self.col:
            return

        self.data = data
        self.mat = np.zeros([self.row, self.col,3])
        self.mat_input = np.zeros([self.row, self.col,3])
        self.hot = np.zeros([self.row, self.col,1])

        for i in range(self.row):
            for j in range(self.col):
                self.mat[i][j] = self.color(int(data[i * self.col + j]))
                self.mat_input[i][j] = self.color(int(data[i * self.col + j]), True)
                self.hot[i][j] = min(int(data[i * self.col + j]), 5) * 255 // 5 

        self.mat = self.mat[:, :, ::-1]

        img = cv2.resize(self.mat, [750, 500],  interpolation=cv2.INTER_AREA)
        hot = cv2.resize(self.hot.astype(np.uint8), [750, 500],  interpolation=Image.BICUBIC)
        hot = cv2.applyColorMap(hot, cv2.COLORMAP_JET)
        
        s = self.predict()
        cv2.putText(img, s, (375, 200), self.font, 1, (240, 240, 240), 1)
        cv2.imshow('img', img)
        cv2.imshow('hot', hot)
        

        ## save
        if IF_SAVE:
            f = open(path, 'a')
            f.write(self.data+'\n') 
        
        if IF_WAIT:
            cv2.waitKey(0)
        else:
            cv2.waitKey(500)

    def getInput(self, img):
        img = Image.fromarray(img.astype(np.uint8))
        img = self.trans(img).to(self.config.device)
        return img

    def predict(self):

        start = time.time()
        img = self.getInput(self.mat_input)
        img = torch.unsqueeze(img, 0)
        output = self.model(img)
        _, pred = output.topk(1, 1, True, True)
        end = time.time()

        print(self.classes[pred[0][0]], round((end - start)*1000, 2),'ms')
        return self.classes[pred[0][0]] + f' {round((end - start)*1000, 2)}ms'

if __name__ == "__main__":
    d = Drawer()
    d.draw("0101")