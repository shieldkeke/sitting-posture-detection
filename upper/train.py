import torch.nn as nn
import torch
import os
import numpy as np
from tqdm import tqdm 

from dataset import pressureDataPosture
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def accuracy(output, target, topk=(1, )):       
    # output.shape (batch_size, num_classes), target.shape (bs, )

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eval():
    pass

class Config(object):
    
    def __init__(self, name):

        self.model_name = name                                                        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                      
        self.num_epochs = 100                                          
        self.batch_size = 32
        self.n_cpu = 0                                          
        self.lr = 5e-5     
        self.weight_decay = 5e-4
        self.class_num = 3                            

if __name__ == "__main__":
    opt = Config("soft-01")

    ## data

    t_d = pressureDataPosture(opt)
    train_dataset = DataLoader(t_d, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    ## result
    log_path = 'result/logs/%s' % opt.model_name
    model_path = 'result/models/%s' % opt.model_name

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = SummaryWriter(log_dir=log_path)

    ## model

    # mobilenet

    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, opt.class_num)

    # resnet
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, opt.class_num)

    model.train().to(opt.device)
    loss_f = torch.nn.CrossEntropyLoss()

    ## init rand
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    ## optimizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


    #### start to train ####
    print('Start to train ...')

    total_step = 0
    eval_step = 0

    for epoch in range(opt.num_epochs):
        print('epoch :', epoch)
        bar = enumerate(train_dataset)
        length = len(train_dataset)
        bar = tqdm(bar, total=length)
        total_acc = 0
        total_loss = 0
        total_batch = 0
        for i, batch in bar:
            total_step += 1
            total_batch += 1
            inputs = batch[0]
            labels = batch[1][:,0].long()
            outputs = model(inputs)

            model.zero_grad()
            loss = loss_f(outputs, labels) # not one hot
            acc = accuracy(outputs, labels, topk = [1])[0]
            total_acc = total_acc + acc.to('cpu').detach().numpy()[0]
            loss.backward()
            total_loss = total_loss + loss.item()
            optimizer.step()

            logger.add_scalar('loss', loss.item(), total_step)
            logger.add_scalar('acc', acc.to('cpu').detach().numpy(), total_step)
            
        total_loss = total_loss / total_batch
        total_acc = total_acc / total_batch
        print(f'Train total Loss: {total_loss},  Train total Acc {total_acc}')
        torch.save(model, model_path + f'/{epoch}-{total_loss}-{total_acc}.pt') #save whole model