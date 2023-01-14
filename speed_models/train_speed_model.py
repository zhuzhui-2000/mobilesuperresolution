import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from SpeedModel import ConvBlockModel
import argparse
import os
import numpy as np

class SpeedLoss(nn.Module):

    def __init__(self, scale=1):
        super(SpeedLoss, self).__init__()
        self.scale = scale

    def forward(self, speed, target, gamma=1, method='clamp'):
        assert method in ['mse', 'clamp', 'abs']
        diff = (speed - target) / self.scale
        loss = pow(diff,2) / target
        return loss.mean() * gamma

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="huawei_p30", help='device')
    parser.add_argument('--processor', type=str, default="cpu", help='device')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--train_device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--step', type=int, default=200)

    args = parser.parse_args()
    if args.train_device == 'cuda':
        train_device = args.train_device
    else:
        train_device = 'cpu'
    lr = args.lr
    epoch = args.epoch
    device = args.device
    processor = args.processor
    batch_size = args.batch_size
    data_dir = os.path.join("speed_data",device)
    data_file = processor+".npy"
    data_file = os.path.join(data_dir,data_file)
    data = np.load(data_file)

    data_x = torch.tensor(data[:,:-1], dtype=torch.float)
    data_y = torch.tensor(data[:,-1:], dtype=torch.float)
    print(data_x.shape,torch.max(data_y))
    
    data_set = TensorDataset(data_x,data_y)
    train_size = int(len(data_set)*0.8)
    test_size = int(len(data_set)-train_size)
    train_dataset, test_dataset = random_split(data_set,[train_size,test_size])
    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    model = ConvBlockModel(num_feat=data_x.shape[1])
    
    loss_fn = SpeedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.step,gamma=0.3,verbose=False)
    model.to(device=train_device)

    for i in range(epoch):
        model.train()
        for step,  [x, speed] in enumerate(train_dataloader):
            
            x = x.to(train_device)
            speed = speed.to(train_device)
            predict_speed = model(x)
            loss = loss_fn(predict_speed, speed).to('cpu')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if ((batch_size * step) % 100) == 0:
                print("Epoch: {}, num: {}, Train_Loss: {}".format(i, batch_size * step, loss.item()))
        
        model.eval()
        total_loss = []
        total_step = 0
        for step, [x, speed] in enumerate(test_dataloader):
            x = x.to(train_device)
            speed = speed.to(train_device)
            predict_speed = model(x)
            loss = loss_fn(predict_speed, speed).to('cpu')
            total_loss.append(loss.item())
            total_step += 1
        print("Epoch: {}, Test_Loss: {}".format(i, sum(total_loss)/total_step))

    dir_path = os.path.join('weights',device)
    if os.path.exists(dir_path)==0:
        os.mkdir(dir_path)
    model_path = os.path.join(dir_path,processor)
    if os.path.exists(model_path)==0:
        os.mkdir(model_path)
    model_file_path = os.path.join(model_path,processor)
    torch.save(model.state_dict(),model_file_path+'.pt')


