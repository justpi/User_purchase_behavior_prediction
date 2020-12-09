import pandas as pd
import data_process
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import mymodels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,Adadelta,SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn


trainfile = './data/train.csv'
testfile = './data/test.csv'

if __name__ == '__main__':

    # 1. 导入数据
    traindata = pd.read_csv(trainfile)
    testdata = pd.read_csv(testfile)

    # 2. 数据处理

    # 2.1 数据信息查看
    # 训练集
    # print("=============== train set ===============")
    # data_process.basic_eda(traindata)
    # #测试集
    # print("=============== test set ===============")
    # data_process.basic_eda(testdata)
    # 2.2 数据清洗
    traindata = data_process.cleaner(traindata.iloc[:5000])
    # print(traindata['event_type'].unique())
    # print(traindata['category_code'])
    # print(traindata['brand'])
    # print(traindata['price'].loc[traindata['price'] == traindata['price'].min()])
    # print(traindata['price'] .loc[traindata['price'] == traindata['price'].max()])
    # plt.hist(traindata['price'], 50,)
    # plt.show()
    # print(traindata['product_id'].loc[traindata['price'] < 0].unique())
    # print("===============  ===============")
    # print(traindata[['user_id','product_id','price','event_type']].loc[traindata['product_id'] == 5716855])
    # print("===============  ===============")
    # print(traindata[['user_id','product_id','price','event_type']].loc[traindata['product_id'] == 5716857])
    # print("===============  ===============")
    # print(traindata[['user_id','product_id','price','event_type']].loc[traindata['user_id'] == 4448])
    # print(traindata[['user_id', 'product_id', 'price', 'event_type']].loc[traindata['user_id'] == 14302])

    # data_process.basic_eda(traindata)

    # trainset = data_process.construct_feature(traindata)
    # train_X = trainset.drop(axis=1, labels=['target'])
    # train_Y = trainset['target']
    # x_train, x_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.3)
    # traindata['brand'] = LabelEncoder().fit_transform(traindata['brand'])
    # print(traindata['brand'].head())
    # trainset = pd.read_csv('./data/features.csv', )



    # 从这才是神经网络
    trainset = data_process.neural_pre_proces(traindata)
    trainset = data_process.myDataset(trainset)

    # trainloader = DataLoader(trainset,batch_size=8,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = mymodels.Net()
    net = net.to(device)
    # net.cuda()
    net.zero_grad()
    net.apply(mymodels.weights_init)
    # print(net)
    # 定义损失函数
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCELoss().cuda()
    # 定义优化器
    optimizer = Adam(net.parameters(), lr= 0.01,)
    # optimizer = SGD(net.parameters(), lr=0.01)
    # scheduler = MultiStepLR(optimizer, milestones=[2000, 3000], )
    epochs = 4
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(trainset.__len__()):

            inputs, labels = trainset.__getitem__(i)
            inputs = inputs.to(device)

            # inputs= inputs.cuda()
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # scheduler.step()
            optimizer.step()
            running_loss += loss.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                parm = {}
                for name, parameters in net.named_parameters():
                    # print(name, ':', parameters.size())
                    parm[name] = parameters.detach().cpu().numpy()
                # print(parm['fc2.weight'])
    PATH = './mynet.pth'
    torch.save(net.state_dict(), PATH)