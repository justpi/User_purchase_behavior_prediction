import pandas as pd
import data_process
import matplotlib.pyplot as plt

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
    traindata = data_process.cleaner(traindata.iloc[:1000])
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

    trainset = data_process.construct_feature(traindata)
    print(trainset['continu_visit'])
