import pandas as pd
import data_process



trainfile = './data/train.csv'
testfile = './data/test.csv'

if __name__ == '__main__':

    # 1. 导入数据
    traindata = pd.read_csv(trainfile)
    testdata = pd.read_csv(testfile)

    # 2. 数据处理

    # 2.1 数据信息查看
    # 训练集
    print("=============== train set ===============")
    data_process.basic_eda(traindata)
    #测试集
    print("=============== test set ===============")
    data_process.basic_eda(testdata)
    # 2.2 数据清洗
    traindata = data_process.cleaner(traindata)
