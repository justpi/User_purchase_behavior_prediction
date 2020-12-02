import pandas as pd
import data_process
import mymodels
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")
trainfile = './data/train.csv'
testfile = './data/test.csv'

if __name__ == '__main__':

    # 1. 导入数据
    print('=================1. 导入数据=================')
    traindata = pd.read_csv(trainfile)
    testdata = pd.read_csv(testfile)
    # print('原始训练集大小：%s, 原始测试集大小： %s' %(traindata.shape, testdata.shape))
    #
    # # 2. 数据处理
    # print('=================2. 数据处理=================')
    # traindata = traindata.merge(testdata,how='outer')
    # print('合并后数据集大小: ', traindata.shape)
    # # 2.1 数据信息查看
    # # 训练集
    # # print("=============== train set ===============")
    # # data_process.basic_eda(traindata)
    # # #测试集
    # # print("=============== test set ===============")
    # # data_process.basic_eda(testdata)
    # # 2.2 数据清洗
    # print('=================2.2 数据清洗=================')
    # traindata = data_process.cleaner(traindata)
    # # 2.3 构造数据集
    # print('=================2.3 构造数据集=================')
    # trainset = data_process.construct_feature(traindata)
    # trainset.to_csv('./data/features.csv')
    # 2.4 分割数据集
    # 此两行为特征数据导入
    trainset = pd.read_csv('./data/features.csv',)
    trainset.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
    trainset.drop('stay_time', axis=1, inplace=True)
    trainset['user_id'] = trainset['user_id'].astype('int')
    trainset['product_id'] = trainset['product_id'].astype('int')
    #设置分割点
    print('=================2.4 分割数据集=================')
    split_point = trainset.loc[trainset['user_id'] == 53978].index.tolist()[0]
    print('分割点为：', split_point)
    testset = trainset.iloc[split_point:] #测试集
    trainset = trainset.iloc[:split_point]    #训练集和验证集
    print('训练集大小: %s ,测试集大小：%s' %(trainset.shape, testset.shape))
    train_X = trainset.drop(axis=1, labels=['target'])
    train_Y = trainset['target']
    x_train, x_val, y_train, y_val = train_test_split(train_X,train_Y, test_size=0.3)


    # 3. 拟合模型
    print('=================3 数据拟合=================')
    # 3.1 随机森林尝试
    # clf = mymodels.RandomFmodel(x_train.iloc[:, 2:], y_train, x_val.iloc[:, 2:], y_val)
    # 3.2 梯度提升树
    # clf = mymodels.GBTmodel(x_train.iloc[:, 2:], y_train, x_val.iloc[:, 2:], y_val)
    # 3.3 xgboost分类器
    clf = mymodels.XGCmodel(x_train.iloc[:, 2:], y_train, x_val.iloc[:, 2:], y_val)




    # 4. 结果预测
    print('=================4 结果预测=================')
    test_X = testset.drop(axis=1, labels=['target'])

    test_y = clf.predict_proba(test_X.iloc[:, 2:])

    test_y = pd.DataFrame(test_y, columns=['purchase_0', 'purchase_1'])
    test_y['user_id'] = test_X['user_id']
    sub = test_X
    # print(test_y['purchase_1'].shape)
    sub['purchase_1'] = None
    sub['purchase_1'].iloc[:] = test_y['purchase_1'].tolist()

    submission = data_process.mysubmission(sub, testdata)
    submission[['user_id', 'product_id']].to_csv('./data/submission.csv',index=None)
    print('================= 完成 =================')


