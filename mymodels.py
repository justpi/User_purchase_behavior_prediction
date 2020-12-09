# from keras import Sequential
# from keras.layers import LSTM,Dense,Dropout,Activation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost
import torch
import torch.nn as nn
import torch.nn.functional as F

# 随机森林分类模型
def RandomFmodel(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100,)
    clf.fit(x_train,y_train)
    predict = clf.predict(x_test)
    print(classification_report(y_test, predict))
    print('accuracy:', accuracy_score(y_test, predict))
    return clf

#梯度提升树
def GBTmodel(x_train, y_train,x_test, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, verbose=1)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print(classification_report(y_test, predict))
    print('accuracy:', accuracy_score(y_test,predict))
    return clf


#LSTM模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(160, 32, 2) #此处7为特征维数
        # self.conv2 = nn.Conv1d(32, 64, 3)
        self.gru = nn.GRU(input_size=4. , hidden_size=64, num_layers=2,)

        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        # x = F.max_pool1d(F.relu(self.conv(x)), 2)
        x = F.tanh(self.conv1(x))
        # x = F.tanh(self.conv2(x))

        x,(h_n,h_c) = self.gru(x)
        # print(x.size(),h_n.size(),h_c.size())
        # x = h_n.view(self.num_flat_features(h_n))
        # x = x.view(1, self.num_flat_features(x))
        # print(x.size())
        x = F.tanh(self.fc1(x[:,-1,:]))
        # print(x.size())
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def weights_init(m):
 if isinstance(m, nn.Conv1d):
  # nn.init.constant_(m.weight, 1)
  m.weight.data.normal_(0, 0.05)
  nn.init.constant_(m.bias, 0)
 elif isinstance(m, nn.GRU):
  # nn.init.constant_(m.weight_ih_l0, 1)
  m.weight_ih_l0.data.normal_(0, 0.02)
  # nn.init.constant_(m.weight_hh_l0, 1)
  m.weight_hh_l0.data.normal_(0, 0.03)
  nn.init.constant_(m.bias_ih_l0, 0)
  nn.init.constant_(m.bias_hh_l0, 0)
  # nn.init.constant_(m.weight_ih_l1, 1)
  m.weight_ih_l1.data.normal_(0, 0.05)
  # nn.init.constant_(m.weight_hh_l1, 1)
  m.weight_hh_l1.data.normal_(0, 0.03)
  nn.init.constant_(m.bias_ih_l1, 0)
  nn.init.constant_(m.bias_hh_l1, 0)
 elif isinstance(m, nn.Linear):
  nn.init.constant_(m.weight, 1)
  nn.init.constant_(m.bias, 0)


#xgboost
def XGCmodel(x_train, y_train, x_test, y_test):


    clf = xgboost.XGBClassifier(n_estimators=300, max_depth=6)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print(classification_report(y_test, predict))
    print('accuracy:', accuracy_score(y_test, predict))
    return clf