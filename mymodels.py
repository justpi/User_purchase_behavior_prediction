# from keras import Sequential
# from keras.layers import LSTM,Dense,Dropout,Activation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost

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
def CNN_LSTMmodel(x_train, y_train, x_test, y_test):
    clf =None

    return clf


#xgboost
def XGCmodel(x_train, y_train, x_test, y_test):
    clf = xgboost.XGBClassifier()
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print(classification_report(y_test, predict))
    print('accuracy:', accuracy_score(y_test, predict))
    return clf
