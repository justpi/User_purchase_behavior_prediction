import pandas as pd
import datetime



def basic_eda(df):  #数据的基本信息
    print("----------TOP 5 RECORDS--------")
    print(df.head(5))
    print("----------INFO-----------------")
    print(df.info())
    print("----------Describe-------------")
    print(df.describe())
    print("----------Columns--------------")
    print(df.columns)
    print("----------Data Types-----------")
    print(df.dtypes)
    print("-------Missing Values----------")
    print(df.isnull().sum())
    print("-------NULL values-------------")
    print(df.isna().sum())
    print("-----Shape Of Data-------------")
    print(df.shape)


def cleaner(df):
    """
    对数据的每列进行清洗，空值处理，异常值处理，另外对价格单位化
    :param df: 待清洗的数据
    :return: 清洗之后的数据
    """
    columns = ['event_time', 'event_type', 'product_id', 'category_id',
               'category_code', 'brand', 'price', 'user_id', 'user_session']
    #event_time
    df['event_time'] = pd.to_datetime(df['event_time'])
    # event_type
    # na
    #product_id
    # na
    #category_id
    # na
    #category_code
    # na
    #brand
    df['brand'].fillna('not_brand', inplace=True)
    #price
    # 这些价格中有500余份为0，还有两个产品为负，分别是5716855和5716857，这两个产品只有两份记录，分别是4448和14302买的，先不做处理
    #user_id
    # na
    #user_session
    # 有38个缺失值，对这38个缺失值按照上一个session填充
    df['user_session'].fillna(method='ffill', inplace=True)

    return df



def construct_feature(df):
    """
    构造特征，形成特征--目标值
    :param df: 待构造数据集
    :return: 构造完成的数据集
    """
    # user_id
    # product_id
    # category_id
    # price
    # brand
    # continu_visit
    # stay_time
    # view
    # cart
    # remove_from_cart
    # pre_1_product_id
    # pre_1_product_behav
    # pre_1_product_smi
    # pre_2_product_id
    # pre_2_product_behav
    # pre_2_product_smi
    # pre_3_product_id
    # pre_3_product_behav
    # pre_3_product_smi

    # # 方法一：按列构造
    # # No.1 product_id和user_id   这两个不算特征，加入便于索引
    #
    # No.2 category_id
    # category为数值类型，但其值过大，将其缩小1e-16
    df['category_id'] = df['category_id'] * 1e-16   #此处有类型转换 int64 -> float64
    #
    # No.3 price
    # 连续数值类型，做一下特征缩放就ok
    # df['price'] = (df['price'] - df['price'].mean())/(df['price'].max() - df['price'].min())    #最大最小缩放
    df['price'] = (df['price'] - df['price'].mean()) / df['price'].std()    #标准归一化
    # # No.4 brand
    # # 字符类型，需做one-hot编码
    # df = df.join(pd.get_dummies(df.brand))
    # # No.5 user_session
    # # 利用session构造一个特征，判断是不是连续访问
    # df['continu_visit'] = 0
    # for i, row in df.iterrows():
    #     if i > 0:
    #         if row['user_session']  == df.iloc[i - 1]['user_session']:
    #             df['continu_visit'][i] = 1
    #
    # # No.6 stay_time
    # # 利用event_time列构造用户停留在商品上的时间
    # df['stay_time'] = 0.0
    # # for i, row in df.iterrows():

    # 方法二：按行构造
    dataset = pd.DataFrame(None, columns=['user_id', 'product_id', 'category_id', 'price', 'brand', 'continu_visit',
                                          'stay_time', 'view', 'cart', 'remove_from_cart', 'pre_1_product_behav',
                                          'pre_1_product_smi', 'pre_2_product_behav', 'pre_2_product_smi',
                                          'pre_3_product_behav', 'pre_3_product_smi', 'target'],
                           index=[i for i in range(len(df['product_id']))])
    df1 = df.groupby(['user_id', 'product_id'])

    k = 0   #设置一个指示数，用于指示数据在表中存放的位置
    for user in df['user_id'].unique():
        pros = df['product_id'].loc[df['user_id'] == user].unique()


        for pro in pros:
            sub_data = df1.get_group((user, pro))
            #构建商品特征
            # No.1 -- 5
            dataset.iloc[k]['user_id'] = user
            dataset.iloc[k]['product_id'] = pro
            dataset.iloc[k]['category_id'] = list(df[df['product_id'] == pro].category_id)[0]
            dataset.iloc[k]['price'] = list(df[df['product_id'] == pro].price)[0]
            dataset.iloc[k]['brand'] = list(df[df['product_id'] == pro].brand)[0]
            #No.6 continu_visit 需要对比前后两行session是否相同
            dataset.iloc[k]['continu_visit'] = 0    #初始化为0,下同
            #No.7 停留时间
            dataset.iloc[k]['stay_time'] = 0
            #No.8 view的次数
            dataset.iloc[k]['view'] = 0
            # No.9 cart的次数
            dataset.iloc[k]['cart'] = 0
            # No.10 remove_from_cart的次数
            dataset.iloc[k]['remove_from_cart'] = 0
            # No.11 pre_1_behav
            dataset.iloc[k]['pre_1_behav'] = None
            dataset.iloc[k]['pre_2_behav'] = None
            dataset.iloc[k]['pre_3_behav'] = None
            #No.12 target
            dataset.iloc[k]['target'] = 0

            start_time = None
            end_time = None
            sub_data.reset_index(inplace=True)
            # print(sub_data)
            for i, row in sub_data.iterrows():  #构建每一个user_id和product_id下的用户行为特征
                #stay time 计算
                if i == 0:
                    start_time = row['event_time']
                    end_time = row['event_time']
                if row['event_type'] != 'purchase':
                    end_time = row['event_time']
                    # tartget赋值
                    dataset.iloc[k]['target'] = 1
                #view,cart, remove_from_cart计算
                if row['event_type'] == 'view':
                    dataset.iloc[k]['view'] = dataset.iloc[k]['view'] + 1
                if row['event_type'] == 'cart':
                    dataset.iloc[k]['cart'] = dataset.iloc[k]['cart'] + 1
                if row['event_type'] == 'remove_from_cart':
                    dataset.iloc[k]['remove_from_cart'] = dataset.iloc[k]['remove_from_cart'] + 1




            dataset.loc[k]['stay_time'] = (end_time - start_time).total_seconds()



            k = k + 1
    dataset.dropna(axis=0, how='all', inplace=True) #删除nan的行

    # 字符列onehot编码
    dataset = dataset.join(pd.get_dummies(dataset.brand))

    #先删除部分未特征处理的列
    dataset.drop(axis=1, labels=['brand','continu_visit', 'pre_1_product_behav', 'pre_1_product_smi', 'pre_2_product_behav',
                                 'pre_2_product_smi', 'pre_3_product_behav', 'pre_3_product_smi'], inplace=True)

    #类型转换
    dataset = dataset.astype('float64')
    dataset[['user_id', 'product_id']] = dataset[['user_id', 'product_id']].astype('int')

    return dataset


def mysubmission(data, origindata):
    """
    找到除用户已经购买的产品外最可能购买的产品
    :param data: 每个用户的每个产品购买概率预测结果
    :param origindata: 原始数据
    :return: 每个用户最可能购买哪个产品
    """


    submission = pd.DataFrame(data['user_id'].unique(), columns=['user_id'])
    submission['product_id']= None
    group_data = data.groupby(['user_id'])

    for user in data['user_id'].unique():

        product_data = group_data.get_group(user)
        origin_pro_data = origindata['product_id'].loc[(origindata['user_id'] == user) & (origindata['event_type'] != 'purchase')]
        res_data = product_data[product_data['product_id'].isin(origin_pro_data)]
        print(res_data)
        res_data.sort_values(by='purchase_1', ascending=False, inplace=True)
        submission['product_id'].loc[submission['user_id'] == user] = res_data['product_id']

    print(submission.head())
    print(submission.describe())
    # submission['product_id'] = submission['product_id'].astype('int')
    return submission
