import pandas as pd



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
    df['brand'].fillna('Not_brand', inplace=True)
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
    # # No.2 category_id
    # # category为数值类型，但其值过大，将其缩小1e-16
    # df['category_id'] = df['category_id'] * 1e-16   #此处有类型转换 int64 -> float64
    #
    # # No.3 price
    # # 连续数值类型，做一下特征缩放就ok
    # # df['price'] = (df['price'] - df['price'].mean())/(df['price'].max() - df['price'].min())    #最大最小缩放
    # df['price'] = (df['price'] - df['price'].mean()) / df['price'].std()    #标准归一化
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
                                          'pre_3_product_behav', 'pre_3_product_smi'])



    return dataset

