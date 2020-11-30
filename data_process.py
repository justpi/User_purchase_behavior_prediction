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
    # 这些价格中有500余份为0，还有两个产品为负，分别是5716855和5716857，这两个产品只有两份记录，分别是



    return df


