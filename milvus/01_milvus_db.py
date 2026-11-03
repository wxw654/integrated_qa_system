"""
    案例:
        演示通过pymilvus模块和 milvus向量数据库交互
    大白话理解: 代码建库
"""

# 1.导包
from pymilvus import MilvusClient, DataType, AnnSearchRequest,RRFRanker,WeightedRanker
import random

client = MilvusClient(url="http://localhost:19530")
client.using_database("milvus_demo")


# todo 1 定义数据库操作函数 ,处理连接和数据库管理
def operate_db():
    # 1.配置数据库连接
    # 1.1本地文件数据库连接
    # MilvusClient(uri="ai_milvus.db")
    # 1.2实际开发连接方式 : 连接本地milvus服务,远程生产服务器的IP地址
    # client = MilvusClient(url="http://localhost:19530")
    # 2.处理milvus_demo数据库 , 有就使用 ,没有就创建
    # 2.1获取数据库列表
    databases = client.list_databases()
    # 2.2判断数据库milvus_demo是否存在
    if "milvus_demo" not in databases:
        # 走这里,说明数据库不存在,就创建数据库
        client.create_database(db_name="milvus_demo")
        print("创建数据库成功!")
    # 2.3走这里,数据库milvus_demo存在, 使用
    client.using_database(db_name="milvus_demo")
    return client


# todo 2  collection集合的操作 -> milvus中创建表
"""
需求:  集合名字为:demo_v1  三个字段 id vector scalar  字段名
"""


def operate_collection():
    # 1.定义集合结构(相当于:表结构)
    # 参1:主键字段需要手动赋(不自动生成)
    # 参2:允许插入未在schema中定义的额外字段
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # 2.向shema中添加具体的字段
    # 2.1主键字段 id 64位整数 ,主键
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

    # 2.2添加向量字段  字段名: vector  字段类型:FLOAT_VECTOR  该向量的维度:5
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)

    # 2.3添加标量字段  varchar(255)
    schema.add_field(field_name="scalar", datatype=DataType.VARCHAR, max_length=256, description='标量字段')

    # 3.创建集合,并且定义集合名字为demo_v1
    client.create_collection(collection_name='demo_v1', schema=schema)

    # 4.打印提示
    print("创建milvus集合成功!")

def operate_index():
    # -------------------------场景1:为向量字段添加索引-------------------
    # 1.初始化索引参数配置对象 , 用于设置索引相关参数
    prepare_params = client.prepare_index_params()

    # 2.配置向量字段的索引信息
    # 参1: 要添加向量信息的字段名  参2:索引类型(FLAT,IVF_FLAT,IVF_SQ8,IVF_PQ,HNSW) 参3:向量相似度计算方法(余弦相似度)  参4:索引名称
    prepare_params.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')

    # 3.执行索引创建操作
    client.create_index(collection_name="demo_v1", index_params=prepare_params)
    print("*" * 40)

    res = client.list_indexes('demo_v1')
    print(f'索引信息:{res}')

    res2 = client.describe_index(collection_name='demo_v1',index_name='vector_index')
    print(f"索引信息:{res2}")

    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')

    # client.load_collection(collection_name="demo_v1")
    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')

    # client.release_collection(collection_name="demo_v1")
    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')


    # client.drop_index(collection_name="demo_v1",index_name='vector_index')
    #
    # prepare_params.add_index(field_name='scalar',index_name='scalar_index',index_type='')
    # client.create_index(collection_name='demo_v1',index_params=prepare_params)
    # print(f'索引信息:{client.list_indexes(collection_name="demo_v1")}')
    # res3 = client.describe_index(collection_name="demo_v1",index_name="scalar_index")
    # print(f'索引信息:{res3}')

def operate_entity():
#     --------------场景1:创建collection集合--------------
    # 1.直接创建集合,无需指定schema 简化方式
    # 参1:集合名称     参2:向量维度      参3:向量相似度计算方式(IP,内积)
    # client.create_collection(collection_name="demo_v2", dimension=5, metric_type="IP")

    # --------------场景2:向集合中添加数据--------------
    # 1.准备要添加的数据(实体列表)
    # 小结 :     id     vector     color :动态标量字段
    # data = [
    #     {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
    #                          0.9029438446296592], "color": "pink_8682"},
    #     {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
    #                          0.838729485096104], "color": "red_7025"},
    #     {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
    #                          0.20785793220625592], "color": "orange_6781"},
    #     {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
    #                          0.95791889146345], "color": "pink_9298"},
    #     {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
    #                          0.30337481143159106], "color": "red_4794"},
    #     {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
    #                          -0.1446277761879955], "color": "yellow_4222"},
    #     {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
    #                          -0.8984947637863987], "color": "red_9392"},
    #     {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
    #                          0.5378064918413052], "color": "grey_8510"},
    #     {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
    #                          -0.6140360785406336], "color": "white_9381"},
    #     {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
    #                          -0.6980531615588608], "color": "purple_4976"}
    # ]

    #2.执行insert操作
    # client.insert(collection_name="demo_v2",data=data)

    # 3.执行upsert操作(insert+update -> 如果数据存在就更新,没有就添加)
    # client.upsert(collection_name='demo_v2',data=data)
    # print("数据添加成功")


    client.create_partition(collection_name='demo_v2', partition_name='partitionA')


    data = [
        {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
                              0.6699215060604258], "color": "red_1202"},
        {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
                              -0.14152860714878068], "color": "blue_4150"},
        {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
                              -0.8351194181285713], "color": "orange_4590"},
        {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
                              -0.0010987671273892108], "color": "pink_9619"},
        {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
                              -0.7986720938400362], "color": "orange_4863"},
        {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
                              0.24448151140671204], "color": "orange_7984"},
        {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
                              0.15077052220816167], "color": "blue_9010"},
        {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
                              -0.33007998971889263], "color": "blue_4521"},
        {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
                              -0.7387383128324057], "color": "orange_2529"},
        {"id": 19, "vector": [0, 0, 0, 0,
                              0], "color": "red_9437"}
    ]

    client.upsert(collection_name='demo_v2',partition_name='partitionA',data=data)

    res = client.delete(collection_name="demo_v2",partition_name='partitionA',ids=[13,2])
    print(f"删除结果:{res}")

def operate_query():

    result1 = client.search(
        collection_name='demo_v2',
        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
        # 查询的向量列表   用户的文本转向量
        # limit=2,
        output_fields=['id', 'vector', 'color'],
        search_params={'metric_type': "IP"}  # 内积
    )
    print(f"result1:{result1}")

    result2 = client.search(
        collection_name='demo_v2',
        data=[
            [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
            [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
        ],
        # 查询的向量列表   用户的文本转向量
        limit=2,
        output_fields=['id', 'vector', 'color'],
        search_params={'metric_type': "IP"}  # 内积
    )
    print(f"result2:{result2}")

    result3=client.search(
        collection_name="demo_v2",
        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
        limit=2,
        search_params={'metric_type': "IP"},
        partition_names=['partitionA']

    )
    print(f"result3:{result3}")


    result4 = client.search(
        collection_name='demo_v2',
        data=[
            [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
            [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
        ],
        # 查询的向量列表   用户的文本转向量
        # limit=2,
        output_fields=['id'],
        search_params={'metric_type': "IP"}  # 内积
    )
    print(f"result4:{result4}")


    result5 = client.search(
        collection_name='demo_v2',
        data=[
            [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]
        ],
        # 查询的向量列表   用户的文本转向量
        limit=2,
        output_fields=['id', 'color'],
        search_params={'metric_type': "IP"},  # 内积
        filter='color like "red%"'  # 类似于sql语句 模糊查询
    )
    print(f"result5:{result5}")


    # 1.定义搜索范围参数
    search_params = {
        "metric_type": "IP",
        "params": {
            "radius": 0.5,  # 搜索范围的下限  相似度>=0.9
            "range_filter": 1  # 搜索范围的上限  相似度<=1
        }
    }
    # 注意 radius 和  range_filter 参数必须同时指定 , 共同构成了 [0.9,1] 仅返回相似度在该区域的结果
    result5 = client.search(
        collection_name='demo_v2',
        data=[
            [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]
        ],
        # 查询的向量列表   用户的文本转向量
        # limit=2,
        output_fields=['id', 'color'],
        search_params=search_params,
    )
    print(f"result5:{result5}")



def complex_query_add_data():
    # --------------------------- 场景1: 定义集合schema(包含多个向量字段的表结构) ---------------------------
    # 1. 创建集合schema, 禁用动态字段(仅允许使用预定义字段)    只能使用你在 Schema 中预定义的字段
    schema = client.create_schema(enable_dynamic_field=False)  # 不允许插入未在Schema中定义的字段.

    # 2. 像schema中添加字段.
    # 2.1 主键字段film_id(电影id), 64位整数, 标记为主键.
    schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
    # 2.2 向量字段filmVector(电影特征向量), 浮点型向量, 向量维度为5.
    schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5)
    # 2.3 第2个箱容量字段posterVector(电影海报特征向量), 浮点型向量, 向量维度为5.
    schema.add_field(field_name='posterVector', datatype=DataType.FLOAT_VECTOR, dim=5)

    # --------------------------- 场景2: 定义多个向量字段的索引 ---------------------------
    # 1. 初始化索引参数配置对象.
    index_params = client.prepare_index_params()
    # 2. 为filmVector_电影特征向量 字段配置索引.
    # 参1:要创建索引的字段名, 参2: 索引类型(适合大规模向量检索), 参3: 相似度度量方式(欧式距离), 参4: 索引参数配置对象(IVF索引参数, 聚类数量为: 128)
    index_params.add_index(field_name='filmVector', index_type="IVF_FLAT", metric_type="L2", params={'nlist': 128})
    # 3. 为posterVector_电影海报特征向量 字段配置索引.
    # 参1: 要创建索引的字段名, 参2: 索引类型(未指定, 使用默认), 参3: 相似度度量方式(余弦相似度)
    index_params.add_index(field_name='posterVector', index_type="", metric_type="COSINE")

    # --------------------------- 场景3: 创建包含多向量字段的集合(即: Milvus表) ---------------------------
    # 1. 创建集合(表), 应用上述的 schema和索引配置.
    # 参1: 集合名称(即: Milvus的表名), 参2: 集合schema(表结构, 字段...), 参3: 索引参数配置对象
    client.create_collection(collection_name='demo_v3', schema=schema, index_params=index_params)

    # --------------------------- 场景4: 生成并插入实体数据(批量添加多条记录) ---------------------------
    # 1. 初始化实体列表(用于存储多条数据)
    entities = []
    # 2. 循环生成1000条实体数据.   _我后面100%用不到的,所以可以使用下划线方式
    for _ in range(1000):
        # 2.1 生成每条实体的 字段值.
        film_id = random.randint(1, 10000)  # 随机生成电影ID (1 ~ 10000之间)
        film_vector = [random.random() for _ in range(5)]  # 随机生成电影特征向量(5维)
        poster_vector = [random.random() for _ in range(5)]  # 随机生成电影海报特征向量(5维)

        # 2.2 构建实体字典(字段需要与schema中定义的字段名一致)
        entity = {"film_id": film_id, "filmVector": film_vector, "posterVector": poster_vector}
        # 2.3 添加到实体列表中.
        entities.append(entity)

    # 3. 打印实体数量 和 第一条实体数据, 验证数据生成结果.
    print(f'实体数量: {len(entities)}')
    print(f' 第一条实体数据: {entities[0]}')

    # 4. 执行upsert操作(批量插入/更新数据)
    # 参1: 集合名称(要添加数据到的Milvus表), 参2: 实体列表(要插入的具体的数据)
    client.upsert(collection_name='demo_v3', data=entities)

def complex_query():
    query_filmVector = [[0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]

    search_parma_1 = {
        "data": query_filmVector,
        "anns_field": "filmVector",
        "param" : {"metric_type": "L2","nprobe": 10},
        "limit": 2
    }

    request_1 = AnnSearchRequest(**search_parma_1)

    query_posterVector = [
        [0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]

    search_parma_2 = {
        "data": query_posterVector,
        "anns_field": "posterVector",
        "param" : {"metric_type": "COSINE"},
        "limit": 2
    }

    request_2 = AnnSearchRequest(**search_parma_2)

    reqs = [request_1, request_2]

    Weightedranker = WeightedRanker(0.1,0.7)
    rrf_ranker = RRFRanker(100)

    outputs = client.hybrid_search(
        collection_name="demo_v3",
        reqs=reqs,
        ranker=rrf_ranker,
        limit=2,
        output_fields=["filmVector", "posterVector"]
    )

    print(f"混合搜索结果:{outputs}")

if __name__ == '__main__':
    # 测1:创建数据库
    # operate_db()

    # 测2:创建集合demo_v1
    # operate_collection()

    # 测3:给集合字段添加索引
    # operate_index()

    # 测4:给集合插入数据
    # operate_entity()

    # 测5:查询测试
    # operate_query()

    # complex_query_add_data()

    complex_query()