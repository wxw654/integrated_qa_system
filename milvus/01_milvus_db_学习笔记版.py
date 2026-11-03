"""
==============================================================================
Milvus 向量数据库学习总结(带详解注释版)
==============================================================================

案例:
    演示通过pymilvus模块和 milvus向量数据库交互
大白话理解: 代码建库

------------------ 先搞懂几个核心概念(大白话) ------------------
Milvus 是"向量数据库",专门存"向量"(一组数字,比如 [0.1, 0.2, ...]),
用来做"相似度搜索"(比如: 给我找和这张图片/这句话最像的前10条数据)。

    数据库(Database)   ≈ MySQL 的 database      -> 最外层隔离单位
    集合(Collection)   ≈ MySQL 的 表(table)      -> 存一类数据
    字段(Field/Schema) ≈ MySQL 的 列(列名+类型)  -> 集合的结构
    实体(Entity)       ≈ MySQL 的 行(一条记录)   -> 一条数据
    分区(Partition)    ≈ 表里面的"文件夹"        -> 把集合数据按块分开存
    索引(Index)        ≈ MySQL 的 INDEX          -> 让搜索更快
==============================================================================
"""

# 1.导包
# MilvusClient: 操作 milvus 的核心客户端对象(连接、建库、建表、增删改查都靠它)
# DataType:     字段类型枚举,比如 INT64(整数)、VARCHAR(字符串)、FLOAT_VECTOR(浮点向量)
from pymilvus import MilvusClient, DataType

# 创建客户端对象,并连接到 milvus 服务
# url 参数: milvus 服务地址。localhost 表示本机,19530 是 milvus 默认端口
# (如果是单机版 milvus-lite,则用 uri="xxx.db" 指向本地文件,不需要连接服务)
client = MilvusClient(url="http://localhost:19530")
client.using_database(db_name="milvus_demo")


# =============================================================================
# todo 1 定义数据库操作函数 ,处理连接和数据库管理
# 【本节作用】连接 milvus 服务,并保证 milvus_demo 这个数据库存在(没有就创建)
# =============================================================================
def operate_db():
    # 1.配置数据库连接
    # 1.1本地文件数据库连接
    # MilvusClient(uri="ai_milvus.db")  <- 适合学习测试,数据存在本地文件里
    # 1.2实际开发连接方式 : 连接本地milvus服务,远程生产服务器的IP地址
    # client = MilvusClient(url="http://localhost:19530")

    # 2.处理milvus_demo数据库 , 有就使用 ,没有就创建
    # 2.1获取数据库列表
    # client.list_databases(): 返回当前 milvus 服务里所有数据库名字组成的列表
    databases = client.list_databases()

    # 2.2判断数据库milvus_demo是否存在(因为 list 里是字符串,所以用 in 判断)
    if "milvus_demo" not in databases:
        # 走这里,说明数据库不存在,就创建数据库
        # client.create_database(db_name="xxx"): 创建一个新数据库
        #   参 db_name: 要创建的数据库名字
        client.create_database(db_name="milvus_demo")
        print("创建数据库成功!")

    # 2.3切换当前使用的数据库(之后所有的建表/插入/查询都在这个库里操作)
    # client.using_database(db_name="xxx"): 指定后续操作使用哪个数据库
    #   参 db_name: 数据库名字
    client.using_database(db_name="milvus_demo")
    return client


# =============================================================================
# todo 2  collection集合的操作 -> milvus中创建表
# 【本节作用】用"自定义 schema"的方式创建集合 demo_v1(相当于 MySQL 建表)
# 需求:  集合名字为:demo_v1  三个字段 id vector scalar  字段名
#
# 【重要概念】向量字段必须指定 dim(维度), dim = 向量里数字的个数
#   比如 [0.1, 0.2, 0.3, 0.4, 0.5] 是 5 维向量, dim 就必须写 5
#   之后往这个字段插入的向量, 维度必须严格等于 dim, 多一个少一个都报错!
# =============================================================================
def operate_collection():
    # 1.定义集合结构(相当于:表结构)
    # client.create_schema(): 创建一个"空 schema(表结构)"对象,之后再往里面加字段
    #   参1 auto_id: 主键是否由 milvus 自动生成
    #        False = 手动指定 id(自己填 1、2、3...)
    #        True  = 插入时不用传 id, 数据库自增生成
    #   参2 enable_dynamic_field: 是否允许插入"schema里没定义的额外字段"
    #        True  = 比如表只定义了 id/vector/scalar, 但我插入时多带一个 "color" 字段,
    #                也不会报错(后面场景2就是这样用的!)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # 2.向schema中添加具体的字段
    # schema.add_field(): 往表结构里加一列
    #   参 field_name: 字段名(列名)
    #   参 datatype:   字段类型(DataType.xxx)
    #   参 is_primary: 是否为主键(每个集合必须有且只能有一个主键)
    # 2.1主键字段 id 64位整数 ,主键
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

    # 2.2添加向量字段  字段名: vector  字段类型:FLOAT_VECTOR(浮点向量)  该向量的维度:5
    #   参 dim=5: 向量维度, 表示每条数据的 vector 必须是包含 5 个浮点数的列表
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)

    # 2.3添加标量字段  varchar(255) —— 普通字段,存字符串
    #   参 max_length=256: 字符串最大长度, 超过会报错(类似 MySQL 的 varchar(256))
    #   参 description:   字段备注说明, 只是注释作用
    schema.add_field(field_name="scalar", datatype=DataType.VARCHAR, max_length=256, description='标量字段',)

    # 3.创建集合,并且定义集合名字为demo_v1
    # client.create_collection(): 真正执行"建表"
    #   参 collection_name: 集合(表)名字
    #   参 schema: 上面定义好的表结构
    #   注意: 如果同名集合已存在, 再次执行会报 "collection already exists" 错误
    client.create_collection(collection_name='demo_v1', schema=schema)

    # 4.打印提示
    print("创建milvus集合成功!")


# =============================================================================
# todo 3 为milvus创建索引
# 【本节作用】给字段添加索引(加速搜索) + 索引的查/删 + 集合的加载/释放
#
# 【重要概念】为什么向量数据必须建索引?
#   没有索引: 搜索时要和库里每一条向量逐个算相似度 -> 数据一多就慢到不可用
#   有了索引: milvus 提前把向量组织成特殊结构 -> 搜索快得多(牺牲一点准确度换速度)
#
# 【重要概念】索引类型(index_type)常用:
#   FLAT:     暴力精确搜索, 100%准确但慢, 适合小数据量学习用
#   IVF_FLAT: 倒排索引, 把向量分桶, 只搜索部分桶, 常用
#   HNSW:     图索引, 速度快准确度高, 内存占用大, 生产常用
#   标量字段的索引类型统一写 "INVERTED"(倒排索引)
#
# 【重要概念】metric_type(相似度计算方法):
#   COSINE: 余弦相似度, 看两个向量的"夹角方向"像不像, 越接近1越像(文本向量常用)
#   IP:     内积, 两个向量对应位置相乘再相加(也和夹角相关)
#   L2:     欧氏距离, 算两点间距离, 越小越像
# =============================================================================
def operate_index():
    # -------------------------场景1:为向量字段添加索引-------------------
    # 1.初始化索引参数配置对象 , 用于设置索引相关参数
    # client.prepare_index_params(): 创建一个"索引参数清单"对象,
    #   可以先往清单里 add 多个字段的索引配置, 最后一次性 create_index
    prepare_params = client.prepare_index_params()

    # 2.配置向量字段的索引信息(加入清单,此时还没真正创建)
    # prepare_params.add_index(): 向清单里添加一条索引配置
    #   参 field_name: 要给哪个字段建索引
    #   参 index_type: 索引类型(FLAT / IVF_FLAT / HNSW ...)
    #   参 metric_type: 相似度计算方法(IP / L2 / COSINE)
    #        !!注意: 必须和建集合时使用的相似度方式保持一致, 否则搜索时报错
    #   参 index_name: 索引的名字(方便后面查看/删除)
    # [修正] 原代码 index_type='' 是空字符串, 运行会报 "index_type not supported"!
    #        这里改为 FLAT(最基础的精确索引, 适合学习)
    prepare_params.add_index(field_name='vector', index_type='FLAT', metric_type='COSINE', index_name='vector_index')

    # 3.执行索引创建操作(把清单里的配置真正应用到集合上)
    # client.create_index(): 参 collection_name=给哪个集合建索引, 参 index_params=参数清单
    # client.create_index(collection_name="demo_v1", index_params=prepare_params)
    # print("*" * 40)

    # -------------------------场景2:查看索引信息-------------------
    # 1.列出集合所有的索引
    # client.list_indexes(): 参 collection_name=集合名; 返回该集合所有索引名的列表
    res = client.list_indexes("demo_v1")
    print(f"索引信息:{res}")

    # 2.查看指定索引的详细配置
    # client.describe_index(): 参 collection_name=集合名, 参 index_name=索引名
    #   返回值是字典: {'index_type': 'FLAT', 'metric_type': 'COSINE', ...}
    res2 = client.describe_index(collection_name="demo_v1", index_name="vector_index")
    print(f"索引信息:{res2}")

    # -------------------------场景3:集合的加载与释放-------------------
    # 【重要概念】milvus 是"内存计算型"数据库:
    #   搜索/查询前, 必须先把集合和索引"加载到内存"才能用!
    #   未加载状态 = 可写(insert/upsert) 不可读(search/query)
    #
    # 1.查看集合当前加载状态  默认是: 未加载  集合加载状态:{'state': <LoadState: NotLoad>}
    # client.get_load_state(): 参 collection_name=集合名; 返回 {'state': ...}
    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')
    #
    # 2.将集合加载到内存,加载后才能搜索
    # client.load_collection(): 参 collection_name=集合名
    # client.load_collection(collection_name="demo_v1")
    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')
    #
    # 3.释放集合 将集合状态变为未加载(从内存中清出去,节省内存)
    # client.release_collection(): 参 collection_name=集合名
    # client.release_collection(collection_name="demo_v1")
    # print(f'集合加载状态:{client.get_load_state(collection_name="demo_v1")}')

    # 小结
    # 1:milvus中,集合必须处于加载状态,才能执行搜索和查询操作.否否会报错
    # 未加载=可写不可读

    # -------------------------场景4:删除索引-------------------
    # 1.删除指定集合中的指定索引
    # client.drop_index(): 参1:集合名称  参2:(要被删除的)索引名称
    # client.drop_index(collection_name="demo_v1",index_name="vector_index")

    # -------------------------场景5:给标量字段添加索引-------------------
    # 1.配置标量字段的索引参数
    # [修正] 原代码 index_type='' 空字符串同样会报错; 标量字段索引类型固定用 "INVERTED"
    # 注意: 标量索引不写 metric_type, 因为相似度计算只针对向量字段
    prepare_params.add_index(field_name='scalar', index_type='INVERTED', index_name='scalar_index')
    # 2.执行索引创建操作
    client.create_index(collection_name="demo_v1", index_params=prepare_params)

    # 3.查看索引信息
    print(f"索引信息:{client.list_indexes(collection_name='demo_v1')}")
    res3 = client.describe_index(collection_name="demo_v1", index_name="scalar_index")
    print(f"索引信息:{res3}")


# =============================================================================
# todo 4 表数据的增删改
# 【本节作用】实体(Entity=行)的插入 insert / 更新插入 upsert / 分区 / 删除 delete
# =============================================================================
def operate_entity():
    # --------------场景1:创建collection集合--------------
    # 1.直接创建集合,无需指定schema 简化方式(快速创建)
    # client.create_collection() 的"快速模式": 不传 schema, 只传维度即可,
    #   milvus 自动生成默认表结构: id(INT64主键) + vector(FLOAT_VECTOR)
    #   参1 collection_name: 集合名称
    #   参2 dimension: 向量维度(快速创建时必填!! 不传直接报错)
    #   参3 metric_type: 向量相似度计算方式(IP,内积; 默认是COSINE)
    # [踩坑记录] 快速创建时 dimension 是必填参数; 且 partition_name 不是本方法的参数!
    client.create_collection(collection_name="demo_v2", dimension=5, metric_type="IP")

    # --------------场景2:向集合中添加数据--------------
    # 1.准备要添加的数据(实体列表) —— 每条数据是一个字典
    # 小结 : id(主键)  vector(5维向量)  color(动态标量字段)
    #   color 没在 schema 里定义过, 但因为建表时 enable_dynamic_field=True,
    #   所以能直接存进去(这就是"动态字段"的威力)
    data = [
        {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                             0.9029438446296592], "color": "pink_8682"},
        {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
                             0.838729485096104], "color": "red_7025"},
        {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
                             0.20785793220625592], "color": "orange_6781"},
        {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
                             0.95791889146345], "color": "pink_9298"},
        {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
                             0.30337481143159106], "color": "red_4794"},
        {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
                             -0.1446277761879955], "color": "yellow_4222"},
        {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
                             -0.8984947637863987], "color": "red_9392"},
        {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
                             0.5378064918413052], "color": "grey_8510"},
        {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
                             -0.6140360785406336], "color": "white_9381"},
        {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
                             -0.6980531615588608], "color": "purple_4976"}
    ]
    # 2.执行insert操作(纯插入)
    # client.insert(): 参 collection_name=插入哪个集合, 参 data=数据列表
    # 注意: insert 如果主键 id 已存在, 不会报错, 而是"追加"一条同id数据(容易产生重复)
    # client.insert(collection_name="demo_v2",data=data)

    # 3.执行upsert操作(insert+update -> 如果主键存在就更新,没有就添加)
    # client.upsert(): 参数和 insert 一样; 实际开发中更常用 upsert, 防止数据重复
    # client.upsert(collection_name='demo_v2',data=data)
    # print("数据添加成功")

    # -------------------------------------场景3:给集合创建分区------------------------------------------
    # 1.创建分区(分区用于对集合数据分类存储,方便管理)
    # 【大白话】分区 = 同一个表里的"文件夹", 比如按月份/类别把数据分开放,
    #   搜索时指定分区名, 就只扫描那个分区 -> 更快、更省资源
    # client.create_partition():
    #   参 collection_name: 在哪个集合(表)里建分区
    #   参 partition_name:  分区名字
    # 注意: 每个集合自动有一个默认分区 "_default", 不指定分区的数据都存在它里面
    # client.create_partition(collection_name='demo_v2', partition_name='partitionA')

    # -------------------------------------场景4:向指定分区插入数据--------------------------------------
    # 1.准备要插入的分区数据(结构要和场景2的保持一致 , 但是要注意避免主键冲突 10~19)
    # 注意: id=19 是全0向量 [0,0,0,0,0], upsert 不报错, 但之后用 IP/COSINE 搜索时
    #       全0向量没有方向, 相似度结果会异常, 实际项目中要避免!
    # data = [
    #     {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
    #                           0.6699215060604258], "color": "red_1202"},
    #     {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
    #                           -0.14152860714878068], "color": "blue_4150"},
    #     {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
    #                           -0.8351194181285713], "color": "orange_4590"},
    #     {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
    #                           -0.0010987671273892108], "color": "pink_9619"},
    #     {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
    #                           -0.7986720938400362], "color": "orange_4863"},
    #     {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
    #                           0.24448151140671204], "color": "orange_7984"},
    #     {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
    #                           0.15077052220816167], "color": "blue_9010"},
    #     {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
    #                           -0.33007998971889263], "color": "blue_4521"},
    #     {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
    #                           -0.7387383128324057], "color": "orange_2529"},
    #     {"id": 19, "vector": [0, 0, 0, 0,
    #                           0], "color": "red_9437"}
    # ]
    #
    # 2.向指定的分区执行upsert操作
    # 参 partition_name: 指定数据写到哪个分区(分区必须已存在, 否则报错!)
    # client.upsert(collection_name='demo_v2',partition_name='partitionA',data=data)

    # -------------------------------------场景5:从集合中删除数据------------------------------------------
    # 1. 按过滤器条件删除(不指定分区,默认遍历整个集合) 类似于sql的条件查询
    # client.delete():
    #   参 collection_name: 从哪个集合删
    #   参 filter: 删除条件表达式(字符串), 语法类似 SQL 的 where:
    #       "id in [12,5,6]"         -> 删 id 是 12、5、6 的三个
    #       "id > 100"               -> 删 id 大于 100 的所有
    #       'color like "red%"'      -> 删 color 以 red 开头的
    #       "scalar == 'abc' && id < 5" -> 组合条件
    # res=client.delete(collection_name='demo_v2',filter='id in [12,5,6]')
    # print(res)

    # 2.按主键 +指定分区删除(仅删除指定分区的目标数据)
    # 参 ids: 主键列表, 会删掉该分区里 id=13 和 id=2 的两条数据
    # 注意: id=2 在默认分区 _default 里, 不在 partitionA 里, 所以这条只会删掉 13;
    #       想删 id=2 必须去 _default 分区再删一次(分区隔离的意义就在这)
    res = client.delete(collection_name='demo_v2', partition_name='partitionA', ids=[13, 2])
    print(res)


# =============================================================================
# todo 5 集合数据的简单查询(向量搜索 = milvus 最核心的价值)
# 【本节作用】用"一个向量"去集合里找"最相似的 N 条数据"
#
# 【大白话流程】RAG 中的用法:
#   用户提问 -> 把问题转成向量 -> search 找最相似的几条知识 -> 把知识喂给大模型回答
#
# 【client.search 核心参数一览】
#   collection_name: 在哪个集合里搜
#   data:            查询向量列表(二维列表! 外层是"一批查询", 内层是一个向量)
#   limit:           每个查询要返回前几条最相似的(默认10)
#   output_fields:   结果里要额外返回哪些字段(不写就只返回 id 和相似度分数)
#   search_params:   搜索参数, 一般写 {'metric_type': "IP"} 指定相似度算法
#                    (必须和建索引时的 metric_type 一致!)
#   partition_names: 只在指定分区里搜(缩小范围提速)
#   filter:          标量过滤条件(先按条件筛掉一部分, 再在剩下的里找相似)
#
# 【前提!!】搜索前集合必须处于"已加载"状态: client.load_collection(...)
# =============================================================================
def operate_query():
    # ------------------场景1: 单一向量查询(根据单个向量找相似结果)------------------
    # data 只放了 1 个向量 -> 返回 1 组最相似结果
    # result1 = client.search(
    #     collection_name='demo_v2',
    #     data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
    #     # 查询的向量列表   用户的文本转向量
    #     # limit=2,          # 不写默认返回10条
    #     output_fields=['id', 'vector', 'color'],  # 希望结果里带出这些字段
    #     search_params={'metric_type': "IP"}  # 内积
    # )
    # print(f"result1:{result1}")
    # 返回结构: [[{'id': 1, 'distance': 0.99, 'entity': {'color': 'red_7025', ...}}, ...]]
    #   distance: 相似度分数(IP 越大越像 / L2 越小越像 / COSINE 越接近1越像)

    # ------------------场景2: 批量向量查询(一次传多个向量,各找各的相似结果)------------------
    # data 里放了 2 个向量 -> 结果也是两组, 一一对应
    # result2 = client.search(
    #     collection_name='demo_v2',
    #     data=[
    #         [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
    #         [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
    #     ],
    #     # 查询的向量列表   用户的文本转向量
    #     # limit=2,
    #     output_fields=['id', 'vector', 'color'],
    #     search_params={'metric_type': "IP"}  # 内积
    # )
    # print(f"result2:{result2}")

    # ------------------场景3: 分区搜索(仅在指定的分区内查找结果)------------------
    # 参 partition_names: 分区名列表, 只在这些分区里搜 -> 数据量大时明显提速
    # 注意: 指定了分区就只搜分区里的数据, 默认分区 _default 里的数据会被忽略!
    # result3=client.search(
    #     collection_name="demo_v2",
    #     data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
    #     limit=2,
    #     search_params={'metric_type': "IP"},
    #     partition_names=['partitionA']
    #
    # )
    # print(f"result3:{result3}")

    # ------------------场景4: 指定输出字段 ------------------
    # output_fields: 控制返回结果里带哪些字段
    #   只要 id       -> 结果最轻量(实际项目常见: 拿 id 再回 MySQL 查详情)
    #   要 vector     -> 能把相似向量本身也拿回来
    #   要 $meta      -> 动态字段(color 就是动态字段)可写 ['color'] 直接取
    # result4 = client.search(
    #     collection_name='demo_v2',
    #     data=[
    #         [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
    #         [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
    #     ],
    #     # 查询的向量列表   用户的文本转向量
    #     # limit=2,
    #     output_fields=['id'],
    #     search_params={'metric_type': "IP"}  # 内积
    # )
    # print(f"result4:{result4}")

    # ------------------场景5: 带过滤条件的搜索(先过滤,再找相似度结果) ------------------
    # 参 filter: 先用标量条件筛掉不符合的数据, 再在剩下的里做向量相似度搜索
    #   'color like "red%"'  = 只在 color 以 red 开头的数据里找相似(模糊查询)
    #   也可写: 'color in ["red_7025","pink_9298"]'、'id > 5 && id < 15'
    # result5 = client.search(
    #     collection_name='demo_v2',
    #     data=[
    #         [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]
    #     ],
    #     # 查询的向量列表   用户的文本转向量
    #     limit=2,
    #     output_fields=['id', 'color'],
    #     search_params={'metric_type': "IP"},  # 内积
    #     filter='color like "red%"'  # 类似于sql语句 模糊查询
    # )
    # print(f"result5:{result5}")

    # ------------------场景6: 范围搜索(仅返回相似度在指定范围的结果) ------------------
    # 1.定义搜索范围参数
    # search_params 里嵌套 params, 增加两个范围控制参数:
    #   radius:       相似度下限, 结果必须 >= 这个值(过滤掉"不够像"的)
    #   range_filter: 相似度上限, 结果必须 <= 这个值(过滤掉"过度相似/重复"的)
    # 用途: 比如只要相似度 0.5~1.0 的结果, 不相似的直接丢弃 -> RAG 中用来
    #       控制"检索质量门槛", 太相关的知识不要, 不相关的也不要
    # 注意 IP/COSINE 是"越大越像"; 如果 metric 是 L2(越小越像), 则 radius 是距离下限
    search_params = {
        "metric_type": "IP",
        "params": {
            "radius": 0.5,  # 搜索范围的下限  相似度>=0.5
            "range_filter": 1  # 搜索范围的上限  相似度<=1
        }
    }
    # 注意 radius 和 range_filter 参数必须同时指定 , 共同构成了 [0.5,1] 仅返回相似度在该区域的结果
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

    # 【补充区分】search 和 query 是两个不同的东西:
    #   search: "找相似" —— 输入向量, 按相似度排序返回(本节讲的就是它)
    #   query:  "精确查" —— 按 filter 条件(主键/标量)原样取出数据, 不算相似度
    #     例: client.query(collection_name='demo_v2', ids=[1,2], output_fields=['color'])


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
    operate_query()
