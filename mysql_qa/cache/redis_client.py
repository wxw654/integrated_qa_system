"""
【步骤5: Redis客户端 - 数据缓存和查找】
作用: 封装 Redis 连接，提供"存储/读取"缓存的接口

为什么需要 Redis 缓存? 在本系统中有两个作用:
    1. 缓存问答结果: 用户问过的问题，下次再问直接从Redis返回，不用重新走BM25+MySQL
    2. 缓存分词结果: BM25索引构建时的分词数据存Redis，重启服务时不用重新分词

性能对比: Redis读操作 ~0.1ms  vs  MySQL查询 ~5ms  vs  BM25计算 ~50ms

其他模块怎么用?
    from mysql_qa.cache.redis_client import RedisClient
    rc = RedisClient()
    rc.set_data("key", "value")
    rc.get_data("key")
"""
import redis   # Python的Redis客户端库
import json    # 用于序列化/反序列化(Redis只能存字符串，列表/dict需要转JSON)
import os, sys

from base.logger import logger   # 步骤3的日志
from base.config import Config   # 步骤2的配置


class RedisClient:
    """
    Redis操作封装类
    为什么封装成类? -> 连接只建立一次，多个方法复用同一个client对象
    """

    def __init__(self):
        """
        初始化: 建立 Redis 连接
        为什么在构造时就连接? -> 后续方法频繁调用，提前建立好连接池
        """
        self.logger = logger
        try:
            conf = Config()  # 读取 config.ini 中的 Redis 配置

            # redis.StrictRedis() 创建连接对象，参数解释:
            #   host:     Redis服务器地址
            #   port:     端口号(默认6379)
            #   password: 访问密码(防止未授权访问)
            #   db:       使用第几个逻辑库(0-15，互相隔离)
            #   decode_responses: True=自动把bytes解码为str
            #       为什么需要? 不设置的话取出来的是b'xxx'字节串，还得手动decode
            self.client = redis.StrictRedis(
                host=conf.REDIS_HOST,
                port=conf.REDIS_PORT,
                password=conf.REDIS_PASSWORD,
                db=conf.REDIS_DB,
                decode_responses=True
            )
            self.logger.info("Redis连接成功")
        except Exception as e:
            self.logger.error(f"Redis connection error: {e}")
            raise  # 重新抛出异常，让调用方知道连接失败

    def set_data(self, key, value):
        """
        存储数据到 Redis

        参数说明:
            key:   str, 缓存的键名(唯一标识)
                   例如: "answer:python是什么" 或 "qa_tokenized_questions"
            value: 任意可序列化对象(str/list/dict)
                   为什么要json.dumps? -> Redis的SET命令只接受字符串，
                   list/dict必须先转成JSON字符串才能存
                   ensure_ascii=False:

        调用场景:
            - BM25搜索成功后缓存答案: set_data("answer:查询问题", "答案")
            - 缓存分词结果: set_data("qa_tokenized_questions", [["python","是什么"], ...])
        """
        try:
            # json.dumps() 把Python对象转为JSON字符串
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f"Redis存储信息成功: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis存储信息失败: {e}")
            # 注意: 这里没raise，因为缓存写入失败不影响主流程(只是下次要重新算)

    def get_data(self, key):
        """
        从 Redis 读取数据

        参数说明:
            key: str, 要读取的键名

        返回值:
            反序列化后的Python对象(list/dict/str)，不存在返回 None
            为什么要json.loads? -> 存入时dumps成了字符串，取出时要还原为Python对象

        调用场景: BM25Search._load_data() 中读取缓存的分词结果
        """
        try:
            data = self.client.get(key)  # Redis GET命令，key不存在返回 None
            # 三元表达式: 有值就json.loads反序列化，没值返回None
            return json.loads(data) if data  else None
        except redis.RedisError as e:
            self.logger.error(f"Redis获取数据失败: {e}")

    def get_answer(self, query):
        """
        专门用于查询"答案缓存"的方法

        参数说明:
            query: str, 用户的原始问题文本
                内部会自动加前缀 "answer:" 作为Redis的key
                为什么加前缀? -> 命名空间隔离，避免和其他缓存key冲突

        返回值:
            str 或 None: 缓存的答案文本，未命中返回 None

        与 get_data 的区别:
            - get_data: 通用读取，会json.loads反序列化
            - get_answer: 直接返回字符串(答案本身就是纯文本，不需要反序列化)

        调用场景: BM25Search.search() 步骤2，先查缓存是否命中
        """
        try:
            # 拼接key: "answer:" + 用户问题
            answer = self.client.get(f"answer:{query}")
            if answer:
                self.logger.info(f"Redis获取数据成功: {query}")
                return answer
            return None  # 缓存未命中，返回None让调用方继续走BM25流程
        except redis.RedisError as e:
            self.logger.error(f"Redis获取数据失败: {e}")
            raise


if __name__ == '__main__':
    # 测试代码: 单独运行本文件时验证Redis连接和读写功能

    # 测1: 测试客户端连接(能实例化不报错就说明连接成功)
    redis_client = RedisClient()

    # 测2: 添加数据(set_data存储，会自动json序列化)
    # redis_client.set_data("PyCharm只能以单元测试方式运行python代码怎么解决", "方法或者类中包含 test")

    # 测3: 获取所有的key(查看Redis中当前存储了哪些缓存)
    # print(redis_client.client.keys())

    # 测试4: 测试get_answer专用方法(配合BM25Search的答案缓存逻辑)
    # 先存入一个带"answer:"前缀的缓存
    redis_client.set_data("answer:PyCharm只能以单元测试方式运行python代码怎么解决", "hahaha")
    # 然后用get_answer读取(它会自动加"answer:"前缀去查)
    redis_client.get_answer(query="PyCharm只能以单元测试方式运行python代码怎么解决")


































