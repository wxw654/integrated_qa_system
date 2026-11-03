# -*- coding: utf-8 -*-
# @Time : 2026/10/13
# @Author : hao
# @File : bm25_search.py
# @Project : integrated_qa_system
"""
【步骤7: BM25核心搜索引擎】  <-- 这是整个MySQL_QA模块的核心！

整体执行流程图:
    1 用户输入问题
    2/3 先查 Redis 缓存  -> 命中直接返回 (答案, False)
    4   问题分词 + BM25 打分
    5   softmax 归一化
    6   argmax 取最高分及其索引
    7   判断是否大于阈值
        否(8-2) -> 返回 (None, True)  表示需要走 RAG
        是(8-1) -> 取原始问题 -> (9) 从 MySQL 检索答案
                    -> (10) 答案写回 Redis -> 返回 (答案, False)

BM25算法简介:
    - Best Matching 25，信息检索经典算法
    - 核心思想: "查询词在文档中出现的频率越高、文档越短，得分越高"
    - 为什么用BM25而不是向量相似度? -> 本项目是"精确问答"场景，关键词匹配比语义相似度更可控

softmax的作用:
    - BM25原始分数是无范围的(可能0到几十)，没法直接用阈值判断
    - softmax把分数归一化为概率分布(0-1之间，总和=1)
    - 这样就可以用"置信度>0.85"来判断是否可信

返回值约定: (answer, need_rag)
    - ("答案文本", False): MySQL_QA匹配成功，直接返回答案
    - (None, True):              未匹配，需要走RAG流程
"""

from rank_bm25 import BM25Okapi  # pip install rank-bm25，BM25算法实现
import numpy as np               # 用于softmax计算和argmax取最大值的索引

from base.logger import logger
from mysql_qa.utils.preprocess import preprocess_text    # 步骤6: 分词
from mysql_qa.cache.redis_client import RedisClient      # 步骤5: 缓存
from mysql_qa.db.mysql_client import MysqlClient         # 步骤4: 数据库


class BM25Search:
    """
    BM25搜索引擎类
    职责: 加载数据建索引 -> 接收查询 -> 返回匹配答案或提示走RAG
    """

    def __init__(self, redis_client, mysql_client):
        """
        初始化搜索引擎

        参数说明:
            redis_client:  RedisClient实例，用于读写缓存
            mysql_client:  MysqlClient实例，用于查询问题和答案

        为什么用依赖注入(传参)而不是内部创建?
            - 方便测试(可以传入mock对象)
            - 复用连接(多个BM25Search实例共享同一个连接)
        """
        self.logger = logger
        self.redis_client = redis_client    # 缓存客户端
        self.mysql_client = mysql_client    # 数据库客户端

        self.bm25 = None                 # BM25 模型对象(稍后初始化)
        self.questions = None            # 分词后的问题列表(建索引用)
        self.original_questions = None   # 原始问题列表(命中后取答案用，保持顺序一致)

        # 构造时就加载数据建索引，保证对象可用
        self._load_data()

    def _load_data(self):
        """
        加载数据并构建BM25索引(对应流程4的前置步骤)

        执行逻辑:
            1. 先尝试从Redis读取缓存的分词结果(快速启动)
            2. 缓存没有则从MySQL加载原始问题
            3. 对问题分词，结果写回Redis(下次启动更快)
            4. 用分词结果初始化BM25模型

        为什么缓存分词结果? -> 如果有1万条问题，每次重启都重新分词太慢
        """
        try:
            # Redis中缓存分词结果的key名
            original_key = "qa_original_questions"   # 存原始问题列表
            tokenized_key = "qa_tokenized_questions" # 存分词后的列表

            # 第一步: 尝试从Redis读取缓存
            self.original_questions = self.redis_client.get_data(original_key)
            # print(f"original_questions: {self.original_questions}")
            tokenized_questions = self.redis_client.get_data(tokenized_key)
            # print(f"tokenized_questions: {tokenized_questions}")

            # 第二步: 缓存没有，从MySQL加载
            if not self.original_questions or not tokenized_questions:
                # fetch_questions() 返回 [("问题1",), ("问题2",), ...]
                self.original_questions = self.mysql_client.fetch_questions()

                if not self.original_questions:
                    self.logger.warning("未从MySQL中加载任何数据")
                    return

                # 对每个问题分词: question[0] 是因为fetch返回tuple，取第一个元素
                # 结果: [["python","是","什么"], ["git","怎么","用"], ...]
                tokenized_questions = [preprocess_text(question[0]) for question in self.original_questions]

                # 写回Redis缓存，下次启动直接用
                # 注意存的是纯字符串列表: [question[0] for question in ...]
                self.redis_client.set_data(original_key, [question[0] for question in self.original_questions])
                self.redis_client.set_data(tokenized_key, tokenized_questions)

            self.questions = tokenized_questions

            # 第三步: 初始化BM25模型
            # BM25Okapi(corpus): 传入"文档集合"，每个文档是一个词列表
            # 它内部会计算每个词的IDF值、平均文档长度等统计信息
            self.bm25 = BM25Okapi(self.questions)

            self.logger.info("BM25 模型初始化完成")
        except Exception as e:
            self.logger.error(f"加载数据/构建索引失败: {e}")
            raise

    @staticmethod
    def _softmax_static(scores):
        """
        流程5: 对 BM25 得分做 softmax 归一化, 转成概率分布

        参数:
            scores: numpy.ndarray, BM25计算的原始分数数组
                    例如: [12.5, 3.2, 8.7, 0.5]  (无范围限制)

        返回值:
            numpy.ndarray: 归一化后的概率数组，总和=1
                    例如: [0.65, 0.03, 0.31, 0.01]

        公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))
        为什么减去最大值? -> 防止 np.exp(大数) 溢出变成 inf
                         减最大值后结果等价(分子分母同除一个常数)
        """
        exp_scores = np.exp(scores - np.max(scores))  # 稳定版softmax
        return exp_scores / exp_scores.sum()

    def _softmax(self, scores):
        """
        实例方法版softmax(和上面功能一样，只是可以通过self._softmax调用)
        参数: scores - BM25原始分数数组
        返回: 归一化后的概率数组
        """
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        """
        核心搜索方法: 处理用户查询，计算相似度，返回匹配答案或提示走RAG

        参数说明:
            query:     str, 用户输入的查询问题
                       例如: "python是什么"
            threshold: float, 相似度阈值，默认0.85
                       为什么是0.85? -> softmax后概率>0.85表示非常确信匹配
                       太低会误匹配，太高会漏匹配，0.85是经验值

        返回值: tuple (answer, need_rag)
            - ("答案文本", False): 匹配成功，直接返回答案
            - (None, True):              匹配失败，需要走RAG流程
            为什么返回两个值? -> 上层调用方(main.py)根据need_rag决定走哪个分支

        执行步骤:
            1. 参数校验
            2. 查Redis缓存(命中直接返回)
            3. 分词
            4. BM25打分
            5. softmax归一化
            6. 取最高分
            7. 阈值判断
            8. 返回结果
        """

        # ========== 步骤1: 参数校验 ==========
        # isinstance() 判断query是否是字符串类型
        # 为什么要校验? -> 防止传入None/数字等异常值导致后续报错
        if not query or not isinstance(query, str):
            self.logger.error("无效查询:查询为空或者非字符串类型")
            return None, True  # 视为"未匹配"，让上层走RAG

        # ========== 步骤2: 查Redis缓存 ==========
        # 为什么先查缓存? -> 性能最优路径: 用户问过的问题直接返回，省掉3-8步
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False  # 命中缓存，直接返回

        try:
            # ========== 步骤3: 对查询文本分词 ==========
            # "python是什么" -> ["python", "是", "什么"]
            query_tokens = preprocess_text(query)

            # ========== 步骤4: BM25打分 ==========
            # get_scores(query_tokens): 计算查询与所有已索引问题的相似度分数
            # 返回: numpy数组，长度=问题总数，每个元素是与对应问题的BM25得分
            scores = self.bm25.get_scores(query_tokens)

            # ========== 步骤5: softmax归一化 ==========
            # 把无范围的BM25分数 -> 0~1之间的概率分布
            softmax_scores = self._softmax(scores)

            # ========== 步骤6: 取最高分及其索引 ==========
            # argmax(): 返回数组中最大值的索引位置
            # 例如 softmax_scores=[0.1, 0.85, 0.05], best_idx=1
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]  # 最高分的值

            # ========== 步骤7: 阈值判断 ==========
            if best_score >= threshold:
                # 7.1 获取最高匹配度的原始问题文本
                # self.original_questions 是从Redis/MySQL加载的纯字符串列表
                original_question = self.original_questions[best_idx]

                # 7.2 用原始问题去MySQL精确查询答案
                answer = self.mysql_client.fetch_answer(original_question)

                if answer:
                    # 7.3 将"用户查询->答案"缓存到Redis
                    # 注意: key是用户的原始query(不是匹配到的original_question)
                    # 这样下次同样的问题再问，步骤2就能直接命中
                    self.redis_client.set_data(f"answer:{query}", answer)

                    self.logger.info(f"搜索成功,softmax相似度:{best_score:.3f}")
                    return answer, False  # 匹配成功

            # ========== 步骤8: 未超过阈值，返回走RAG ==========
            # :.3f 表示保留3位小数，如 0.452
            self.logger.info(f"未找到可靠的答案,最高softmax相似度:{best_score:.3f}(低于阈值{threshold})")
            return None, True  # 未匹配，上层走RAG

        except Exception as e:
            self.logger.error(f"搜索查询失败:{e}")
            return None, True  # 异常时也走RAG，保证系统可用性




if __name__ == '__main__':
    # 测试代码: 单独运行本文件验证BM25搜索全流程

    # 创建依赖对象(实际项目中由上层统一创建并传入)
    redis_client = RedisClient()    # 步骤5: Redis缓存
    mysql_client = MysqlClient()    # 步骤4: MySQL数据库

    # 创建搜索引擎(构造时会自动调用 _load_data 建索引)
    bm25_search = BM25Search(redis_client, mysql_client)

    # 测试搜索功能: 看能否从知识库中找到匹配的答案
    # 如果MySQL里有"python是什么"相关的问题，会返回(答案, False)
    # 如果没有匹配的，会返回(None, True)表示需要走RAG
    result = bm25_search.search("python是什么")
    print(result)
