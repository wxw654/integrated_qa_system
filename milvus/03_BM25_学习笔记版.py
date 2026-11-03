"""
==============================================================================
BM25 关键词检索学习总结(带详解注释版)
==============================================================================

案例:
    用 jieba 分词 + rank_bm25 库, 实现一个最简单的"中文关键词搜索"类

大白话理解: BM25 是"百度搜关键词"的迷你版
    你给它一堆文档(语料库), 再给它一个查询词,
    它会算出"每篇文档和这个查询词的相关度分数", 分数最高的就是最匹配的文档。

------------------ 先搞懂几个核心概念(大白话) ------------------
    BM25:  经典的信息检索打分算法(搜索引擎的鼻祖算法之一)。
           核心思想: 查询词在文档里出现得越多、且这个词在其他文档里越少见,
           这篇文档得分越高。(类似 TF-IDF 的改进版)
           注意: BM25 只看"字面是否命中关键词", 不懂语义!
                 搜"编程", "编写程序" 这种同义表达它是匹配不到的 ——
                 这正是它和"向量检索(RAG语义搜索)"的本质区别, 两者常互补使用。

    jieba: 中文分词库。英文天生用空格分词, 中文句子是连着的,
           必须先切成词 ["他", "喜欢", "编程"], BM25 才能按词去匹配。

    BM25L: BM25 的一个变种算法(L = Length normalized),
           对"长文档中词频饱和"做了修正, 其他用法和 BM25 完全一样。
           rank_bm25 库里还有 BM25Okapi、BM25Plus 等变体可选。

    倒排索引思路: BM25 内部把语料库建成 "词 -> 出现在哪些文档" 的索引,
           所以文档是分词后传入的, 查询时也要用相同的分词方式。
==============================================================================
"""

# 2.导包
# jieba: 中文分词库, 需要 pip install jieba
#   注意: 原代码写的是 `import  jieba`(两个空格), python 不报错但不规范
import jieba
# BM25L: rank_bm25 库提供的 BM25L 算法实现, 需要 pip install rank_bm25
#   它不是标准库! 是第三方的纯 python 实现
from rank_bm25 import BM25L
# logging: 标准库日志模块(用法详见 02_logging_学习笔记版.py)
import logging

# 获取当前模块的记录器
# __name__: 特殊变量, 直接运行本文件时它的值就是 '__main__'
#   在本模块里 getLogger(__name__) 是养成"每个模块用自己的 logger"好习惯的写法
logger = logging.getLogger(__name__)

# 全局简单配置: INFO 级别 + 时间/级别/消息 格式
# 注意顺序: 原代码是先 getLogger 再 basicConfig —— 没关系,
# basicConfig 配置的是"根记录器", 模块 logger 会把日志"上报"给根记录器统一输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =============================================================================
# todo 1 定义 Bm25Search 类: 封装"建索引 + 搜索"两步
# 【本节作用】把 BM25 检索封装成可复用的类:
#   初始化时: 分词语料库 -> 训练 BM25 模型(建索引)
#   search时: 分词查询   -> 算所有文档得分 -> 取分数最高的一篇返回
# =============================================================================
class Bm25Search():

    def __init__(self, documents):
        # 参 documents: 原始文档列表(语料库), 如 ["我喜欢编程", "编程很有趣"]
        # 保存原始文本, 后面搜到索引后要靠它取回"人能看的完整句子"
        self.documents = documents

        # 对每一篇文档做 jieba 分词
        # jieba.lcut(句子): 返回"切好的词列表"(l = list), 如
        #   jieba.lcut("我喜欢编程") -> ['我', '喜欢', '编程']
        #   (还有 jieba.cut 返回迭代器, 省内存; lcut 直接返回列表, 方便)
        # 列表推导式: 对 documents 里每个 doc 都分一次词
        # BM25 要求传入的必须是"分词后的二维列表": [['我','喜欢','编程'], ...]
        self.tokenized_documents = [jieba.lcut(doc) for doc in documents]

        # 用分词后的语料库初始化 BM25L 模型(这一步就是"建索引")
        # BM25L(语料库): 内部统计每个词出现在哪些文档、词频、文档长度等
        # 建好后 self.bm25 就是一个可以打分的"模型"了
        self.bm25 = BM25L(self.tokenized_documents)
        logger.info("BM25模型初始化完成")

    def search(self, query):
        """对 query 做检索, 返回(最相关文档, 分数); 失败返回(None, 0)"""
        # 1.查询语句也要分词 —— 必须和建索引时"同一套分词器", 否则词对不上
        #   如 query="他喜欢编程" -> ['他', '喜欢', '编程']
        tokenized_query = jieba.lcut(query)

        # try/except: 兜底保护, 搜索出错时不让程序崩掉, 只记错误日志并返回空结果
        try:
            # 2.计算 query 与"每一篇文档"的相关度分数
            # self.bm25.get_scores(分词后的查询):
            #   返回 numpy 数组, 长度 = 文档数量, 每个元素是对应文档的得分
            #   例如 [0. 1.2] 表示 文档0 得 0 分, 文档1 得 1.2 分
            #   注意: 查询里没在语料库出现过的词(如"他")不贡献分数
            scores = self.bm25.get_scores(tokenized_query)
            print(f"检索结果: {scores}")

            # 3.找到分数最高的文档下标
            # scores.argmax(): numpy 方法, 返回数组中最大值所在的下标(index)
            #   如 scores=[0. 1.2] -> argmax()=1 (第二个文档分最高)
            # 注意: 这里只取了"第1名"; 实际项目中一般用
            #   scores.argsort()[::-1][:top_k] 取前 K 条(Top-K 检索)
            best_idx = scores.argmax()
            print(f"最佳文档索引: {best_idx}")

            # 4.用下标取出最高分数(scores[下标] 就是 numpy 数组取值)
            best_score = scores[best_idx]

            # 5.用同一个下标, 从"原始文档列表"里取回完整句子
            #   这就是为什么 __init__ 里要保存 self.documents 的原因
            best_doc = self.documents[best_idx]

            # 6.记录一条 INFO 日志: 查了什么、命中了什么、多少分
            logger.info(f"检索完成，查询: {query} 最佳文档: {best_doc}, 最佳分数: {best_score}")

            # 7.返回结果: (文档文本, 分数) 组成元组返回
            #   调用方用 doc, score = bm25.search(...) 接收(元组解包)
            #   把分数一起返回是因为: 下游常需要按分数判断"相关性够不够, 要不要采纳"
            return best_doc, best_score
        except Exception as e:
            # 出异常(如语料库为空、query 类型不对): 记 ERROR 日志, 返回"没找到"
            # e 是异常对象, f-string 里直接打印异常信息(异常类型+描述)
            logger.error(f"搜索出错: {e}")
            return None, 0

    # 【说明/修正】原代码这里 main() 没有写 self 参数, 是直接通过
    #   "类名.Bm25Search.main()" 调用的, 相当于把 main 当普通函数用 —— 能跑,
    #   但这不是规范的实例方法写法。学习时注意:
    #   - 实例方法: def main(self) 且用 对象.main() 调用
    #   - 或更推荐: 测试代码直接写在 if __name__ == '__main__': 里, 不放类内部
    def main():
        # 1.准备语料库: 两条文档(下标0: "我喜欢编程", 下标1: "编程很有趣")
        documents = ["我喜欢编程", "编程很有趣"]

        # 2.创建检索器对象 -> 触发 __init__: 分词 + 建 BM25L 索引
        bm25 = Bm25Search(documents)

        # 3.模拟一次用户查询: "他喜欢编程"
        #   分词后命中 "喜欢""编程"(文档0) 和 "编程"(文档1),
        #   文档0 命中两个词, 理论得分更高 -> 预期最佳文档是 "我喜欢编程"
        query = "他喜欢编程"

        # 4.执行检索, 元组解包接收 (最佳文档, 最佳分数)
        doc, score = bm25.search(query)

        # 5.根据返回结果分支: 找到了打印详情; search 内部出错返回 None 走 else
        if doc:
            logger.info(f"查询结果:{doc}, 分数:{score}")
        else:
            logger.info("未找到相关文档")


# 程序入口: 只有"直接运行本文件"时才执行测试代码
# 被其他文件 import 时不会自动跑, 别人只需要 from xxx import Bm25Search 即可复用
if __name__ == '__main__':
    # 通过"类名.方法名"调用没有 self 的 main(原代码写法, 见上面的说明/修正)
    Bm25Search.main()
