"""
集成两个模块 mysqlqa + ragqa
"""
# 导包_公用模块
from base.logger import logger
from base.config import Config

# 导包_mysql模块
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search

# 导包_rag模块
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem

# 导包_通用工具
from openai import OpenAI
import time


class IntegratedQASystem:
    def __init__(self):
        # 1.日志
        self.logger = logger
        self.config = Config()

        # 2.初始化数据库和缓存
        self.mysql_client = MysqlClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        # 3.初始化大语言模型
        try:
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"大模型客户端初始化失败:{e}")
            raise

        # 4.初始化rag系统
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个靠谱的助手,给你消息好好回答"},
                    {"role": "user", "content": prompt}
                ]

            )
            return completion.choices[0].message.content if completion.choices else "错误:llm无响应"

        except Exception as e:
            self.logger.error(f"LLM调用失败:{e}")
            return f"错误:LLM调用失败:{e}"

    # 处理查询核心方法:用户的问题从这里进
    def query(self, query, source_filter=None):
        # 1.开始记录时间
        start_time = time.time()
        self.logger.info(f"开始处理问题:{query},来源过滤:{source_filter or '不限'}")
        # 2.调用BM25查Mysql里面的答案
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        # 3.根据bm25判断处理结果
        if answer:
            # 情况1:拿到靠谱的回答直接返回
            self.logger.info(f"BM25找到了靠谱的答案:{answer}")
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成,用时:{processing_time:.2f}秒")
            return answer
        elif need_rag:
            # 情况2:答案不靠谱,调用rag系统进行处理
            self.logger.info(f"BM25答案不靠谱,让RAG系统来处理")
            answer = self.rag_system.generate_answer(query, source_filter)
            self.logger.info(f"RAG系统生成答案:{answer}")
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成,用时:{processing_time:.2f}秒")
            return answer
        else:
            # 情况3 啥相关都没有 ,告诉用户没有答案
            self.logger.info(f"BM25没有知道任何答案")
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成,用时:{processing_time:.2f}秒")
            return "没有找到和你问题相关的问题,换一个问题在试试吧"


def main():
    """
    系统入口:用户可以在这里输入问题 ,选 过滤学科 ,然后看答案
    :return:
    """
    # 1.调用系统
    qa_system = IntegratedQASystem()
    qa_system.logger.info("系统初始化好了,可以开始进行提问了")
    try:
        print("\n -----------集成问答系统-----------")
        print(f"支持的来源:{qa_system.config.VALID_SOURCES}")
        print(f"用法:输入问题后直接回车查询,输入exiT 按回车键退出系统")

        while True:
            query = input("\n请输入您的问题").strip()
            if query.lower() == "exit":
                qa_system.logger.info("用户按照exit,准备退出系统")
                print("退出系统中.....")
                break

            sourece_filter = input(f"输入来源过滤( {'/'.join(qa_system.config.VALID_SOURCES)})按(Enter跳过)")
            if sourece_filter:
                if sourece_filter not in qa_system.config.VALID_SOURCES:
                    qa_system.logger.warning(f"用户输入无效来源:{sourece_filter}")
                    sourece_filter = None
                else:
                    qa_system.logger.info(f"用户选择了来源过滤:{sourece_filter}")

            answer = qa_system.query(query, sourece_filter)
            print(f"\n答案:{answer}")
    except Exception as e:
        qa_system.logger.error(f"系统出错了:{e}可能需要重启系统!")
        print(f"处理问题时出错:{e},请联系管理员")


if __name__ == '__main__':
    main()
