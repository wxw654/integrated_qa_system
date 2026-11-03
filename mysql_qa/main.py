from db.mysql_client import MysqlClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search
from base.logger import logger
import time


class MySQLQASystem:
    def __init__(self):
        self.mysql_client = MysqlClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        self.logger = logger

    def query(self, query):
        start_time = time.time()
        self.logger.info(f"用户查询开始: {query}")
        # BM25检索，阈值0.85
        answer, _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            self.logger.info(f"用户查询结束, MySQL答案: {answer}")
        else:
            self.logger.info("用户查询结束, BM25检索无匹配结果，需要调用RAG系统")
            answer = "BM25检索未匹配到相关答案，请调用RAG系统"

        process_time = time.time() - start_time
        self.logger.info(f"用户查询处理耗时: {process_time:.2f}s")
        return answer


def main():
    mysql_system = None
    try:
        # 实例化MySQL问答系统
        mysql_system = MySQLQASystem()
        print("\n欢迎使用MySQL问答系统!")
        print("输入查询进行问答,输入'exit'退出.")
        # 循环交互
        while True:
            query = input("\n输入查询：").strip()
            # 空输入跳过
            if not query:
                print("输入不能为空，请重新输入！")
                continue
            # 判断退出指令，大小写不敏感
            if query.lower() == "exit":
                logger.info("退出MySQL问答系统")
                print("再见")
                break
            # 调用问答
            answer = mysql_system.query(query)
            print(f"\n答案:{answer}")
    # 捕获所有异常，记录错误日志
    except Exception as e:
        logger.error(f"系统发生错误:{e}", exc_info=True)
    # finally：无论正常退出还是报错，关闭mysql连接，防止连接泄漏
    finally:
        if mysql_system is not None:
            mysql_system.mysql_client.close()
            logger.info("MySQL连接已关闭")


if __name__ == '__main__':
    main()