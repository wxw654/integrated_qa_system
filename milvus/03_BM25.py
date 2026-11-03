import  jieba
from rank_bm25 import BM25L
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Bm25Search():

    def __init__(self,documents):
        self.documents = documents
        self.tokenized_documents = [jieba.lcut(doc) for doc in documents]
        self.bm25 = BM25L(self.tokenized_documents)
        logger.info("BM25模型初始化完成")

    def search(self, query):
        tokenized_query = jieba.lcut(query)
        try:
            scores = self.bm25.get_scores(tokenized_query)
            print(f"检索结果: {scores}")

            best_idx = scores.argmax()
            print(f"最佳文档索引: {best_idx}")

            best_score = scores[best_idx]

            best_doc = self.documents[best_idx]

            logger.info(f"检索完成，查询: {query} 最佳文档: {best_doc}, 最佳分数: {best_score}")

            return best_doc, best_score
        except Exception as e:
            logger.error(f"搜索出错: {e}")
            return None,0

    def main():
        documents = ["我喜欢编程", "编程很有趣"]

        bm25 = Bm25Search(documents)

        query = "他喜欢编程"

        doc, score = bm25.search(query)

        if doc:
            logger.info(f"查询结果:{doc}, 分数:{score}")
        else:
            logger.info("未找到相关文档")



if __name__ == '__main__':
    Bm25Search.main()




















