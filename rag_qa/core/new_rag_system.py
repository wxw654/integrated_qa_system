# todo 1 导包
import os, sys
import time
from rag_qa.core.prompts import RAGPrompts  # 导入rag相关的提示词模板
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器
from rag_qa.core.vector_store import VectorStore  # 导入向量数据库(存储和检索向量文档)
from base.config import Config
from base.logger import logger

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()


# todo 2 定义RAGSystem类 :封装RAGQA系统所需要的核心组件(向量库,大模型,分类器)
class RAGSystem:
    # 2.1 初始化方法 :创建ragqa系统所需要核心组件
    def __init__(self, vector_store, llm):
        # 设置向量数据库对象 : 后续检索相关文档
        self.vector_store = vector_store
        # 设置大语言模型调用函数:用于生成答案,策略选择
        self.llm = llm
        # 设置rag提示词模板:定义生成答案固定格式
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 设置查询分类器(意图识别模型)
        classifier_path = os.path.join(rag_qa_path, "./models", "bert_query_classifier")
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        # 设置分类器对象
        self.strategy_selector = StrategySelector()

    # 回溯问题检索
    def _retrieve_with_backtracking(self, query, source_filter):
        """
        针对复杂查询 ,先生成简化"回溯"问题,再用回溯问题检索文档(解决复杂查询直接检索,效果差的问题)
        :param query:
        :return:
        """
        # 1.获取回溯问题生成提示词模板: 定义如何将复杂查询简化为基础问题
        logger.info(f"使用回溯策略进行检索,查询信息为:{query}")
        backtracking_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            # 2.调用大语言模型生产回溯问题:用户输入问题,针对用户输入的问题进行改写
            simplified_query = self.llm(backtracking_prompt_template.format(query)).strip()
            logger.info(f"生成简化问题:{simplified_query}")

            # 3.调用向量数据库进行检索
            return self.vector_store.hybrid_search_with_rerank(
                query=simplified_query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )

        except Exception as e:
            logger.error(f"回溯问题策略执行失败:{e}")
            return []

    # 子查询检索
    def _retrieve_with_subqueries(self, query, source_filter):
        """
        针对涉及多个实体/多方面复杂问题 ,拆分为多个简单子查询->分别检索后合并去重
        :param query:
        :return:
        """
        logger.info(f"使用子查询策略进行检索,查询信息为:{query}")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            #  调用大于模型生成子查询
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            # 拆分子查询 : 根据换行符进行拆分,获取子查询列表
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成子查询为:{subqueries}")

            # 检查子查询的有效性 : 若未生成子查询, 记录警告返回空列表
            if not subqueries:
                logger.warning("子查询生成无效,请检查模型生成结果")
                return []

            # 遍历子查询 , 分别检索并收集结果
            all_docs = []
            for sub_q in subqueries:
                # 用子查询调用向量数据库检索
                docs = self.vector_store.hybrid_search_with_rerank(
                    query=sub_q,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询:{sub_q},查询结果:{len(docs)}个文档")

            # 对所有的检索结果去重 : 基于文档内容(page_content)去重
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到{len(all_docs)}个文档,去重后还剩:{len(unique_docs)}个文档")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略执行失败:{e}")
            return []

    # 假设问题检索
    def _retrieve_with_hyde(self, query, source_filter):
        """
        使用HyDE策略检索文档->解决抽象查询直接检查匹配度低问题
        :param query:
        :return:
        """
        # 1.获取假设答案生成提示词模板
        logger.info(f"使用HyDE策略进行检索,查询的信息为:{query}")
        hyde_prompt_template = RAGPrompts.hyde_prompt()

        try:
            hyde_answer = self.llm(hyde_prompt_template.format(query)).strip()
            logger.info(f"生成的假设答案为:{hyde_answer}")
            # 用假设答案调用向量数据库检索
            docs = self.vector_store.hybrid_search_with_rerank(
                query=hyde_answer,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )
            return docs
        except Exception as e:
            logger.error(f"HyDE策略执行失败:{e}")
            return []

    # 核心方法: 根据检索策略 检索文档 ,并合并/筛选最终的上下文
    def retrieval_and_merge(self, query, source_filter, strategy):
        """
        定义方法 ,检索并合并相关文档 ,即:统一入口,根据指定的策略调用对应的检索方法,筛选最终生成答案的上下文文档
        :param query:用户的查询文本
        :param source_filter:检索来源过滤
        :param strategy:策略筛选器
        :return:  list[Document] ,筛选最终上下文文档列表
        """
        # 1.未指定策略,自动选择策略检索
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 2.根据策略调用检索方法,获取候选文档列表
        # 2.1
        ranked_chunks = []  # 初始化候选文档列表
        # 2.2判断策略,调用对应检索方案
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == "子查询检索":
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query, source_filter)
        else:
            logger.info(f"使用直接检索策略,查询{query}")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                query=query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )

        # 3.筛选最终上下文文档 ,截取前M个
        logger.info(f"策略:{strategy},检索到{len(ranked_chunks)}个候选文档")
        final_context_docs = ranked_chunks[:conf.CANDIDATE_M]
        # 4.打印日志返回结果
        logger.info(f"最终上下午文档:{len(final_context_docs)}")
        return final_context_docs

    def generate_answer(self, query, source_filter=None,history=None):
        """
        RAGQA系统对外核心接口:接收用户查询, 自动完成 查询分类 策略选择 文档检索 答案生成  全流程
        :param query: 用户的原始查询文本
        :param source_filter:检索来源过滤
        :return:生成最终的答案
        """
        # 1.记录时间
        start_time = time.time()
        logger.info(f"开始处理查询:{query},学科过滤:{source_filter}")

        # [新增代码]
        #验证历史
        if history is not None and not  isinstance(history,list):
            logger.warning(f"无效的历史格式:{type(history)},忽略历史")
            history=[]
        elif history:
            history=history[-5:]

        #构建历史的上下文
        history_context=''
        if history:
            history_context="\n".join([f"Q:{h['question']}\nA:{h['answer']}" for h in history])
            logger.info(f"使用对话历史:{history_context[:50]}")

        # 2.调用查询分类器 ,判断查询类型 (通用知识 ,专业咨询)
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果:{query_category}")
        # 3.对查询结果进行判断
        if query_category == "通用咨询":
            logger.info(f"查询<<通用咨询>>,直接调用LLM生成答案")
            prompt_input = self.rag_prompt.format(
                context="",  # 通用知识无需上下文
                history=history_context,
                question=query,
                phone=conf.CUSTOMER_SERVICE_PHONE
            )
            try:
                # 调用大模型生成答案
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用LLM失败:{e}")
                answer = f"抱歉,您在处理通用知识的时候遇到问题,请稍后再试或者联系人工客服:{conf.CUSTOMER_SERVICE_PHONE}"
            # 4.记录通用知识处理结果(耗时)
            processing_time = time.time() - start_time
            logger.info(f"处理通用知识查询结果,耗时:{processing_time:.2f},查询内容:{query}")
            return answer
        # 4 .判断专业咨询
        logger.info("查询为<<专业咨询>>,执行完整RAG流程")
        # 4.1 选择检索策略:调用策略选择器确定最合适的检索策略
        strategy = self.strategy_selector.select_strategy(query)
        # 4.2 检索相关的文档
        context_docs = self.retrieval_and_merge(query, source_filter=source_filter, strategy=strategy)
        # 4.3 构建上下文  文本
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"最终上下文文档:{len(context_docs)}个,总长度:{len(context)}字")
        else:
            # 若未检索到文档, 上下文空字符串
            context = ''
            logger.info(f"未检索到文档,上下文字符串为空")
        # 4.4构建LLM提示词,拼接上下文文本
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )
        try:
            # 4.5调用大模型生成答案
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用LLM失败:{e}")
            answer = f"抱歉,处理您的专业咨询时出错 ,请稍后再试或者联系人工客服:{conf.CUSTOMER_SERVICE_PHONE}"

        processing_time = time.time() - start_time
        logger.info(f"查询处理完成,耗时:{processing_time:.2f}s,查询内容为:{query}")
        return answer

if __name__ == '__main__':
    vector_store=VectorStore()
    llm=StrategySelector().call_dashscope
    rag_system=RAGSystem(vector_store,llm)
    # answer=rag_system.generate_answer(query="地球为什么是圆的?",source_filter="ai")
    answer = rag_system.generate_answer(query="人工智能就业课好学吗?难吗?学哪些知识?", source_filter="ai")
    print(answer)