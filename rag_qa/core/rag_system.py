"""
rag_system.py —— RAG 系统核心调度类(整个 rag_qa 模块的"总指挥")

它在整个系统中的位置(接上一个 mysql_qa 模块):
    用户提问 -> main.py 接收 -> RAGSystem.generate_answer() 总调度
        1) 先用 BERT 分类器判断问题类型(通用咨询 / 专业咨询)
        2) 通用咨询: 不检索，直接让 LLM 回答(省资源)
        3) 专业咨询: 选检索策略 -> 向量库检索 -> 拼上下文 -> LLM 生成答案

四个核心组件如何协作:
    QueryClassifier   -> 判断问题走哪条路(小模型分类，快且便宜)
    StrategySelector  -> 专业问题选哪种检索增强策略(调 LLM 判断)
    VectorStore       -> 真正去 Milvus 拿相关文档(混合检索+重排)
    self.llm          -> 一个函数回调，把拼好的 prompt 发给大模型生成答案
"""
import os, sys
import time  # 用于统计一次查询的总耗时(性能监控)

from rag_qa.core.prompts import RAGPrompts                # 各种 Prompt 模板(答案生成/HyDE/子查询/回溯)
from rag_qa.core.query_classifier import QueryClassifier  # BERT 问题分类器(通用/专业二分类)
from rag_qa.core.strategy_selector import StrategySelector  # 检索策略选择器(调LLM选策略)
from rag_qa.core.vector_store import VectorStore          # 向量库封装(Milvus检索)
from base.config import Config                            # 全局配置
from base.logger import logger                            # 全局日志

# 计算 rag_qa 目录绝对路径，保证无论从哪个目录启动都能定位到 BERT 模型文件
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../rag_qa/core
rag_qa_path = os.path.dirname(current_dir)  # .../rag_qa
conf = Config()  # 模块加载时实例化一次，后续复用(避免重复读文件)


class RAGSystem:
    """
    RAG 系统核心类: 把"分类 -> 选策略 -> 检索 -> 生成"整条链路串起来。

    核心能力:
        - generate_answer(): 对外主入口，输入问题返回答案
        - retrieval_and_merge(): 根据策略调度检索并返回候选文档
        - 三种检索增强策略的私有实现(回溯/子查询/HyDE)
    """

    def __init__(self, vector_store, llm):
        """
        初始化 RAG 系统，注入依赖并构建各子组件。

        参数说明:
            vector_store (VectorStore): 向量库对象，负责实际检索。
                为什么外部传入而不是内部创建? -> 依赖注入，方便测试和复用连接
            llm (callable): 一个"函数"(不是字符串!)，接收 prompt 返回答案文本。
                实际传入的是 main.py 里的 call_dashscope 函数
                为什么传函数? -> 解耦，RAGSystem 不关心底层用的是哪家大模型

        返回值: None
        异常说明: 若 BERT 模型文件缺失，QueryClassifier 初始化时可能报错
        """
        self.vector_store = vector_store  # 向量检索组件
        self.llm = llm                    # LLM 调用回调函数
        # 答案生成用的 Prompt 模板(含 {context}{question}{phone} 占位符)
        self.rag_prompts = RAGPrompts.rag_prompt()

        # BERT 分类器模型路径: rag_qa/models/bert_query_classifier
        classifier_path = os.path.join(rag_qa_path, "models", "bert_query_classifier")
        # 加载微调好的 BERT，用于判断"通用咨询"还是"专业咨询"
        self.query_classifier = QueryClassifier(classifier_path)

        # 策略选择器: 专业问题时用它决定用哪种检索增强策略
        self.strategy_selector = StrategySelector()

    # 定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter):
        """
        回溯问题检索策略: 把复杂问题简化为更基础、更好检索的问题。

        适用场景: 问题太复杂/太具体，直接检索命中不了，先"回退"到基础问题再查
        执行流程: LLM生成简化问题 -> 用简化问题去向量库检索

        参数说明:
            query (str): 用户原始问题
            source_filter (str/None): 学科过滤条件(如"AI")，None表示全库检索

        返回值: list[Document] 检索到的文档列表，失败返回空列表
        """
        logger.info(f"使用回溯问题检索策略,查询{query}")
        # 拿到"简化问题"的 Prompt 模板(把复杂query变成简单query)
        backtracking_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            # ==========修复点：改为关键字传参 format(query=query)==========
            # self.llm(...) 调用大模型生成简化问题; .split("\n") 把多行结果拆成列表
            # (因为 hybrid_search 支持传入多个query的列表)
            simplified_query = self.llm(backtracking_prompt_template.format(query=query)).split("\n")
            logger.info(f"生成简化问题: {simplified_query}")
            # 用简化后的问题去检索: k=返回条数, source_filter=学科过滤
            return self.vector_store.hybrid_search_with_rerank(
                query=simplified_query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []  # 失败不抛异常，返回空列表保证主流程不中断

    # 定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter):
        """
        子查询检索策略: 把复杂问题拆成多个简单子查询，分别检索后合并去重。

        适用场景: 问题涉及多个实体/方面(如"比较A和B的优缺点")，需要分别检索
        执行流程: LLM拆子查询 -> 逐个检索 -> 合并去重

        参数说明:
            query (str): 用户原始问题
            source_filter (str/None): 学科过滤条件

        返回值: list[Document] 去重后的文档列表
        """
        logger.info(f"使用子查询检索策略,查询{query}")
        # 拿到"拆子查询"的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            # ==========修复点：改为关键字传参 format(query=query)==========
            # LLM 返回的是多行文本(每行一个子查询)
            subqueries_text = self.llm(subquery_prompt_template.format(query=query))
            # 按行拆分 + strip去空白 + 过滤掉空行，得到干净的子查询列表
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成子查询: {subqueries}")
            if not subqueries:
                logger.warning("子查询生成无效，请检查模型生成结果")
                return []

            # 逐个子查询检索，把结果都累加到 all_docs
            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    query=sub_q,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter,
                )
                all_docs.extend(docs)  # extend把这批文档逐个加入(不是嵌套)
                logger.info(f"子查询 {sub_q} 检索到 {len(docs)} 个文档")

            # 去重: 不同子查询可能检索到相同文档
            # 用 dict 以文档正文(page_content)为key，自动覆盖重复项
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到{len(all_docs)}个文档,去重后还剩:{len(unique_docs)}个文档")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    # 定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter):
        """
        假设文档检索(HyDE, Hypothetical Document Embeddings):
            先让 LLM 生成一个"假设答案"，再用这个假设答案去检索。

        为什么有效? 直接检索时"问题"和"文档"长得不像；但"假设答案"和真实文档
            文本分布更接近，用答案去匹配答案，检索更准。
        适用场景: 查询抽象，直接检索效果不佳时

        参数说明:
            query (str): 用户原始问题
            source_filter (str/None): 学科过滤条件

        返回值: list[Document] 检索到的文档列表
        """
        logger.info(f"使用假设文档检索策略,查询{query}")
        # 拿到"生成假设答案"的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            # ==========修复点：改为关键字传参 format(query=query)==========
            # LLM 生成假设答案，按行拆成列表后拿去检索
            hyde_answer = self.llm(hyde_prompt_template.format(query=query)).split("\n")
            logger.info(f"生成假设文档: {hyde_answer}")
            return self.vector_store.hybrid_search_with_rerank(
                query=hyde_answer,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.error(f"假设文档策略执行失败: {e}")
            return []

    def retrieval_and_merge(self, query, strategy, source_filter):
        """
        根据选定策略分发到对应的检索方法，并截取前 M 个作为最终上下文。

        参数说明:
            query (str): 用户问题
            strategy (str/None): 检索策略名; 传 None 时会现场调用 StrategySelector 选择
            source_filter (str/None): 学科过滤条件

        返回值: list[Document] 最终用于拼接上下文的文档(最多 CANDIDATE_M 个)

        为什么要截取 CANDIDATE_M? -> 检索可能返回很多，但 LLM 上下文窗口有限，
            只取排名靠前的 M 个，兼顾效果与成本
        """
        # 如果上层没传策略，就自己调策略选择器选一个(LLM判断)
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 根据策略名分发到对应的检索实现(策略模式)
        ranked_chunks = []
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == "子查询检索":
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query, source_filter)
        else:  # 默认分支: 直接检索(不增强，用原query直接查)
            logger.info(f"使用直接检索策略,查询{query}")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                query=query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
            )

        logger.info(f"策略:{strategy},检索到{len(ranked_chunks)}个候选文档")
        # 切片取前 CANDIDATE_M 个文档作为最终上下文
        final_context_docs = ranked_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终上下文文档:{len(final_context_docs)}")

        return final_context_docs

    def generate_answer(self, query, source_filter=None):
        """
        【对外主入口】接收用户问题，返回最终答案。整个 RAG 流程的总调度。

        参数说明:
            query (str): 用户原始问题
            source_filter (str/None): 学科过滤条件，默认 None 表示全库检索

        返回值: str 最终答案文本

        两条分支:
            通用咨询 -> 不检索，直接 LLM 回答(快、省资源)
            专业咨询 -> 完整 RAG 流程(选策略+检索+拼上下文+生成)
        """
        start_time = time.time()  # 记录开始时间，用于最后统计耗时
        logger.info(f"开始处理查询: {query}，学科过滤: {source_filter}")

        # 第一步: BERT 分类器判断问题类型(小模型，快且便宜)
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果: {query_category}")

        # 分支A: 通用咨询(如闲聊/常识) -> 不需要知识库，直接让 LLM 回答
        if query_category == "通用咨询":
            logger.info("查询<通用咨询>直接调用LLM生成答案")
            # context 传空字符串(因为没检索)，但仍需填入 phone 备用
            prompt_input = self.rag_prompts.format(
                question=query,
                context="",
                phone=conf.CUSTOMER_SERVER_PHONE
            )

            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用LLM调用出错: {e}")
                # LLM 挂了也要给用户一个兑底回复(附客服电话)
                answer = f"抱歉，我无法回答这个问题。请拨打我们的客服电话 {conf.CUSTOMER_SERVER_PHONE} 获取帮助。"

            processing_time = time.time() - start_time
            logger.info(f"查询处理完成，耗时: {processing_time:.2f}秒, 查询内容: {query}")
            return answer

        # 分支B: 专业咨询 -> 执行完整 RAG 流程
        logger.info("查询<专业咨询>，执行完整RAG流程")
        # 第二步: LLM 选择最合适的检索增强策略
        strategy = self.strategy_selector.select_strategy(query)
        # 第三步: 按策略检索并拿到上下文文档
        context_docs = self.retrieval_and_merge(query, strategy=strategy, source_filter=source_filter)

        # 第四步: 把多个文档正文拼接成一个上下文字符串
        if context_docs:
            # 用两个换行分隔各个文档，便于 LLM 阅读
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"最终上下文文档: {len(context_docs)}个文档，总长度: {len(context)}字")
        else:
            context = ''
            logger.info(f"未检索到文档，上下文字符串为空")

        # 第五步: 把"问题+上下文+电话"填入答案生成 Prompt，发给 LLM
        prompts_input = self.rag_prompts.format(question=query, context=context, phone=conf.CUSTOMER_SERVER_PHONE)
        try:
            answer = self.llm(prompts_input)
        except Exception as e:
            logger.error(f"调用LLM出错: {e}")
            answer = f"抱歉，我无法回答这个问题。请拨打我们的客服电话 {conf.CUSTOMER_SERVER_PHONE} 获取帮助。"

        processing_time = time.time() - start_time
        logger.info(f"查询处理完成，耗时: {processing_time:.2f}秒, 查询内容: {query}")
        return answer


if __name__ == '__main__':
    # 本地测试: 单独运行本文件验证 RAG 全流程
    vector_store = VectorStore()  # 初始化向量库连接
    # 注意: 这里用 StrategySelector 的 call_dashscope 方法当作 LLM 回调(测试用)
    # 实际项目中 main.py 传入的是自己定义的 call_dashscope 函数
    llm = StrategySelector().call_dashscope
    rag_system = RAGSystem(vector_store, llm)
    # "地球为什么是圆的" 会被分类为通用咨询，直接走 LLM 回答分支
    answer = rag_system.generate_answer(query="地球为什么是圆的", source_filter="")
    print(answer)
