"""
strategy_selector.py —— 检索策略选择器

它解决的问题: 一个专业问题，该用哪种检索增强策略效果最好?
    四种候选策略: 直接检索 / 假设问题检索(HyDE) / 子查询检索 / 回溯问题检索
    交给大模型(LLM)根据问题语义自动判断，选完交给 RAGSystem 去执行对应检索

为什么需要它? 不同问题适合不同策略:
    简单事实题 -> 直接检索就够; 抽象题 -> HyDE; 多实体对比 -> 子查询; 复杂题 -> 回溯
    人工写规则难覆盖，用 LLM 判断更灵活

在整个系统中的位置:
    RAGSystem.generate_answer() -> select_strategy(query) -> 返回策略名
"""
import os, sys
from langchain.prompts import PromptTemplate  # LangChain 提示词模板工具，用于构造带变量的 LLM Prompt
from openai import OpenAI  # OpenAI 兼容 SDK，用于调用 DashScope 大模型接口

from base.config import Config  # 全局配置类，读取 API Key、Base URL、模型名称等配置项
from base.logger import logger  # 全局日志对象，用于记录策略选择过程与异常信息

#todo 定义策略选择类： 用于根据用户查询选择最合适的检索策略增强

class StrategySelector:
    """
    检索策略选择器。

    核心能力：接收用户查询文本，借助大语言模型（DashScope）从四种检索增强策略中
    选出最适合当前查询的一种，并返回策略名称字符串供上层 RAG 系统调度使用。
    四种候选策略：直接检索、假设问题检索（HyDE）、子查询检索、回溯问题检索。
    """

    def __init__(self):
        """
        初始化策略选择器。

        入参：无（self 除外）
        返回值：None
        异常说明：若 Config 中缺少必要配置项（如 DASHSCOPE_API_KEY），OpenAI 客户端构造时会抛出异常。
        """
        self.config = Config()  # 加载全局配置对象，后续读取模型名称等参数
        self.logger = logger  # 绑定日志对象，方便在当前实例中输出日志
        # 构造 OpenAI 兼容客户端：api_key 为 DashScope 密钥，base_url 为 DashScope 接口地址
        # 为什么用 OpenAI SDK? -> DashScope 提供了兼容 OpenAI 协议的接口，可直接复用生态
        self.openai_client = OpenAI(api_key=Config().DASHSCOPE_API_KEY, base_url=Config().DASHSCOPE_BASE_URL)

        # 预加载策略选择 Prompt 模板，避免每次调用时重复构造，提升性能
        self.strategy_prompt_template = self._get_strategy_prompt()

    def _get_strategy_prompt(self):
        """
        构造并返回策略选择专用的 PromptTemplate 对象（私有方法）。

        入参：无（self 除外）
        返回值：PromptTemplate 对象，包含策略说明文本与 {query} 占位符，供后续 format 填充。
        异常说明：无显式异常，LangChain 构造失败时由调用方感知。
        """
        #   定义私有方法，获取策略选择 Prompt 模板
        # PromptTemplate 的 template 字段包含四种策略的详细描述与示例，引导 LLM 直接输出策略名
        # input_variables 声明模板中需要外部填充的变量列表
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    def call_dashscope(self, prompt):
        """
        调用 DashScope 大模型接口，将格式化后的 Prompt 发送给 LLM 并获取策略名称。

        入参：
            prompt (str): 已经填充好用户 query 的完整提示词文本
        返回值：
            str: LLM 返回的策略名称文本，例如 "直接检索"、"假设问题检索" 等；
                 若调用失败或返回为空，则兜底返回 "直接检索"。
        异常说明：
            捕获所有 Exception（网络超时、API Key 错误等），记录错误日志后静默降级，不向上抛出。
        """
        try:
            # chat.completions.create：OpenAI SDK 的对话补全方法
            # model: 使用配置中的 LLM 模型名称；temperature=0.1 使输出更确定性，适合分类任务
            completion = self.openai_client.chat.completions.create(
                model=Config().LLM_MODEL,  #
                messages=[
                    {"role": "system", "content": "你是一个有用的问题分类助手"},
                    {"role": "user", "content": prompt}
                ]
                , temperature=0.1
            )

            # 2.提取返回值
            # choices[0].message.content 为模型生成的策略名称；若 choices 为空则兜底
            # print(f"completion:{completion}")
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            logger.error(f"DashScope API调用识别:{e}")
            return "直接检索"  # 兜底策略 "直接检索"

    def select_strategy(self, query):
        """
        根据用户输入的查询，选择最合适的检索增强策略（对外主入口方法）。

        入参：
            query (str): 用户的原始问题文本
        返回值：
            str: LLM 选定的检索策略名称（去除首尾空白），例如 "假设问题检索"
        异常说明：
            call_dashscope 内部已兜底处理，本方法不向外抛出异常。
        """
        # 1.格式化提示词模板：将用户 query 填入 PromptTemplate 的 {query} 占位符
        prompt = self.strategy_prompt_template.format(query=query)
        # 2.调用LLM：将格式化后的 prompt 发送给大模型，获取策略名称并去除首尾空白
        strategy = self.call_dashscope(prompt).strip()
        logger.info(f"为查询:{query},选择的检索策略是:{strategy}")
        return strategy


if __name__ == '__main__':
    # 本地测试: 验证不同问题能否被 LLM 分到正确的策略
    ss = StrategySelector()
    # ss.call_dashscope("今天天气怎么样?")  # 直接测试底层LLM调用
    # ss.select_strategy(query="什么是AI?")  # 预期: 直接检索(意图明确的事实题)
    # ss.select_strategy(query="人工智能专业和Java后端应用专业就业竞争力如何?")  # 预期: 子查询检索(多实体对比)
    ss.select_strategy(query="人工智能专业未来10年发展潜力如何?")  # 预期: 假设问题检索(抽象题)



















































