"""
main.py 是 EduRAG 系统的运行入口，提供两种运行模式：
    数据处理模式：加载并向量化文档，构建向量数据库，支持多学科目录处理。
    查询模式：通过命令行交互式回答用户查询，支持学科过滤。

两种模式为什么要分开? 一次性建库(数据处理) vs 高频查询(交互)，职责分离
    数据处理模式(query_model=False): 把文档写入 Milvus，只需跑一次
    查询模式(query_model=True): 日常使用，接收问题调用RAG生成答案

启动流程: 初始化LLM客户端 -> 连接向量库 -> 根据模式进入对应分支
"""

# todo 1 导包
import os, sys  # os: 拼接/判断目录路径; sys: 路径操作备用

from base.logger import logger  # 全局日志对象，记录各阶段运行信息
from base.config import Config  # 全局配置类，读取 Milvus、LLM、学科目录等配置项
from rag_qa.core.document_processor import process_documents  # 文档处理函数：加载、切分文档为父子块
from rag_qa.core.vector_store import VectorStore  # 向量存储类：封装 Milvus 集合的写入与检索
from rag_qa.core.rag_system import RAGSystem  # RAG 系统核心类：组合检索策略与 LLM 生成答案
from openai import OpenAI  # OpenAI 兼容 SDK，用于调用 DashScope 大模型接口

conf = Config()  # 实例化全局配置对象，后续统一通过 conf 读取各项配置


def main(query_model=True, directory_path="data"):
    """
    ragqa函数入口: 控制两种运行模式
        数据处理模型:query_model=False ,解析指定目录文档,分割文档块,并且存入向量数据库
        交互式查询模块:query_model=True ,接收用户问题 ,调用rag系统生成答案,并且展示
    :param query_model:  运行模式标识  True:查询模式   False:数据处理模式
    :param directory_path:数据处理模型所需要的文件路径
    :return:
    """

    try:
        # 初始化Dashscope API客户端
        # OpenAI: api_key 为 DashScope 密钥，base_url 为 DashScope 兼容接口地址
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f"初始化openai客户端失败(请检查API Key 或 Base Url):{e}")
        # 查询模式依赖 LLM，客户端不可用则直接退出；数据处理模式不依赖 LLM，可置 None 继续
        if query_model:
            print("错误:无法初始化大语言模型客户端,无法进入查询模式")
            return
        # 数据处理模型不需要LLM,可以继续
        client = None

    # 定义LLM调用函数,封装大模型调用逻辑 ,供rag系统使用
    def call_dashscope(prompt):
        """
        内部函数：调用 DashScope 大模型生成答案，作为 LLM 回调传入 RAGSystem。

        入参：
            prompt (str): 待发送给大模型的完整提示词文本
        返回值：
            str: 大模型生成的答案内容；若客户端不可用或调用失败，返回错误提示字符串。
        异常说明：
            捕获所有 Exception 并记录日志，不向调用方抛出，保证上层流程不中断。
        """
        # 客户端初始化失败时（client 为 None），无法调用 LLM，直接返回错误信息
        if not client:
            logger.error("LLM客户端初始化失败,无法调用call_dashscope")
            return f"错误:LLM客户端不可用"

        try:
            # chat.completions.create：OpenAI SDK 对话补全接口
            # model: 指定使用的 LLM 模型名称；messages: 系统角色 + 用户角色的对话列表
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[

                    {"role": "system", "content": "你是一个有用助手"},
                    {"role": "user", "content": prompt}
                ]
            )
            # 校验返回值：choices 非空且首条消息存在时提取内容，否则返回错误提示
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logger.error("LLM API调用无效应或者空消息")
                return "错误LLM返回无响应"
        except Exception as e:
            logger.error(f"LLM API 调用失败:{e}")
            return f"错误:调用LLM失败-{e}"

    # 3.初始化向量数据库,连接milvus ,用于存储和检索文档向量
    # VectorStore: 封装 Milvus 连接，collection_name 指定集合名，host/port/database 指定服务地址与库名
    try:
        vector_store = VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
        )

    except Exception as e:
        logger.error(f"初始化:VectorStore失败(请检查milvus连接配置):{e}")
        print(f"错误:无法连接到向量数据库,程序无法继续执行")
        return  # 向量库不可用则两种模式均无法工作，直接退出

    # 4.进入执行模式判断
    if not query_model:
        # ------数据处理模模式------
        logger.info("进入数据处理模式........")
        total_chunks_added = 0  # 计数器
        # 遍历配置中所有合法学科来源，拼接对应数据目录路径逐一处理
        for source_dir in conf.VALID_SOURCES:
            dir_path = os.path.join(directory_path, f"{source_dir}_data")  # 例：data/AI_data
            if os.path.exists(dir_path):  # 目录存在才处理，避免报错
                logger.info(f"开始处理目录:{dir_path}")
                try:
                    # process_documents: 加载目录下所有文档并按父子块切分
                    # dir_path: 待处理目录; PARENT_CHUNK_SIZE/CHILD_CHUNK_SIZE: 父/子块大小; CHUNK_OVERLAP: 块间重叠字符数
                    chunks = process_documents(
                        dir_path,
                        conf.PARENT_CHUNK_SIZE,
                        conf.CHILD_CHUNK_SIZE,
                        conf.CHUNK_OVERLAP,
                    )
                    # 切分结果非空则批量写入向量库，并累加计数器
                    if chunks:
                        vector_store.add_documents(chunks)  # 将文档块向量化后写入 Milvus
                        total_chunks_added += len(chunks)
                        logger.info(f"成功处理目录{dir_path},添加了{len(chunks)}个文档块")
                    else:
                        logger.info(f"目录{dir_path}未发现有效的文档或者处理结果为空")
                except Exception as e:
                    # 单个目录处理失败不中断整体流程，记录错误后继续下一个目录
                    logger.error(f"处理目录:{dir_path}时出错:{e}")
            else:
                logger.warning(f"目录{dir_path}不存在,跳过处理")
        logger.info(f"数据处理完成,共添加了{total_chunks_added}个文档块到向量存储")
    else:
        # -------交互查询模式-------
        logger.info("进入交互式查询模式........")
        # 查询模式必须有可用的 LLM 客户端，否则直接退出
        if not client:
            print("错误,查询模型需要语言模型客户端,但初始化失败")
            return

        logger.info("进入交互查询模式.....")
        try:
            # RAGSystem: 初始化 RAG 核心系统，注入向量存储和 LLM 调用回调函数
            # 为什么传 call_dashscope 函数而不是客户端对象? -> 依赖倒置，RAGSystem 只需一个"能传prompt返回答案"的函数
            rag_system = RAGSystem(vector_store, call_dashscope)
        except Exception as e:
            logger.error(f"初始化RAGSystem失败:{e}")
            return

        valid_sources=conf.VALID_SOURCES  # 获取支持的学科类别列表，用于提示用户输入过滤条件
        print("\n 欢迎使用EduRag交互查询系统")
        print(f"支持的学科类别:{valid_sources}")
        print("请输入您的问题或者exit退出")

        # 主交互循环：持续接收用户输入，直到输入 exit 才退出
        while True:
            query=input("\n请输入您的问题")
            if query.lower() =="exit":  # 不区分大小写判断退出指令
                logger.info("用户退出查询模式")
                print("再见")
                break

            # 接收可选的学科过滤条件，直接回车则不过滤（source_filter 保持 None）
            source_filter_input=input(f"请输入学科类别(  {'/'.join(valid_sources)})  直接回车默认不过滤:").strip()
            source_filter =None
            if source_filter_input:
                if source_filter_input in valid_sources:  # 输入在合法学科列表内则生效
                    source_filter=source_filter_input
                    logger.info(f"用户选择了学科过滤:{source_filter}")
                else:  # 输入非法学科名则记录警告并忽略过滤
                    logger.warning(f"无效的学科类别:{source_filter_input},将不过滤")
                    print(f"提示:输入的学科 {source_filter_input}无效,不过滤")

            try:
                print("正在生产答案,请稍等......")
                # generate_answer: RAG 系统主入口，执行检索+策略增强+LLM生成
                # query: 用户原始问题; source_filter: 可选的学科过滤条件（None 表示全库检索）
                answer=rag_system.generate_answer(query,source_filter=source_filter)
                print("*"*30)
                print(f"问题:{query}")
                print(f"answer:{answer}")
                print("*"*30)
            except Exception as  e:
                # 单次查询失败不中断循环，记录错误并提示用户重试
                logger.error(f"处理查询:{query}时失败:{str(e)}")
                print(f"抱歉,处理您的问题时遇到了错误 ,请稍后重试或者连接管理员")
if __name__ == '__main__':
    # 演示: 根据要运行的模式切换注释
    # 查询模式: 交互式输入问题，调用RAG回答
    # main(query_model=True)

    # 数据处理模式: 把 data/ 下各学科文档向量化写入 Milvus(只需跑一次)
    main(query_model=False)

