# todo 1 导包
# todo 1.1 自定义模块导包
from base.logger import logger  # 导入日志工具,用于系统运行状态与错误信息记录
from base.config import Config  # 导入配置类,加载系统全局配置参数
from mysql_qa.db.mysql_client import MysqlClient  # 导入MySQL客户端,封装数据库增删改查操作
from mysql_qa.cache.redis_client import RedisClient  # 导入Redis客户端,用于热点数据缓存与加速检索
from mysql_qa.retrieval.bm25_search import BM25Search  # 导入BM25检索模块,实现关键词精准匹配召回
# from rag_qa.core.rag_system import RAGSystem  # !!!!!!!!!!提醒我改, 这里面核心的方法都需要加入历史的记录
from rag_qa.core.vector_store import VectorStore  # 导入向量存储类,管理Milvus向量知识库
from rag_qa.core.new_rag_system import RAGSystem  # 导入RAG系统类,实现检索增强生成全流程

# todo 1.2 第三方库与标准库导包
from openai import OpenAI  # 导入OpenAI兼容客户端,用于调用DashScope大模型API
import time  # 导入时间库,用于统计查询耗时、记录时间戳
import uuid  # 导入UUID库,生成全局唯一的会话ID   会产生128位唯一字符串
import pymysql  # 导入pymysql库,用于捕获MySQL数据库操作异常


# todo 2 定义集成问答系统类
# 作用: 融合BM25关键词检索、RAG向量检索、对话历史管理、大模型调用能力,实现多级召回的端到端问答服务
# 召回优先级: BM25精准匹配 > RAG向量检索 > 默认无答案提示
class IntegratedQASystem:
    # todo 2.1 类的初始化方法: 初始化各类组件客户端,构建完整问答系统链路
    def __init__(self):
        # 1. 初始化基础工具组件
        self.logger = logger  # 日志记录器,输出系统运行日志、告警与错误信息
        self.config = Config()  # 全局配置对象,加载所有系统参数(API密钥、模型名、学科列表等)

        # 2. 初始化存储与BM25检索组件
        self.mysql_client = MysqlClient()  # MySQL客户端,负责对话历史持久化与结构化知识库存储
        self.redis_client = RedisClient()  # Redis客户端,缓存热点数据,加速BM25检索性能
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)  # BM25检索器,基于词法匹配快速召回答案

        # 3. 初始化大模型API客户端
        try:
            # 3.1 基于配置的密钥与地址,初始化OpenAI兼容客户端(对接DashScope服务)
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,  # DashScope API密钥,从配置文件读取
                base_url=self.config.DASHSCOPE_BASE_URL  # DashScope API服务访问地址
            )
        except Exception as e:
            # 3.2 初始化失败时记录错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 3.3 抛出异常终止初始化,避免后续组件加载无效
            raise

        # 4. 初始化RAG检索增强组件
        self.vector_store = VectorStore()  # 向量存储实例,管理Milvus向量知识库的增删查
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)  # RAG系统实例,传入向量存储与大模型调用方法

        # 5. 初始化对话历史数据表(不存在则创建,存在则跳过)
        self.init_conversation_table()

    # todo 2.2 初始化对话历史表
    def init_conversation_table(self):
        """
        作用:在mysql中创建conversation ,持久化存储用户对话历史,支持多轮会话上下文关联
        设计会话表
        id ,
        session_id,
        question :
        answer
        时间戳
        :return:

        INDEX idx_session_id (session_id)   为session_id增加索引 , 加快会话历史查询速度
        """
        # 1.创建表
        try:
            self.mysql_client.cursor.execute("""
              CREATE TABLE IF NOT EXISTS conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(36) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        INDEX idx_session_id (session_id)
                    )  
            """)
            # 2.提交事务
            self.mysql_client.connection.commit()
            self.logger.info("对话历史初始化表成功")
        except Exception as e:
            self.logger.error(f"初始化对话历史表失败:{e}")
            raise

    # todo 2.3 调用DashScope大模型API(流式输出)
    # 作用: 向大模型发送提示词,以生成器形式逐token返回回答内容,支持前端流式实时展示
    # :param prompt: 输入给大模型的提示词文本
    # :return: 生成器,逐次返回单个token内容;流式结束后返回完整回答字符串
    def call_dashscope(self, prompt):
        try:
            # 1. 发起聊天补全请求,启用流式响应模式
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 指定使用的大模型名称,从配置文件读取
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统角色提示,设定模型基础行为
                    {"role": "user", "content": prompt},  # 用户角色提示,传入业务提示词
                ],
                timeout=30,  # 请求超时时间30秒,避免长时间阻塞进程
                stream=True  # 启用流式输出,逐token返回结果而非一次性返回
            )

            # 2. 初始化完整回答字符串,用于累积所有token
            collected_content = ""
            # 3. 遍历流式响应的每个数据块
            for chunk in completion:
                # 3.1 判断当前块是否包含有效文本内容
                if chunk.choices and chunk.choices[0].delta.content:
                    # 3.2 提取当前块的文本内容
                    content = chunk.choices[0].delta.content
                    # 3.3 累积到完整回答中
                    collected_content += content
                    # 3.4 以生成器形式返回当前token,供调用方实时展示
                    yield content
            # 4. 流式传输结束后,返回完整回答文本
            return collected_content
        except Exception as e:
            # 捕获所有调用异常,记录错误日志
            self.logger.error(f"LLM调用失败: {e}")
            # 返回错误提示文本
            return f"错误：LLM调用失败 - {e}"

    # todo 2.4  私有方法 ,获取最近N的对话
    # 作用: 从MySQL中查询指定会话的最近对话记录,按时间正序返回,用于构建多轮对话上下文
    # :param session_id: 会话唯一ID
    # :return: 对话历史列表,每个元素为包含question和answer的字典,按时间升序排列
    def _fetch_recent_history(self, session_id: str) -> list:
        try:
            # 1. 执行参数化查询SQL,按时间倒序取最近5条记录
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, 5))  # 传入会话ID与查询条数(固定保留5轮)

            # 2. 将查询结果转换为字典列表格式,便于后续使用
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 3. 反转列表,将倒序结果转为时间正序(旧->新),符合对话上下文逻辑
            return history[::-1]
        except pymysql.MySQLError as e:
            # 捕获查询异常,记录错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 查询失败返回空列表,不影响主问答流程
            return []

        # todo 2.5 更新会话历史记录
        # 作用: 插入新对话记录,同时清理超出轮数限制的旧记录,保证单会话仅保留最近5轮
        # :param session_id: 会话唯一ID
        # :param question: 用户问题文本
        # :param answer: 系统回答文本
        # :return: 更新后的对话历史列表
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
            try:
                # 1. 插入当前新的对话记录到数据库
                self.mysql_client.cursor.execute("""
                    INSERT INTO conversations (session_id, question, answer, timestamp)
                    VALUES (%s, %s, %s, NOW())
                """, (session_id, question, answer))  # 参数化插入,避免SQL注入风险

                # 2. 获取更新后的最近5轮对话历史
                history = self._fetch_recent_history(session_id)

                # 3. 删除当前会话中超出5轮的历史记录,控制数据体量
                # 子查询先查出最近5条记录的ID,删除不在其中的旧数据
                self.mysql_client.cursor.execute("""
                    DELETE FROM conversations
                    WHERE session_id = %s AND id NOT IN (
                        SELECT id FROM (
                            SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                            LIMIT %s
                        ) AS sub
                    )
                """, (session_id, session_id, 5))

                # 4. 提交事务,所有操作持久化生效
                self.mysql_client.connection.commit()
                # 5. 记录更新成功日志
                self.logger.info(f"会话 {session_id} 历史更新成功")
                # 6. 返回最新的对话历史列表
                return history
            except pymysql.MySQLError as e:
                # 捕获MySQL操作异常,记录错误
                self.logger.error(f"更新会话历史失败: {e}")
                # 回滚事务,撤销本次未提交的所有操作
                self.mysql_client.connection.rollback()
                # 抛出异常,通知上层调用处理
                raise
            except Exception as e:
                # 捕获其他未知异常,记录错误
                self.logger.error(f"更新会话历史意外错误: {e}")
                # 回滚事务
                self.mysql_client.connection.rollback()
                # 抛出异常
                raise

            # todo 2.6 获取指定会话的历史记录
            # 作用: 对外暴露的历史查询接口,内部封装私有查询方法
            # :param session_id: 会话唯一ID
            # :return: 对话历史列表

    def get_session_history(self, session_id: str) -> list:
        # 调用私有方法获取最近5轮对话历史
        return self._fetch_recent_history(session_id)
    # todo 2.7 清除指定会话的所有历史记录
    # 作用: 删除指定会话的全部对话数据,用于重置会话上下文
    # :param session_id: 会话唯一ID
    # :return: 清除成功返回True,失败返回False
    def clear_session_history(self, session_id: str) -> bool:
        try:
            # 1. 执行删除SQL,移除指定会话的所有记录
            self.mysql_client.cursor.execute("""
                    DELETE FROM conversations
                    WHERE session_id = %s
                """, (session_id,))
            # 2. 提交事务
            self.mysql_client.connection.commit()
            # 3. 记录清除成功日志
            self.logger.info(f"会话 {session_id} 历史已清除")
            # 4. 返回成功标志
            return True
        except pymysql.MySQLError as e:
            # 捕获删除异常,记录错误
            self.logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 返回失败标志
            return False



    # todo 2.8 核心问答查询方法
    # 作用: 多级召回策略的主入口,优先BM25精准匹配,无匹配则回退RAG检索,全程支持流式输出与会话历史
    # :param query: 用户输入的问题文本
    # :param source_filter: 学科过滤条件,限定检索的知识库范围,None表示不过滤
    # :param session_id: 会话ID,用于关联对话历史;None表示无历史会话
    # :return: 生成器,每次返回 (文本片段, 是否结束) 元组;文本片段为空且is_complete为True表示流结束
    def query(self, query, source_filter=None, session_id=None):
        # 1. 记录查询开始时间,用于统计总耗时
        start_time = time.time()
        # 2. 记录查询请求日志
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        # 3. 获取对话历史: 传入会话ID则查询历史,否则使用空列表
        history = self.get_session_history(session_id) if session_id else []

        # 4. 第一级召回: BM25关键词检索,获取匹配答案与是否需要RAG的标志
        # threshold=0.85: 相似度阈值,高于该值认为BM25答案可靠,直接返回
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        # 5. BM25命中可靠答案,直接返回结果
        if answer:
            # 5.1 记录BM25命中日志
            self.logger.info(f"MySQL答案: {answer}")
            # 5.2 存在会话ID时,将本次问答更新到历史记录中
            if session_id:
                self.update_session_history(session_id, query, answer)
            # 5.3 计算本次查询总耗时
            processing_time = time.time() - start_time
            # 5.4 记录耗时日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 5.5 一次性返回完整答案,is_complete标记为True表示回答结束
            yield answer, True

        # 6. BM25未命中可靠答案,触发第二级召回: RAG向量检索+大模型生成
        elif need_rag:
            # 6.1 记录回退RAG的日志
            self.logger.info("无可靠MySQL答案，回退到RAG")
            # 6.2 初始化完整答案字符串,用于累积流式输出
            collected_answer = ""
            # 6.3 调用RAG系统生成回答,逐token获取流式结果
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                # 6.4 累积每个token到完整答案
                collected_answer += token
                # 6.5 逐token返回给调用方,is_complete标记为False表示回答未结束
                yield token, False
            # 6.6 流式输出结束后,存在会话ID则更新对话历史
            if session_id:
                self.update_session_history(session_id, query, collected_answer)
            # 6.7 计算查询总耗时
            processing_time = time.time() - start_time
            # 6.8 记录耗时日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 6.9 返回空文本+结束标记,通知调用方流式输出完成
            yield "", True

        # 7. 两级召回均无有效结果,返回默认无答案提示
        else:
            # 7.1 记录未找到答案的日志
            self.logger.info("未找到答案")
            # 7.2 计算查询耗时
            processing_time = time.time() - start_time
            # 7.3 记录耗时日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 7.4 返回默认无答案文本,标记回答结束
            yield "未找到答案", True


# todo 3 命令行交互主函数
# 作用: 提供本地测试的命令行交互界面,演示集成问答系统的完整使用流程
def main():
    # 1. 初始化集成问答系统实例
    qa_system = IntegratedQASystem()
    # 2. 生成全局唯一的会话ID,标识本次交互会话
    session_id = str(uuid.uuid4())

    # 3. 打印欢迎与系统基础信息
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        # 4. 进入交互循环,持续接收用户输入
        while True:
            # 4.1 获取用户输入的问题,去除首尾空白字符
            query = input("\n输入查询: ").strip()
            # 4.2 输入exit则退出循环,结束程序
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break

            # 4.3 获取用户输入的学科过滤条件
            source_filter = input(
                f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            # 4.4 校验学科类别是否合法,不合法则置空并输出警告日志
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None

            # 4.5 打印答案前缀,启用实时刷新
            print("\n答案: ", end="", flush=True)
            # 4.6 初始化完整答案字符串
            answer = ""
            # 4.7 遍历查询生成器,逐token打印回答内容
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # token非空时打印,不换行
                    print(token, end="", flush=True)
                    # 累积答案内容
                    answer += token
                if is_complete:
                    # 回答结束时换行,退出遍历
                    print()
                    break

            # 4.8 获取并打印最近对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")

    except Exception as e:
        # 捕获全局异常,记录错误日志
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        # 无论程序是否异常,最终关闭MySQL数据库连接
        qa_system.mysql_client.close()

# todo 4 程序入口
if __name__ == "__main__":
    # 脚本作为主程序运行时,执行主交互函数
    main()