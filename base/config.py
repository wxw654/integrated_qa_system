"""
【步骤2: 配置加载模块】
作用: 读取 config.ini 文件，把配置项封装成 Python 属性，供其他模块统一使用。
设计思想: "单一数据源" —— 所有模块都 from base.config import Config，只读一份配置。

其他模块怎么用?
    from base.config import Config
    conf = Config()
    print(conf.MYSQL_HOST)  # 直接通过属性访问，不用每次都去读文件
"""
import configparser  # Python内置模块，专门解析 .ini 格式文件
import os            # 用于拼接文件路径

# ========== 第一部分: 定位 config.ini 的绝对路径 ==========
# 为什么要这样算? 因为不管你在哪个目录运行 python，__file__ 都能定位到当前文件位置

# __file__ = "base/config.py"(相对路径)，abspath 转成绝对路径
# 例如: "C:/Users/.../integrated_qa_system/base/config.py"
current_file_path = os.path.abspath(__file__)

# dirname() 取文件所在目录: "C:/.../integrated_qa_system/base"
current_dir_path = os.path.dirname(current_file_path)

# 再取一次 dirname: "C:/.../integrated_qa_system"  <-- 项目根目录
project_root = os.path.dirname(current_dir_path)

# 拼接出 config.ini 的完整路径: "C:/.../integrated_qa_system/config.ini"
# 用 join 而不是手写 "/" ，因为 Windows 用 \ Linux 用 /，join 会自动处理
config_file_path = os.path.join(project_root, "config.ini")
print(config_file_path)  # 调试用，确认路径正确


class Config:
    """
    配置类: 把 ini 文件的 [段名]键 映射为 self.大写属性
    为什么用类而不是直接返回字典? -> 属性访问更直观(conf.MYSQL_HOST)，也方便IDE补全
    """

    def __init__(self, config_file_path=config_file_path):
        """
        参数说明:
            config_file_path: str, 要读取的 ini 文件路径
                - 默认值就是上面算好的绝对路径，一般不用手动传
                - 测试时可以传相对路径(如 "../../config.ini")
        """
        # 创建解析器对象(相当于一个"ini文件阅读器")
        self.config = configparser.ConfigParser()
        # read() 解析文件内容，encoding="utf-8" 防止中文乱码
        self.config.read(config_file_path, encoding="utf-8")

        # ========== 读取 MySQL 配置 ==========
        # get("段名", "键名", fallback=默认值)
        # fallback 的作用: 如果 ini 里没有这个配置项，就用默认值，防止报错
        self.MYSQL_HOST = self.config.get("mysql", "host", fallback="localhost")
        self.MYSQL_USER = self.config.get("mysql", "user", fallback="root")
        self.MYSQL_PASSWORD = self.config.get("mysql", "password", fallback="123456")
        self.MYSQL_DATABASE = self.config.get("mysql", "database", fallback="subjects_kg")

        # ========== 读取 Redis 配置 ==========
        # 为什么要缓存? -> BM25搜索时先查Redis，命中就不用再访问MySQL，提升响应速度
        self.REDIS_HOST = self.config.get("redis", "host", fallback="localhost")
        self.REDIS_PORT = self.config.get("redis", "port", fallback=6379)
        self.REDIS_PASSWORD = self.config.get("redis", "password", fallback="1234")
        self.REDIS_DB = self.config.get("redis", "db", fallback=0)

        # ========== 读取日志配置 ==========
        # 为什么日志路径也配置化? -> 不同环境日志目录不同(开发/测试/生产)
        self.LOG_FILE = self.config.get("logger", "log_file", fallback="logs/app.log")

        # 3.4milvus配置
        self.MILVUS_HOST = self.config.get("milvus", "host", fallback="localhost")
        self.MILVUS_PORT = self.config.get("milvus", "port", fallback="19530")
        self.MILVUS_COLLECTION_NAME = self.config.get("milvus", "collection_name", fallback="edurag_final")
        self.MILVUS_DATABASE_NAME = self.config.get("milvus", "databases_name", fallback="itcast")

        # 3.5 LLM配置
        self.DASHSCOPE_BASE_URL = self.config.get("llm", "dashscope_base_url",
                                                  fallback="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # self.DASHSCOPE_API_KEY= self.config.get("llm","dashscope_api_key")
        self.LLM_MODEL = self.config.get("llm", "model", fallback="qwen-plus")
        self.DASHSCOPE_API_KEY = self.config.get("llm", "dashscope_api_key", fallback="")

        # 3.6检索参数
        self.PARENT_CHUNK_SIZE = self.config.getint("retrieval", 'parent_chunk_size', fallback=800)
        self.CHILD_CHUNK_SIZE = self.config.getint("retrieval", 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint("retrieval", 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint("retrieval", 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint("retrieval", 'candidate_m', fallback=2)

        # 3.7 应用配置
        self.CUSTOMER_SERVICE_PHONE = self.config.get("app", "customer_service_phone", fallback="12345678")
        self.VALID_SOURCES = eval(self.config.get("app", "", fallback=' ["ai","java","test","ops","bigdata"]'))













if __name__ == '__main__':
    # 测试代码: 单独运行本文件时，验证配置是否正确读取
    # 注意: 这里用相对路径是因为当前工作目录是 base/，所以要回退两级
    config_path = "../../config.ini"

    config = Config(config_path)  # 实例化，触发 __init__ 读取文件

    # 打印所有配置项，确认读取成功
    # print(config.MYSQL_HOST)      # localhost
    # print(config.MYSQL_USER)      # root
    # print(config.MYSQL_PASSWORD)  # 123456
    # print(config.MYSQL_DATABASE)  # subjects_kg
    # print(config.REDIS_HOST)      # localhost
    # print(config.REDIS_PORT)      # 6379
    # print(config.REDIS_PASSWORD)  # 1234
    # print(config.REDIS_DB)        # 0
    # print(config.LOG_FILE)        # logs/app.log


    print(config.MILVUS_DATABASE_NAME)
    print(config.PARENT_CHUNK_SIZE)
    print(config.VALID_SOURCES)







