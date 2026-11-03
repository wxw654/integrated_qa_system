"""
【步骤3: 日志模块(双路输出)】
作用: 统一配置日志记录器，同时输出到"控制台"和"文件"。
为什么需要日志? -> 后端服务没有界面，日志是唯一的"眼睛，用于排查问题、记录执行流程

"双路输出"设计思想:
    1. 控制台输出: 开发时实时看到运行状态
    2. 文件输出: 生产环境保存历史，事后排查

其他模块怎么用?
    from base.logger import logger
    logger.info("操作成功")
    logger.error("出错了: xxx")
"""
import logging  # Python内置日志框架
import os

# ========== 定位日志文件路径(和 config.py 同样的思路) ==========
module_path = os.path.abspath(__file__)       # .../base/logger.py
# print(module_path)

base_path = os.path.dirname(module_path)      # .../base
# print(base_path)

current_path = os.path.dirname(base_path)     # .../integrated_qa_system (项目根)
# print(current_path)

# 日志文件最终路径: .../integrated_qa_system/logs/app.log
log_file = os.path.join(current_path, "logs\\app.log")
# print(log_file)



def setup_logger(log_file=log_file):
    """
    初始化日志记录器(只调用一次，返回全局统一的 logger 对象)

    参数说明:
        log_file: str, 日志文件存储路径
            - 默认用上面算好的绝对路径
            - 为什么可配置? -> 测试时可以写到临时目录

    返回值:
        logging.Logger 对象，其他模块直接用它的 .info()/.error() 等方法

    为什么封装成函数而不是直接写在全局? -> 可以传参数复用，也方便单元测试
    """
    # 确保 logs/ 目录存在，不存在就创建(exist_ok=True 表示已存在也不报错)
    # 为什么需要? -> 第一次运行时 logs/ 目录可能还没有，FileHandler 会自动创建文件但不会创建目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建日志记录器，名字"EduRAG"是项目标识
    # getLogger() 是单例模式: 同名多次调用返回同一个对象，避免重复创建
    logger = logging.getLogger("EduRAG")
    logger.setLevel(logging.INFO)  # 设置最低级别为 INFO(DEBUG < INFO < WARNING < ERROR < CRITICAL)

    # ---------- 创建"控制台处理器" ----------
    # 作用: 把日志打印到终端(命令行窗口)，方便开发时实时观察
    console_handler = logging.StreamHandler()  # 默认输出到 sys.stderr
    console_handler.setLevel(logging.INFO)     # 这个处理器只处理 INFO 及以上级别

    # ---------- 创建"文件处理器" ----------
    # 作用: 把日志写入文件，用于持久化保存
    # mode='a': 追加模式(append)，不会覆盖之前的日志
    # encoding='utf-8': 支持中文日志内容
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # ---------- 定义日志格式 ----------
    # %(asctime)s: 时间戳    %(levelname)s: 级别(INFO/ERROR)
    # %(name)s: 记录器名称   %(message)s: 你传入的日志内容
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(name)s-%(message)s")

    # 把格式应用到两个处理器上
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # ---------- 添加处理器(避免重复添加) ----------
    # 为什么要判断? -> 如果多次 import 这个模块，setup_logger 可能被调用多次
    #                 不判断会导致日志重复打印(每次多加一个 handler)
    if not logger.handlers:
        logger.addHandler(console_handler)  # 绑定控制台输出
        logger.addHandler(file_handler)     # 绑定文件输出

    return logger



# 模块加载时就执行一次，生成全局唯一的 logger 实例
# 其他文件直接 "from base.logger import logger" 就能用
logger = setup_logger()
# print(logger)  # 调试用: <Logger EduRAG (INFO)>














