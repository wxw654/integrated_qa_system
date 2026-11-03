"""
==============================================================================
Python logging 日志模块学习总结(带详解注释版)
==============================================================================

案例:
    演示 python 标准库 logging 模块的 4 种 progressively 深入的用法(dm01~dm04)

大白话理解: 日志就是程序的"运行日记"
    print 的问题: 所有信息混在一起、分不清轻重、程序上线后没法关掉、没法存文件
    logging 的作用: 给每条信息"分级 + 打时间戳 + 决定输出到哪(屏幕/文件)"

------------------ 先搞懂几个核心概念(大白话) ------------------
    日志级别(Level):    信息的"严重程度等级",从低到高 5 档:
                        DEBUG < INFO < WARNING < ERROR < CRITICAL
                        (调试 < 日常运行 < 警告 < 错误 < 致命错误)
                        设置一个级别后, 只有 >= 该级别的日志才会被输出!
                        比如设置 INFO, 那么 DEBUG 就被过滤掉了

    记录器(Logger):     负责"产生"日志的对象, 相当于"写日记的笔"
                        logging.getLogger("名字") 获取, 不同名字是不同记录器

    处理器(Handler):    负责决定日志"送往哪里" —— 屏幕? 文件? 网络?
                        StreamHandler = 输出到控制台(屏幕)
                        FileHandler   = 输出到文件

    格式器(Formatter):  负责决定日志"长什么样子"(时间、级别、行号...)

    四者关系: Logger(产生) -> Handler(决定去哪) -> Formatter(决定格式)
==============================================================================
"""

# 导包: logging 是 python 标准库, 不需要额外 pip install
import logging


# =============================================================================
# todo 1 最简用法: basicConfig 一行配置 + 5 个级别的日志
# 【本节作用】认识 logging 最基本的用法和 5 个日志级别
# =============================================================================
def dm01():
    # 配置基本的日志设置
    # logging.basicConfig(): 对"根记录器(root logger)"做全局简单配置
    #   参 level: 全局日志级别, 低于这个级别的日志将被忽略
    #   level=logging.DEBUG 是最低级别 -> 所以 5 种日志全部都会打印出来
    #   注意: basicConfig 只在根记录器"还没有任何 Handler"时生效一次,
    #         重复调用不会覆盖(这也是为什么每个 dm 函数独立运行测试)
    logging.basicConfig(level=logging.DEBUG)

    # 获取日志记录器
    # logging.getLogger("Example1"): 获取(或创建)一个名字为 Example1 的记录器
    #   名字只是标识, 用于区分不同模块的日志来源(%(name)s 会打印它)
    logger = logging.getLogger("Example1")

    # 记录不同级别的日志(5 个级别从轻到重, 方法名就是级别名)
    logger.debug("这是调试信息，通常用于开发")          # 开发时排查用的细节信息
    logger.info("日常打印 程序运行正常")                 # 正常运行的一般信息
    logger.warning("程序可能有一些小问题")               # 警告: 不影响运行但需注意
    logger.error("发生错误")                            # 错误: 某功能失败了
    logger.critical("严重错误，程序可能崩溃")            # 致命: 程序可能要挂了

    # 运行输出(默认格式: 级别 - 名字 - 消息):
    # DEBUG    Example1:这是调试信息，通常用于开发
    # INFO     Example1:日常打印 程序运行正常
    # WARNING  Example1:程序可能有一些小问题
    # ERROR    Example1:发生错误
    # CRITICAL Example1:严重错误，程序可能崩溃


# =============================================================================
# todo 2 自定义日志格式: format 参数
# 【本节作用】让日志带上 时间、级别、文件名、记录器名、行号 —— 方便定位问题
# =============================================================================
def dm02():
    logging.basicConfig(
        level=logging.DEBUG,
        # format: 日志输出格式的"模板字符串", %(xxx)s 是占位符, 常用占位符:
        #   %(asctime)s   -> 时间, 如 2026-10-12 10:30:00,123
        #   %(levelname)s -> 级别名, 如 INFO / ERROR
        #   %(filename)s  -> 发出日志的 .py 文件名
        #   %(name)s      -> 记录器名字(就是 getLogger 传的那个名字)
        #   %(lineno)d    -> 发出日志的代码行号(%(xxx)d 表示整数)
        #   %(message)s   -> 日志正文
        # 注意: 占位符拼写必须严格正确! 原代码这里把 filename 写成了
        #       %(filename)s - %(name)s 顺序, 运行结果会一目了然
        format='%(asctime)s - %(levelname)s - %(filename)s - %(name)s - Line:%(lineno)d - %(message)s',
    )
    logger = logging.getLogger("Example2")
    logger.debug("这是调试信息，通常用于开发")
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")

    # 运行输出示例:
    # 2026-10-12 10:30:00,123 - DEBUG - 02_logging_学习笔记版.py - Example2 - Line:23 - 这是调试信息，通常用于开发


# =============================================================================
# todo 3 日志写入文件: filename / filemode / encoding
# 【本节作用】日志不打印到屏幕, 而是保存到 app.log 文件里(生产环境常用)
# =============================================================================
def dm03():
    logging.basicConfig(
        # 这里级别改为 INFO -> 运行后打开 app.log 会发现 DEBUG 那条没有写入!
        # (验证了"只输出 >= 设置级别"的规则)
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s  - %(message)s',
        # 参 filename: 日志文件路径(相对路径 = 写在当前工作目录下)
        #   一旦传了 filename, 日志就写文件, 默认不再输出到控制台
        filename='app.log',
        # 参 filemode: 文件打开模式, 和 python open() 一样:
        #   'a' = append 追加(保留旧日志, 新日志接在后面, 默认值)
        #   'w' = write 覆盖(每次启动都清空旧日志)
        filemode='a',
        # 参 encoding: 文件编码, 写中文必须指定 utf-8, 否则可能乱码
        #   (Windows 默认用系统编码 gbk, 不写 encoding 中文容易出问题)
        encoding='utf-8',
    )
    logger = logging.getLogger("Example3")
    logger.debug("这是调试信息，通常用于开发")   # DEBUG < INFO, 这条会被丢弃
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")

    # 小结: 控制台和文件二选一? 不是! 想"既要屏幕又要文件", 看 dm04 的组合用法


# =============================================================================
# todo 4 组件化用法(重点!): Logger + Handler + Formatter 自由组合
# 【本节作用】生产环境标准写法 —— 手动组装四大组件, 日志同时输出到屏幕和文件
#
# 【为什么不用 basicConfig?】
#   basicConfig 是"一把梭"的全局配置, 只能配一套;
#   组件化写法可以: 一个 logger 挂多个 handler, 每个 handler 还能有不同级别/格式
#   例如: 屏幕只显示 INFO 以上, 文件只存 ERROR 以上 —— basicConfig 做不到
# =============================================================================
def dm04():
    # 1.创建日志记录器(生产者)
    logger = logging.getLogger("Example4")
    logger.setLevel(logging.INFO)  # 设置记录器级别: 第一道门槛, INFO 以下直接不产生

    # 2.创建控制台处理器(运输工1: 送往屏幕)
    # logging.StreamHandler(): 不带参数时默认输出到 sys.stderr(控制台)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 第二道门槛: 控制台显示INFO及以上级别

    # 3.创建文件处理器(运输工2: 送往文件)
    # logging.FileHandler(文件名, encoding=...): 把日志写入指定文件
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 第二道门槛: 文件记录INFO及以上级别

    # 4.定义日志格式(化妆师: 决定日志长什么样)
    # logging.Formatter(): 参 format 就是模板字符串, 占位符同 dm02
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 5.为处理器设置格式(格式是挂在 handler 上的, 不是挂在 logger 上!)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 6.将处理器添加到记录器(logger 和 handler 在这里组装起来)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 7.记录日志
    #   流程: logger.debug() -> logger 级别(INFO)过滤 -> 分发给两个 handler
    #         -> 每个 handler 自己的级别再过滤 -> Formatter 排版 -> 输出
    logger.debug("这是调试信息，通常用于开发")   # < INFO, 屏幕和文件都没有它
    logger.info("日常打印 程序运行正常")          # 屏幕 + app.log 各一份
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")

    # 【经典坑】重复添加 handler 导致日志打印 N 遍:
    #   getLogger("Example4") 拿到的是同一个对象(名字相同即同一实例),
    #   如果 dm04() 被调用两次, handler 会挂两套, 每条日志输出两次。
    #   实际项目中一般用"模块级 get 一次 + if not logger.handlers 才添加"来防止


# 程序入口: __name__ == '__main__' 表示"直接运行本文件时"才执行下面代码
# (被别的文件 import 时不会执行, 方便复用这些函数)
if __name__ == '__main__':
    # 想测试哪个就取消哪个注释, 一次只跑一个!
    # 原因: basicConfig 全局只生效一次, 前面跑过的配置会影响后面的函数
    # dm01()
    # dm02()
    # dm03()
    dm04()
