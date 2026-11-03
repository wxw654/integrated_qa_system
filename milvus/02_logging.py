import logging

def dm01():
    # 配置基本的日志设置
    logging.basicConfig(level=logging.DEBUG)

    # 获取日志记录器
    logger = logging.getLogger("Example1")

    # 记录不同级别的日志
    logger.debug("这是调试信息，通常用于开发")
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")

def dm02():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(filename)s - %(name)s - Line:%(lineno)d - %(message)s',
    )
    logger = logging.getLogger("Example2")
    logger.debug("这是调试信息，通常用于开发")
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")
def dm03():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s  - %(message)s',
        filename='app.log',
        filemode='a',
        encoding='utf-8',
    )
    logger = logging.getLogger("Example3")
    logger.debug("这是调试信息，通常用于开发")
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")

def dm04():
    # 创建日志记录器
    logger = logging.getLogger("Example4")


    logger.setLevel(logging.INFO)  # 设置记录器级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别

    # 创建文件处理器
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 文件记录INFO及以上级别

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 为处理器设置格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 记录日志
    logger.debug("这是调试信息，通常用于开发")
    logger.info("日常打印 程序运行正常")
    logger.warning("程序可能有一些小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")










if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()










































































