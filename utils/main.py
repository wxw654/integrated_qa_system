from utils.logger import setup_logger

logger = setup_logger("AI")

def process_data(data):
    logger.info(f"开始处理数据: {data}")

    if not  data:
        logger.error("数据为空,无法处理")
        return None
    logger.info("数据处理完成")
    return data.upper()

def main():
    logger.info("程序启动")
    result = process_data("it")
    if result:
        logger.info(f"处理结果: {result}")
    else:
        logger.error("处理失败")
    logger.info("程序结束")

if __name__ == '__main__':
    main()




































