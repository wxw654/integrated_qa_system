"""
【步骤4: MySQL客户端 - 增删改查】
作用: 封装对 MySQL 数据库的所有操作(建表、导入数据、查询问题、查询答案)

在整个系统中的位置:
    CSV原始数据 -> insert_data()导入MySQL -> fetch_questions()给BM25建索引
                                         -> fetch_answer()命中后取答案

前置条件(手动在MySQL命令行执行):

show databases ;
create database if not exists subjects_kg;
use subjects_kg;
show tables ;

"""
import pandas as pd   # 用于读取CSV文件，迭代每一行数据
import pymysql        # MySQL的Python驱动，提供连接数据库、执行SQL的能力

from base.logger import logger   # 步骤3的日志模块，记录操作状态
from base.config import Config   # 步骤2的配置模块，获取数据库连接参数


class MysqlClient(object):
    """
    MySQL操作客户端类
    为什么用类封装? -> 连接只需要建立一次，多个方法共享同一个connection和cursor
    """

    def __init__(self):
        """
        初始化: 建立数据库连接
        为什么在 __init__ 就连接? -> 后续所有方法都需要用 self.connection，提前准备好
        """
        self.logger = logger
        conf = Config()  # 读取 config.ini 中的 MySQL 配置

        try:
            # pymysql.connect() 建立TCP连接，参数解释:
            #   host:     数据库服务器地址
            #   user:     登录用户名
            #   password: 登录密码
            #   database: 要使用的库名
            #   charset:  utf8mb4 支持中文 + emoji，防止乱码
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.MYSQL_DATABASE,
                charset='utf8mb4',
            )

            # cursor(游标): 执行SQL语句并获取结果的对象
            # 为什么需要单独的cursor? -> connection是"电话线路"，cursor是"具体说话的人"
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL数据库连接成功")
        except pymysql.MySQLError as e:
            # 连接失败时记录错误并抛出异常，让调用方知道出了问题
            self.logger.error(f"MySQL数据库连接失败: {e}")
            raise

    def create_table(self):
        """
        建表: 创建 jpkb 表(学科知识库)
        为什么要建表? -> insert_data() 需要有表才能插入数据

        表结构:
            id:            自增主键，唯一标识每条记录
            subject_name:  学科名称(如"计算机")
            question:      问题文本(最长3000字)
            answer:        答案文本(最长5000字)
        """
        create_table_sql = """
        create table if not exists jpkb (
        id int auto_increment primary key,
        subject_name varchar(50),
        question varchar(3000),
        answer varchar(5000)
    );
        """
        try:
            self.cursor.execute(create_table_sql)  # 执行SQL
            self.connection.commit()  # 提交事务(DDL语句需要commit才生效)
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"创建表失败: {e}")
            raise

    def insert_data(self, csv_path):
        """
        导入数据: 读取 CSV 文件，批量插入到 MySQL 的 jpkb 表

        参数说明:
            csv_path: str, CSV文件的路径
                例如: "../data/JP学科知识问答.csv"
                CSV列名: 学科名称, 问题, 答案

        为什么用 pandas 读CSV而不是手动解析? -> pandas自动处理编码/引号/特殊字符
        为什么用 %s 占位符而不是字符串拼接? -> 防止SQL注入攻击，且自动处理特殊字符
        """
        try:
            data = pd.read_csv(csv_path)  # 读取CSV为DataFrame表格
            # print(f'data: {data}')
            # print(data.iterrows())

            # iterrows(): 逐行遍历，返回 (行索引, 行数据Series)
            # 用 _ 接收索引(不需要)，row 是每行数据
            for _, row in data.iterrows():
                # %s 是 pymysql 的参数占位符，不是 Python 的格式化字符串
                insert_query = "insert into jpkb (subject_name, question, answer) values (%s, %s, %s)"
                # row['学科名称'] 取对应列的值
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            self.connection.commit()  # 所有行插入完成后统一提交(批量提交比每行提交快很多)
            self.logger.info("数据插入成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL数据插入失败: {e}")
            self.connection.rollback()  # 回滚: 出错时撤销本次未提交的所有插入，保证数据一致性
            raise

    def fetch_questions(self):
        """
        查询所有问题: 给 BM25 建索引用

        返回值: tuple列表，每个元素是 (问题文本,) 的元组
            例如: [("python是什么",), ("git怎么用",), ...]
            为什么是元组而不是纯字符串? -> cursor.fetchall() 默认返回每行是tuple

        调用场景: BM25Search._load_data() 中调用，获取所有问题建立索引
        """
        try:
            self.cursor.execute("select question from jpkb")  # 查询所有问题
            result = self.cursor.fetchall()  # 获取全部结果(一次加载到内存，数据量大时应该用fetchmany)
            print(f'result: {result}')
            self.logger.info("数据获取成功")
            return result
        except pymysql.MySQLError as e:
            self.logger.error(f"数据获取失败: {e}")
            return []  # 返回空列表而不是抛异常，让调用方能继续运行

    def fetch_answer(self, question):
        """
        根据问题查询答案: BM25匹配成功后，取对应的答案

        参数说明:
            question: str, 原始问题文本(精确匹配)
                注意: 这里是 WHERE question = %s 精确匹配，不是模糊搜索
                因为 BM25 已经找到了最相似的问题，直接用它去取答案

        返回值:
            str 或 None: 答案文本，找不到返回 None

        调用场景: BM25Search.search() 步骤7.2
        """
        try:
            # fetchone() 只取第一条结果(因为 question 应该唯一)
            self.cursor.execute("select answer from jpkb where question = %s", (question,))
            result = self.cursor.fetchone()  # 返回 tuple 如 ("答案文本",) 或 None
            self.logger.info("数据获取成功")
            # result[0] 取元组第一个元素; 如果没找到返回 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f"数据获取失败: {e}")
            return None

    def close(self):
        """
        关闭数据库连接: 释放资源
        为什么需要显式关闭? -> 数据库连接是有限资源，不关会导致连接泄漏、服务器拒绝新连接
        调用场景: 程序结束时调用
        """
        try:
            self.connection.close()
            self.logger.info("MySQL数据库连接已关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL数据库连接关闭失败: {e}")

if __name__ == '__main__':
    # 测试代码: 单独运行本文件时验证各功能
    mysql_client = MysqlClient()  # 注意: 原代码写的是 MysqlCliient 有拼写错误

    # 测试建表(只需执行一次)
    # mysql_client.create_table()

    # 测试导入数据(只需执行一次，把CSV数据写入MySQL)
    # mysql_client.insert_data('../data/JP学科知识问答.csv')

    # 测试查询所有问题
    # questions = mysql_client.fetch_questions()
    # print(f'questions: {questions}')

    # questions = mysql_client.fetch_questions()
    # print(questions[0:3])  # 只看前3条

    # 测试查询答案(精确匹配)
    # answer = mysql_client.fetch_answer(
    #     "两个人开发项目,我push到github上后\n别人直接pull可以看到我的代码吗\n为什么我要pull request 他才能看到我提交的代码")
    # print(answer)

    # 关闭数据库连接(用完必须关，释放资源)
    # mysql_client.close()
























































