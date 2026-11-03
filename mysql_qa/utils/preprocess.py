"""
【步骤6: 文本预处理 - 分词】
作用: 把中文句子切分成词语列表，供 BM25 建立索引和查询匹配使用

为什么需要分词?
    BM25 是"词袋模型"，它比较的是"词"的重合度
    中文没有空格分隔，必须先切成词才能计算相似度
    例如: "python是什么" -> ["python", "是", "什么"]

jieba分词简介:
    - 最常用的中文分词库
    - 支持精确模式/全模式/搜索引擎模式
    - lcut() 返回列表，cut() 返回生成器(本项目用lcut更简单)

调用场景:
    1. BM25Search._load_data(): 对所有问题分词，建立索引
    2. BM25Search.search():     对用户查询分词，然后计算相似度
"""
import jieba                  # 中文分词库
from base.logger import logger  # 日志记录


def preprocess_text(text):
    """
    文本预处理函数: 转小写 + 分词

    参数说明:
        text: str, 输入的原始文本
              例如: "Python是什么？怎么用？"

    返回值:
        list[str]: 分词后的词语列表
              例如: ['python', '是', '什么', '？', '怎么', '用', '？']

    为什么先 lower() 再分词?
        - 统一大小写，防止 "Python" 和 "python" 被当成不同的词
        - BM25是精确匹配词频的，大小写不一致会导致匹配失败

    为什么用 jieba.lcut() 而不是 jieba.cut()?
        - lcut() 直接返回 list，方便后续传给 BM25Okapi
        - cut() 返回 generator，还需要手动 list() 转换
    """
    logger.info("开始预处理文本")
    try:
        # text.lower() 先转小写，然后 jieba.lcut() 切词
        return jieba.lcut(text.lower())
    except Exception as e:
        # 分词失败时返回空列表，而不是抛异常(防止整个搜索流程中断)
        logger.error(f"文本预处理失败: {e}")
        return []



if __name__ == '__main__':
    # 测试: 验证分词效果
    # 预期输出: ['这是', '一个', '测试', 'test']
    print(preprocess_text("这是一个测试TEST"))





































