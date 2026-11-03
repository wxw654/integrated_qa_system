import os, sys
from langchain_community.document_loaders import TextLoader  # 加载纯文本(.txt)工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader  # 加载maskdown文件
from langchain.text_splitter import MarkdownTextSplitter  # 针对MD文档进行切分
from datetime import datetime

# 自定义的文档加载器和文档分割器 , 直接复制过来,即可 ,无需手动编写.后续遇到类似操作,直接调用代码即可
# 从06资料中找到 三个文件夹edu_text_spliter ， edu_document_loaders ,samples  放置在 rag_qa目录下.即完成导包
from rag_qa.edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from pathlib import Path

from base.config import Config
from base.logger import logger

# todo 2 初始化配置实例 和加载器模块
# 1.初始化配置实例
conf = Config()

# 2.加载器模块映射
# 字典  key:文件拓展名(小写形式,确保匹配)     value : 对应的文档加载器
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".png": OCRIMGLoader,
    ".jpg": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader
}


# todo 3.核心加载函数模块 -> 从指定的目录递归加载所有的支持类型文件,添加元数据并返回文档列表
# 原始格式      metadata={'source': WindowsPath('../data/ai_data/LLM基础知识.pdf')
# 处理完格式  : metadata={'source': 'ai', 'file_path': WindowsPath('../data/ai_data/LLM基础知识.pdf'), 'timestamp': '2026-10-15T16:52:51.744347'}
def load_documents_from_directory(directory_path):
    """
    从指定目录(含子目录)加载所有支持的文件类型,为每个文档添加元数据
    :param directory_path: 目标文件夹的路径
    :return: 加载完成文档列表(每个元素为langchain Document对象 包含:page_content和metadata)
    """
    # 1.初始化空列表,用于存储所有加载成功的文档
    documents = []

    # 2.获取所有支持的文件拓展名集合
    supported_extensions = document_loaders.keys()
    # print(f"支持的文件类型：{supported_extensions}")
    # 支持的文件类型：dict_keys(['.txt', '.pdf', '.docx', '.ppt', '.pptx', '.png', '.jpg', '.md'])

    # 3.从目录中提取"学科类别"元数据(如:目录名: 学科名_data)->提取学科名作为source
    source = os.path.basename(directory_path).replace("_data", "")
    # print(f"source:{source}")

    # 4.递归遍历目标目录及其所有子目录
    for root, _, files in os.walk(directory_path):
        # print(f"root:{root}")  #正在处理目录:"../data/ai_data"
        # # print(f"a:{a}")
        # print(f"files:{files}")  #正在处理的文件 files:['LLM基础知识.pdf', '人工智能就业课课程大纲.md']
        # 5.遍历当前目录下的每一个文件,逐一处理
        for file in files:
            root = Path(root)
            # 5.1 构建文件的完整路径
            file_path = root / file
            # file_path=os.path.join(root,file)  #../data/ai_data/LLM基础知识.pdf
            print(file_path)
            # 5.2提前文件拓展名,并转换为小写
            file_extension = os.path.splitext(file)[1].lower()  # ('LLM基础知识', '.pdf')  .pdf
            print(f"file_extension:{file_extension}")
            # 5.3 若文件类型在支持的列表里,执行加载逻辑
            if file_extension in supported_extensions:
                try:
                    # 5.3.1 根据文件拓展名获取对应的文档加载器
                    loader_class = document_loaders[file_extension]
                    # 5.3.2 实例化加载器 :细节: txt文件需要单独指定码表(避免中文乱码)
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 5.3.3调用load方法,加载文件内容
                    loaded_docs = loader.load()
                    # print(f"处理前: loaded_docs:{loaded_docs}")
                    # 5.3.4 为每个文档添加元数据(便于后期rag检索是过滤/溯源)
                    for doc in loaded_docs:
                        doc.metadata["source"] = source  # 学科类别: AI
                        doc.metadata["file_path"] = file_path  # 文件路径 ../data/ai_data/LLM基础知识.pdf
                        doc.metadata['timestamp'] = datetime.now().isoformat()  # 加载时间戳(精确到秒,方便版本管理)
                    # 5.3.5 将添加元数据的文档列表添加到总列表中
                    documents.extend(loaded_docs)
                    # 5.3.6记录info日志,确认文件加载成功
                    logger.info(f"文件:{file_path}加载成功.")
                except Exception as e:
                    # 5.3.7 捕获文件加载过程中出现的所有异常(文件损坏,权值补助,ocr失败失败等...)
                    logger.error(f"文件:{file_path}加载失败,原因:{str(e)}")
            else:
                logger.warning(f"不支持的文件类型:{file_path}")
    # 6.返回所有的加载成功的文件列表,供后续文本分割,向量存储等rag步骤使用
    return documents


# todo 4 文档处理核心函数:加载文档后进行分层切分(父块+子块) 返回带元数据的子块列表
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE, child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并进行分层切块,先切大粒度父块,再将父块切分小粒度子块,子块关联父块元数据
    :param directory_path: 文档目录路径
    :param parent_chunk_size: 父块切分长度(大粒度)
    :param child_chunk_size: 子块切分长度(小粒度)
    :param chunk_overlap: 子块切分重叠长度(确保上下文关联)
    :return:带元数据的子块列表(每个子块含父块关联信息)
    """
    # 1.调用加载函数 ,从目录加载所有文档
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档的数量:{len(documents)}")
    # 2.初始化切分器 : 区分通用分档和Markdown文档
    # 2.1通用文档
    # 2.1.1 初始化父块和子块分词器(通用),适用于text/pdf/word等,基于中文语义切分
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 2.2专用文档
    # 2.2.1 初始化 MD专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 3.初始化子块列表,存储最终的切分结果
    child_chunks = []

    # 4.遍历每个原始的文档(带索引i,用于生成唯一ID)
    for i, doc in enumerate(documents):
        # 4.1获取文档的拓展名(判断是否为MD文档,选择对应切分器)
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower
        is_markdown = (file_extension == ".md")

        # 4.2根据文档类型选择切分器
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        logger.info(f"处理文档:{'MarkDown' if is_markdown else 'ChineseRecursive'}")

        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(
            f"处理文档:{doc.metadata['file_path']},使用切分器:{'markdown' if is_markdown else 'ChineseRecursive'}")

        # 4.3 第一步:将文档切分为大块
        parent_docs = parent_splitter_to_use.split_documents([doc])


        # 5.遍历每个父块(带索引j,用于生成唯一ID)
        for j, parent_doc in enumerate(parent_docs):
            # 5.1生成父块唯一ID(格式: doc_文档索引_parent_父块索引)
            parent_id = f"doc_{i}_parent_{j}"
            # 5.2为父块添加元数据(rag后续溯源和过滤)
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['parent_content'] = parent_doc.page_content
            # 5.3 第二步:将父块切分成小块->小粒度,便于精准匹配
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 6. 遍历每一个子块(带索引k,生成唯一ID)
            for k, sub_chunk in enumerate(sub_chunks):
                # 6.1为子块添加关联父块的信息
                sub_chunk.metadata['parent_id'] = parent_id
                sub_chunk.metadata['parent_content'] = parent_doc.page_content
                # 6.2生成子块唯一ID(格式:父块id_child_子块索引)
                sub_chunk.metadata['id'] = f"{parent_id}_child_{k}"
                # 6.3 将子块添加到子块列表中
                child_chunks.append(sub_chunk)
        # 7.记录切分后子块的总数
        logger.info(f"子块数量:{len(child_chunks)}")
    # 8.返回子列表(供后续向量存储和检索使用)
    return child_chunks


# todo 5 主函数测试模块
if __name__ == '__main__':
    # # 1.加载文档
    directory_path = "../data/ai_data"
    # # 2.调用核心函数,获取文档列表
    documents = load_documents_from_directory(directory_path)
    # # 3.打印加载结果
    # print(f"加载完成文档数量:{len(documents)}")
    # print(f"前1个文档详情:")
    # print(documents[0] if documents else "未加载任何文档.")

    # chunks = process_documents(
    #     "../data/ai_data",  # 目标目录文本
    #     conf.PARENT_CHUNK_SIZE,  # 父块长度
    #     conf.CHILD_CHUNK_SIZE,  # 子块长度
    #     conf.CHUNK_OVERLAP  # 子父块重叠长度
    # )
    #
    # print(f"分完子块的总数量:{len(chunks)}")
    # print(f"前一个块的详情:")
    # print(chunks[0] if chunks else "未生成任何子块")
