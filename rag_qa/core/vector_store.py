# =====================================================================
# vector_store.py —— RAG 系统的「向量存储 + 混合检索 + 重排序」核心模块
#
# 这个文件干的事，用大白话讲就是三步：
#   1. 把切好的文档块（子块）用 BGE-M3 模型转成"向量"，存进 Milvus 向量数据库
#      —— 向量你可以理解成：把一段文字变成一串数字，文字意思越接近，数字越相似
#   2. 用户提问时，把问题也变成向量，在 Milvus 里同时做两种搜索：
#      —— 稠密向量搜索：语义搜索（"意思像不像"，比如"土豆"和"马铃薯"能匹配上）
#      —— 稀疏向量搜索：关键词搜索（"字面对不对"，类似传统搜索/BM25 的思路）
#      两者合起来叫「混合检索 hybrid search」
#   3. 搜出来的结果再用 BGE-Reranker 模型精排一次（重新打分排序），
#      把真正最相关的文档排在最前面，喂给大模型去生成答案
#
# 检索还用了「父子块策略」：
#   —— 用小块（子块）去搜索：小块语义集中，搜得准
#   —— 返回给大模型的是大块（父块）：大块上下文完整，答得全
# =====================================================================

from milvus_model.hybrid import BGEM3EmbeddingFunction  # BGE-M3 嵌入模型（Milvus 官方封装，能把文本同时转成稠密+稀疏两种向量）
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# MilvusClient      : 连接和操作 Milvus 数据库的客户端
# DataType          : 定义字段类型（字符串、浮点向量、稀疏向量等）
# AnnSearchRequest  : 一次"向量近似最近邻搜索"的请求（ANN = 不逐个比对、快速找相似）
# WeightedRanker    : 多路搜索结果"加权融合"的排序器
from langchain.docstore.document import Document        # LangChain 的标准文档对象：page_content(正文) + metadata(附加信息)
import hashlib                                          # 哈希库：用来给文本算 MD5 指纹，当唯一 ID 用
import os, sys, torch                                   # torch：深度学习框架，这里主要用它判断有没有 GPU
from sentence_transformers import CrossEncoder          # 交叉编码器：BGE-Reranker 重排序模型用的类
from base.config import Config                          # 项目统一配置（Milvus 地址、端口、集合名等）
from base.logger import logger                          # 项目统一日志
from rag_qa.core.document_processor import process_documents  # 文档加载+切分模块（把文件变成一个个 Document 块）

# 计算目录路径，目的是：不管从哪个工作目录启动脚本，都能正确找到 rag_qa/models 下的模型文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录：.../rag_qa/core
rag_qa_path = os.path.dirname(current_dir)                 # 往上跳一级：.../rag_qa

# 实例化配置对象，后面读 Milvus 连接参数、检索条数 K 都从这里拿
conf = Config()


class VectorStore:
    """
    向量库操作类：负责「建库、存数据、混合检索、重排序」四件事。
    整个 RAG 系统里"检索"这一半的活儿都在这个类里。
    """

    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,  # 集合名（Milvus 里的"表"）
                 host=conf.MILVUS_HOST,                        # Milvus 服务器 IP
                 port=conf.MILVUS_PORT,                        # Milvus 端口，默认 19530
                 database=conf.MILVUS_DATABASE_NAME):          # 数据库名（Milvus 2.x 支持多库隔离）
        # ---- 1) 保存连接配置 ----
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger

        # ---- 2) 选择计算设备 ----
        # 有 NVIDIA 显卡就用 GPU（cuda），否则用 CPU
        # 深度模型（BGE-M3 / Reranker）在 GPU 上快几十倍，但没卡也能跑，只是慢
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- 3) 加载重排序模型 BGE-Reranker-Large ----
        # CrossEncoder 是"交叉编码器"：把 [问题, 文档] 拼在一起送进模型，
        # 直接输出一个相关性分数。比向量相似度算得细，所以叫"精排"
        reranker_path = os.path.join(rag_qa_path, "models", "bge-reranker-large")
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # ---- 4) 加载嵌入模型 BGE-M3 ----
        # M3 = Multi-Functionality（一个模型同时输出 3 种向量：稠密/稀疏/多向量）
        # 我们用其中的：稠密向量（管语义）+ 稀疏向量（管关键词）
        bge_path = os.path.join(rag_qa_path, "models", "bge-m3")
        # use_fp16：GPU 时启用半精度（16 位小数）计算 —— 省显存、提速，精度损失可忽略；CPU 上不支持所以关掉
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=bge_path,
                                                         use_fp16=(self.device == "cuda"),
                                                         device=self.device)
        # 稠密向量的维度（BGE-M3 是 1024 维），建表时向量字段必须声明维度
        self.dense_dim = self.embedding_function.dim["dense"]

        # ---- 5) 连接 Milvus 并保证集合可用 ----
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        self._create_or_load_collection()  # 集合不存在就按 schema 创建，存在就直接用

    def _create_or_load_collection(self):
        """
        【建表逻辑】Milvus 存数据前必须先有"集合"（相当于关系库的表），
        表结构（schema）+ 索引（index）在这里一次性定义好。
        私有方法（下划线开头）：只给类内部用。
        """
        if not self.client.has_collection(self.collection_name):
            # has_collection: 检查集合是否已存在，不存在才创建，避免报错/重复建表

            # auto_id=False：主键不自动生成，我们用文本的 MD5 当 ID（好处：同一段文本重复入库只会覆盖，不会产生重复数据）
            # enable_dynamic_field=True：允许存 schema 里没声明的额外字段，方便以后扩展
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)

            # ---------- 定义表的字段（列）----------
            # 主键 ID：文本内容的 MD5 哈希值，VARCHAR=变长字符串，最长 100 字符
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 子块原文：检索命中后展示/比较用的正文，最长 65535（约 64K 字符，足够装长文本）
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 稠密向量：BGE-M3 输出的 1024 维浮点数组，负责"语义搜索"
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 稀疏向量：只有少数位置有值的超大维度向量（本质是"词 -> 权重"的字典），负责"关键词搜索"
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 父块 ID / 父块全文：子块检索命中后，靠这两个字段找回大块上下文（父子块策略的核心）
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 来源标记（比如 "ai" 学科）：检索时可以按 source 过滤，只搜某个学科的数据
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 时间戳：预留的元数据字段
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # ---------- 给向量字段建索引 ----------
            # 索引 = 数据的"目录"。没有索引，向量搜索要把全表逐个比对，量大就慢死
            index_params = self.client.prepare_index_params()

            # 稠密向量索引：IVF_FLAT
            #   IVF = 先把向量聚成 nlist(128) 个簇，搜索时只进最像的 nprobe 个簇里找 —— 用"可能漏掉边缘结果"换"速度"
            #   FLAT = 簇内部不压缩、精确比对，保证簇内结果准
            #   metric_type="IP" = 用内积(Inner Product)衡量相似度，向量归一化后内积≈余弦相似度
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}          # 聚类中心个数：库越小可调小，库越大可调大
            )

            # 稀疏向量索引：SPARSE_INVERTED_INDEX（稀疏倒排索引）
            #   类似搜索引擎的"倒排表"：记录"哪个词出现在哪些文档里、权重多少"
            #   drop_ratio_build=0.2：建索引时丢掉权重最低的 20% 的词 —— 省内存、提速，对精度影响很小
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 建集合：schema（表结构）和 index_params（索引）一次传入，创建完即可用
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            # 集合已存在：直接复用，不重建（防止把已有数据清掉）
            logger.info(f"已加载集合 {self.collection_name}")
        # load_collection：把数据从磁盘加载进内存，Milvus 规定只有加载后的集合才能被搜索
        self.client.load_collection(self.collection_name)

    def add_documents(self, documents):
        """
        【入库】把切分好的 Document 子块列表转成向量并写入 Milvus。
        documents: List[Document]，每个 doc 的 metadata 里带着 parent_id / parent_content 等字段
        """
        # 1) 抽出所有子块的纯文本，批量做嵌入（一次传列表进去，比一条条传快得多）
        texts = [doc.page_content for doc in documents]
        # 2) BGE-M3 编码：返回 dict：{"dense": N x 1024 的稠密向量矩阵, "sparse": 稀疏矩阵}
        embeddings = self.embedding_function(texts)
        print(f"embdedding:{embeddings}")

        # 3) 逐条组装 Milvus 要写入的记录
        data = []
        for i, doc in enumerate(documents):
            # 用文本内容的 MD5 当主键：同一段文字永远得到同一个哈希
            #   => 重复入库时 upsert 会按 ID 覆盖旧记录，天然去重
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            print(f"text_hash:page_content使用hash：{text_hash}")

            # 稀疏矩阵不能整行塞进 dict，要转成 Milvus 要求的 {维度下标: 权重} 格式
            # embeddings['sparse'][[i]]：双层列表索引是为了"保留二维形状"取出第 i 行（scipy 稀疏矩阵语法）
            sparse_vector = {}
            row = embeddings['sparse'][[i]]
            print(f"row:{row},{row.shape}")
            indices = row.indices   # 非零元素的位置（可以理解为"第几个词"）
            values = row.data       # 对应位置的权重值（可以理解为"这个词的重要程度"）
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value  # 拼成 {下标: 权重} 字典

            # 组装一行完整记录：正文 + 两种向量 + 父子块信息 + 元数据
            # get("source", "unknown")：metadata 里没这个键就用默认值，防止 KeyError 崩掉
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        # 4) upsert = update + insert：ID 存在就更新，不存在就插入（幂等写法，重跑脚本不会产生脏数据）
        if data:
            self.client.upsert(self.collection_name, data)
            logger.info(f"已插入:{len(data)}条数据到集合{self.collection_name}")

    def hybrid_search_with_rerank(self, query: str, k=conf.RETRIEVAL_K, source_filter=None):
        """
        【核心检索方法】混合检索 + 重排序，一条问题进，最相关的父文档列表出。

        流程：问题向量化 -> 稠密/稀疏双路各召回 k 条 -> 加权融合 ->
              子块换父块并去重 -> Reranker 精排 -> 返回

        k: 最终想拿回多少条候选
        source_filter: 按学科/来源过滤，比如传 "ai" 就只在 AI 学科数据里搜
        """
        # ---------- 第 1 步：把问题也转成和文档同一套空间的两种向量 ----------
        query_embedding = self.embedding_function([query])
        dense_query_vector = query_embedding['dense'][0]  # 取第 0 条（因为只输入了 1 个问题）

        # 稀疏向量和入库时一样，把稀疏矩阵第 0 行转成 {下标: 权重} 字典
        sparse_query_vector = {}
        row = query_embedding["sparse"][[0]]
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # ---------- 第 2 步：拼过滤表达式 ----------
        # Milvus 的标量过滤写法，类似 SQL 的 WHERE；不传 source 就是空串（不过滤，全库搜）
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # ---------- 第 3 步：两路召回请求 ----------
        # 稠密路：语义搜索。nprobe=10 表示查询时进 10 个聚类簇里找（越大越准也越慢）
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,           # 这一路各取前 k 名
            expr=filter_expr
        )
        # 稀疏路：关键词搜索。param 里不需要 nprobe（倒排索引不走聚类）
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # ---------- 第 4 步：融合 + 搜索 ----------
        # WeightedRanker(1.0, 0.7)：按请求顺序加权合并两路分数
        #   稠密向量得分权重 1.0，稀疏向量得分权重 0.7
        #   => 语义为主、关键词为辅；两边都能加分的文档排最前
        ranker = WeightedRanker(1.0, 0.7)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,               # 融合后总共返回前 k 条
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
            # output_fields 必须显式声明，否则结果里只带主键，正文字段拿不到
        )[0]
        # hybrid_search 返回 [[hit1, hit2, ...]]（每条查询一个列表），我们只查 1 个问题，所以取 [0]

        # ---------- 第 5 步：命中结果转成 Document 对象（此时还是"子块"）----------
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        print(f'sub_chunks-->{len(sub_chunks)}')

        # ---------- 第 6 步：子块换成父块，并按内容去重 ----------
        # 多个子块很可能来自同一个父块，不去重的话同一个大段内容会重复送给 Reranker/大模型
        parent_docs = self._get_unique_parent_docs(sub_chunks)

        # ---------- 第 7 步：精排（Rerank）----------
        # 只有 0 或 1 个父文档时没排序意义，直接返回（也省一次模型计算）
        if len(parent_docs) < 2:
            return parent_docs
        # ⚠️ 注意：下面这段 is/预测排序代码在 len>=2 时才会走到（上一个分支已 return）
        if parent_docs:
            # Reranker 的输入格式：[[问题, 文档1], [问题, 文档2], ...]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # predict 给每个 [问题,文档] 对打一个相关性分（交叉编码器逐对精读，所以准）
            scores = self.reranker.predict(pairs)
            # zip 把"分数-文档"绑成对，按分数从高到低重排，再解开取出文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            ranked_parent_docs = []

        return ranked_parent_docs

    def _doc_from_hit(self, hit):
        """
        【小工具】把 Milvus 返回的一条 entity（dict）包装成 LangChain 的 Document 对象，
        这样后续流程（去重、喂给大模型）都能用统一的 .page_content / .metadata 接口。
        """
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

    def _get_unique_parent_docs(self, sub_chunks):
        """
        【父块回收 + 去重】把命中的子块列表换成"互不重复的父文档列表"。
        这就是父子块检索（Small-to-Big）的关键一步：搜的是小块，用的是大块。
        """
        # 1. 初始化集合，用于存储已处理的父块内容（set 查找是 O(1)，去重快）
        parent_contents = set()
        # 2. 初始化列表，用于存储唯一的父文档
        unique_docs = []
        # 3. 遍历子块
        for chunk in sub_chunks:
            # 3.1 取子块的父块内容；万一这个子块没有 parent_content（比如没做父子切分），
            #     就退回用子块自己兜底，保证流程不中断
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 3.2 父块内容非空、且之前没见过 => 才算新文档
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs


# ==================== 本地测试入口 ====================
# 直接运行本文件时走这里；被别的模块 import 时不会执行（__name__ != '__main__'）
if __name__ == '__main__':
    vector_store = VectorStore()
    # 下面两行是"入库测试"：加载并切分 data/ai_data 目录的文档，再写入 Milvus（已注释，按需打开）
    # directory_path = "../data/ai_data"
    # documents = process_documents(directory_path)
    # vector_store.add_documents(documents)

    # "检索测试"：模拟用户提问，只在 source="ai" 的数据里做混合检索+重排
    query = "AI学科的课程内容是什么？"
    result = vector_store.hybrid_search_with_rerank(query=query, source_filter="ai")

    print(result)
    print(len(result))
