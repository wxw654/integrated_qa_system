# =====================================================================
# query_classifier.py —— 用户问题「分类器」模块（BERT 微调）
#
# 它解决的问题：用户提的一个问题，到底该走哪条路？
#   —— "通用知识"（类别0）：闲聊/常识类，不需要查知识库，大模型自己就能答
#   —— "专业咨询"（类别1）：学科/专业问题，必须先去 RAG/知识库检索再回答
#   提前分类的好处：不相关的查询不浪费检索资源，专业问题保证走检索保准确。
#
# 为什么用 BERT 微调而不是让大模型直接判？
#   —— 分类是小模型最擅长的活：一个几秒级推理的判别模型，
#      比每次调大模型 API 又便宜、又快、又稳定。
#
# 本文件包含两条线：
#   1) 推理线：加载训练好的模型 -> 对问题分两类（load_model）
#   2) 训练线：读 JSONL 标注数据 -> 切训练/验证集 -> Trainer 微调 -> 保存模型
#      （train_model）
# =====================================================================

import json
import os, sys
import torch                                     # 深度学习框架：管理张量、GPU/CPU 设备
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
# BertTokenizer : 把中文句子切成 BERT 认识的"token"（字级）并转成数字 ID
# BertForSequenceClassification : 在 BERT 主干上面加一个分类头，专做"整句分类"
from transformers import Trainer, TrainingArguments
# Trainer : 负责实际执行训练任务（打包好的训练循环：前向、算损失、反向、更新、评估）
# TrainingArguments: 负责定义训练各个参数和配置（轮数、批大小、日志、保存策略等）
from sklearn.model_selection import train_test_split     # 一行代码把数据随机切成 训练集/验证集
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # 评估用：分类报告、混淆矩阵
from base.config import Config
from base.logger import logger

# 和 vector_store 同样的套路：算出 rag_qa 的绝对路径，
# 保证不管从哪个目录启动脚本，都能找到 rag_qa/models 下的模型文件
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../rag_qa/core
rag_qa_path = os.path.dirname(current_dir)                 # .../rag_qa


class QueryClassifier:
    """
    问题二分类器：
      对外主要提供  train_model()（训练出模型） 和  load_model()（加载模型准备预测）
    """

    def __init__(self, model_path="../models/bert_query_classifier"):
        # model_path：微调后模型保存/加载的位置
        # ⚠️ 注意这是相对路径，实际是否命中取决于运行时的工作目录；
        #    若找不到则 load_model 会走"从头初始化"分支
        self.model_path = model_path

        # bert-base-chinese：谷歌开源的中文 BERT 预训练底座（词表、模型结构都在这里）
        # 我们的微调就是在它之上做二次训练
        bert_path = os.path.join(rag_qa_path, "models", "bert-base-chinese")

        # tokenizer 只负责"文字 -> 数字"的翻译工作，不训练、可复用底座的
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # 模型先留空，由 load_model() 决定是"加载已训练好的"还是"初始化新的"
        self.model = None

        # 有 GPU 用 GPU，没有用 CPU（训练建议 GPU，CPU 会非常慢）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"模型运行设备：{self.device}")

        # 标签名 -> 数字 ID 的映射表（模型只认识数字，不认识"通用知识"这四个字）
        # 训练时用它把文字标签转数字，预测时反过来按编号找类别名
        self.label_map = {"通用咨询": 0, "专业咨询": 1}

        # 构造对象时自动尝试加载已有模型
        self.load_model()

    def load_model(self):
        """
        【加载模型】两条路：
          1) 训练好的模型文件已存在 -> 直接加载（拿"毕业"的模型去用）
          2) 还没训练过 -> 从 bert-base-chinese 底座新建一个"未分类训练"的模型（拿"毛坯"模型，等着训练）
        """
        if os.path.exists(self.model_path):
            # 分支1：存在微调产物（config.json + model.safetensors + label 映射等）
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)   # 把模型参数搬到 GPU/CPU 上
            logger.info(f"模型已加载：{self.model_path}")
        else:
            # 分支2：从预训练底座初始化
            # num_labels=2：在 BERT 顶部新建一个 2 类的分类头
            # ⚠️ 这里用的是相对路径 "../models/bert-base-chinese"，
            #    依赖启动目录恰好是 rag_qa，更稳妥可换成 os.path.join(rag_qa_path, ...)
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese",
                                                                        num_labels=len(self.label_map))
            self.model.to(self.device)
            logger.info(f"模型加载成功：初始化新bert模型成功!")

    def preprocess_data(self, texts, labels):
        """
        【数据预处理】把"人话"变成"模型吃的张量"。
        texts:  句子列表        labels: 对应的中文标签列表
        """
        encodings = self.tokenizer(texts,
                                   truncation=True,        # 超过 max_length 的句子直接截断（丢弃后面）
                                   padding="max_length",   # 不足 128 的句子用填充符补齐到 128
                                   max_length=128,         # 统一长度：模型要求一个 batch 里所有句子等长才能拼成矩阵
                                   return_tensors="pt")    # pt = PyTorch 张量（还有 np/tf/jax 可选）
        # tokenizer 输出 dict：input_ids(字ID序列)、attention_mask(哪些是真字哪些是 padding)、token_type_ids 等

        # 中文标签 -> 数字 ID（"通用知识"->0，"专业咨询"->1），交叉熵损失只认数字
        labels = [self.label_map[label] for label in labels]
        return encodings, labels

    def create_dataset(self, encodings, labels):
        """
        【构造数据集】把编码好的数据包装成 PyTorch Dataset，
        Trainer 要求训练/验证数据必须是 torch.utils.data.Dataset 格式
        （能按下标取样本、能报长度）。
        """
        class Dataset(torch.utils.data.Dataset):
            # 定义在函数内部的轻量类：只在训练时用，不污染外部命名空间

            def __init__(self, encodings, labels):
                super().__init__()                 # 先调用父类 Dataset 的初始化
                self.encodings = encodings         # 存 token 化后的张量字典
                self.labels = labels               # 存数字标签列表

            def __len__(self):
                # PyTorch 协议要求实现：数据集总样本数（决定一个 epoch 有多少批）
                return len(self.labels)

            def __getitem__(self, idx):
                # PyTorch 协议要求实现：按下标取第 idx 条样本
                # 把 dict 里每个键（input_ids 等）对应位置的那一行单独切成 1 维张量
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])  # 附上这条样本的标签
                return item
                # Trainer 之后会自动把这些单条样本攒成 batch 喂给模型

        return Dataset(encodings, labels)

    def train_model(self, data_file=""):
        """
        【训练主流程】读数据 -> 切分 -> 预处理 -> 配参数 -> Trainer 训练 -> 保存。
        data_file: JSONL 格式标注文件路径（每行一个 {"query": "...", "label": "..."}）
        """
        # ---------- 0) 数据文件校验 ----------
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在：{data_file}")
            # ⚠️ 语义小瑕疵：文件"不存在"按惯例应抛 FileNotFoundError，
            #    这里抛的是 FileExistsError（存在才该抛的名字），只是异常类型名字不贴切，
            #    效果一样：中断训练并报错
            raise FileExistsError(f"训练集数据{data_file}不存在")

        # ---------- 1) 读 JSONL 数据 ----------
        # JSONL = 每行是一个独立 JSON 对象的文件格式，逐行 json.loads 解析
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(value) for value in f.readlines()]

        # 拆成"问题列表"和"标签列表"两条平行线
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        # ---------- 2) 切训练集/验证集 ----------
        # test_size=0.2：八成训练、两成当"模拟考试"用来验收
        # random_state=42：固定随机种子，每次切分结果一致，实验可复现
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # ---------- 3) 编码预处理 ----------
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # ---------- 4) 包装成 Dataset ----------
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # ---------- 5) 训练超参数配置 ----------
        training_args = TrainingArguments(
            output_dir="./bert_results",        # 训练中间产物（checkpoint）存放目录
            num_train_epochs=3,                 # 全部训练数据完整过 3 遍
            per_device_train_batch_size=8,      # 每张卡一次喂 8 条样本
            per_device_eval_batch_size=8,       # 评估时同样 8 条一批
            warmup_steps=20,                    # 前 20 步学习率从很小线性升到设定值：
                                                #   刚开始模型是"新手"，大步长容易直接跑偏，先小步热身
            weight_decay=0.01,                  # L2 正则化：惩罚过大的权重参数，防过拟合
            logging_dir="./bert_logs",          # TensorBoard 日志目录
            logging_steps=10,                   # 每 10 步打印一次 loss，看训练是否在下降
            eval_strategy="epoch",              # 每跑完 1 个 epoch 用验证集考一次试
            save_strategy="epoch",              # 每考一次试就保存一份模型
            load_best_model_at_end=True,        # 训练结束后自动回滚到"考得最好"的那一版权重
            metric_for_best_model="eval_loss",  # 评判"最好"的标准：验证集损失最小
            save_total_limit=1,                 # 磁盘上只保留最新 1 个 checkpoint，防爆盘
            fp16=False                          # 不用半精度训练（CPU 上 fp16 反而不支持/无收益）
        )

        # ---------- 6) 组装 Trainer 开练 ----------
        trainer = Trainer(
            model=self.model,               # 要训练的模型
            args=training_args,             # 上面那堆配置
            train_dataset=train_dataset,    # 训练集（平时学习用）
            eval_dataset=val_dataset,       # 验证集（定期考试用）
            # compute_metrics=self.compute_metrics  # 自定义评估指标（目前是空实现，先注释掉，
                                                    # Trainer 默认只算 loss）
        )

        logger.info(f"开始训练bert模型")
        trainer.train()      # 同步阻塞执行整个训练循环（几小时级别的操作）
        self.save_model()    # 训练完把最终模型落盘，供以后 load_model 直接加载

        # 想细看指标时打开：输出 classification_report / confusion_matrix
        # self.evaluate_model(val_texts, val_labels)
        logger.info(f"模型训练完成！")

    def save_model(self):
        """
        【保存产物】模型权重 + tokenizer 配置一起存到 model_path 目录。
        tokenizer 虽然没参数要训练，但它的词表/特殊符号配置也要跟模型放一起，
        下次加载时才能"文字->数字"的翻译规则完全一致。
        """
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型已保存：{self.model_path}")

    def predict_category(self, query):
        if self.model is None:
            logger.error("模型未训练或者未加载")
            return "通用咨询"
        encoding=self.tokenizer(query, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction=torch.argmax(outputs.logits,dim=1).item()
        return "专业咨询 " if prediction == 1 else "通用咨询"



    def evaluate_model(self, texts, labels):

        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        dataset = self.create_dataset(encodings, labels)
        trainer = Trainer(
            model=self.model,
            eval_dataset=dataset,
        )
        predictions=trainer.predict(dataset)
        pred_labels=np.argmax(predictions.predictions, axis=1)
        true_labels = labels
        logger.info("分类报告")
        logger.info(classification_report(true_labels, pred_labels))














# ==================== 本地测试入口 ====================
if __name__ == '__main__':
    query_classifier = QueryClassifier()
    # 训练语料：5000 条 "问题 -> 类别" 的 JSONL 标注数据
    # data_file = r"C:\Users\Administrator\PycharmProjects\integrated_qa_system\rag_qa\model_generic_5000.json"
    # # 训练模型（跑完后模型会保存到 model_path，之后系统就能用这个分类器给问题分路由了）
    # query_classifier.train_model(data_file)

    test_queries = [
        "今天天气怎么样？",
        "AI学科主要学习哪些技术？",
        "大数据课程的学习强度大不大",
        "中午吃什么好"
    ]

    # 4.循环预测
    for q in test_queries:
        res = query_classifier.predict_category(q)
        print(f"输入：{q}  => 类别：{res}")