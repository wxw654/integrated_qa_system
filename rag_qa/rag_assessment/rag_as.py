# todo 1 导入所需要的库 和模块
import pandas as pd  # 导入pandas库, 用于数据处理和保存csv
from ragas import evaluate  # 导入ragas 的evaluate函数 用于执行rag评估
# 导入ragas评估指标 , 忠实度 ,答案相关性 ,上下文相关性 和上下文召回率
from ragas.metrics import (

    context_precision,  # 上下文相关性
    Faithfulness, AnswerRelevancy, ContextRecall  # 上下文召回率
)
from datasets import Dataset  # 导入dataset库的dataset类 用于构建ragas所需要的格式
import json
# from langchain_openai import ChatOpenAI ,OpenAIEmbeddings   科学上网 & chatgpt api key

from langchain_ollama import ChatOllama, OllamaEmbeddings  #

from base.config import Config

"""
需求: 使用ragas框架对rag系统进行量化评估
    思路:
        1:准备数据
            加载json文件
            转换格式为dataset
        2:环境配置
            初始化大模型用于指标推理
            配置嵌入模型计算语义相似度
        3:指标选择
            忠实度
            答案相关性
            上下文相关性
            上下文召回率
        4:执行评估
        5:处理结果
            保存csv文件

"""

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 1.1准备数据
with open("./rag_evaluate_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"加载数据条数:{len(data)}")

# 1.2 转换格式为dataset
eval_data = {
    # 提取每个数据question字段,组成问题列表
    "question": [item['question'] for item in data],
    # 提取每个数据answer字段,组成答案列表
    "answer": [item['answer'] for item in data],
    # 提取每个数据context字段,组成上下文列表
    "retrieved_contexts": [item['context'] for item in data],
    # 提取每个数据ground_truth字段,组成真实答案列表
    "ground_truth": [item['ground_truth'] for item in data],
}
dataset = Dataset.from_dict(eval_data)

# 2. 环境配置
config = Config()

# 2.1初始化大模型用于指标推理
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=config.DASHSCOPE_API_KEY,
    temperature=0.01
)

# 2.2
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=config.DASHSCOPE_API_KEY)

# 3.指标选择

result=evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        context_precision,
        ContextRecall()
    ],
    llm=llm,
    embeddings=embeddings
)

print(f"ragas评估结果:{result}")
result_df=pd.DataFrame([result])
result_df.to_csv("ragas_evaluation_result.csv",index=False)
