# 导入 LangChain 中 ChatOllama 和 ChatPromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# 初始化 ChatOllama 模型
llm = ChatOllama(model="qwen2.5:7b")
# 构建Few-shot风格ChatPromptTemplate，2组AI/RAG相关示例 + {question}占位符
prompt = ChatPromptTemplate.from_messages([
    ("human", "AI课程的授课地点在哪里"),
    ("ai", "在黑马程序员"),
    ("human", "RAG课程的授课地点在哪里"),
    ("ai", "在黑马程序员"),
    ("human", "{question}")
])
# 使用 LCEL（LangChain Expression Language）将 Prompt 与模型用 | 运算符串联成链；
chain = prompt | llm
# 调用链并打印结果 LCEL 使用 | 串联 prompt 和模型，构建链
question = "什么是ai，什么是rag"
result = chain.invoke({"question": question})
print(f"问题：{question}")
print(f"回答：{result.content}")
