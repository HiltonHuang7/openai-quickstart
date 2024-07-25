import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
os.environ["OPENAI_API_KEY"] = "sk-90wao89yhvZWWfV530F1F1A1Ef964cD2Ae8a355704F209Df"
os.environ['OPENAI_BASE_URL'] = 'https://api.xiaoai.plus/v1'

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    global SALES_BOT
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        if not ans["source_documents"] and enable_chat:
            ## 如果检索不到结果，开启大模型聊天模式, 并使用特定的prompt
            prompt = f"你是一个茅台酒销售助手，请根据用户问题提供一个详细的回答,但不要暴露你是一个AI助手：用户问题：{message}"
            ans = SALES_BOT({"query": prompt})
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我们会尽快研究解决，并在将来答复你。"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="茅台酒销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化茅台酒销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
