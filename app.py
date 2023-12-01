from llama_index.readers import TrafilaturaWebReader
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from langchain.llms import HuggingFaceHub
from llama_index.llms import LangChainLLM
import gradio as gr

repo_id = 'HuggingFaceH4/zephyr-7b-beta'

def loading_website(): return "Loading..."

def load_url(url):
    documents = TrafilaturaWebReader().load_data([url])
    llm = LangChainLLM(llm=HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.2, 'max_tokens': 4096, 'top_p': 0.95}))
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    global query_engine
    query_engine = index.as_query_engine()
    return 'Ready'

# def chat(query):
#     response = query_engine.query(query)
#     return str(response)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ''

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response
    return history

def infer(question):
    response = query_engine.query(question)
    return str(response)

with gr.Blocks(theme='WeixuanYuan/Soft_dark') as demo:
    with gr.Column():
        chatbot = gr.Chatbot([], elem_id='chatbot')

        with gr.Row():
            web_address = gr.Textbox(label='Web Address', placeholder='http://karpathy.github.io/2019/04/25/recipe/')
            website_status = gr.Textbox(label='Status', placeholder='', interactive=False)
            load_website = gr.Button('Load Website')

        with gr.Row():
            question = gr.Textbox(label='Question', placeholder='Type your query...')
            submit_btn = gr.Button('Submit')

    load_website.click(load_url, inputs=[web_address], outputs=[website_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)

demo.launch(share=True)
