# -*- coding: utf-8 -*-
"""
MedLit-QA 医疗文献问答 Demo（RAG 最小原型）
功能：上传医疗文献(txt/md/pdf) -> 自动切片+向量化入库 -> 提问时检索相关段落 -> 大模型基于检索内容作答并标注来源
作者：左倩（个人学习项目）
"""
import os
import pickle
import numpy as np
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv

# ---------------- 配置区 ----------------
load_dotenv()
API_KEY   = os.getenv("API_KEY")                       # 必填：你的API Key
BASE_URL  = os.getenv("BASE_URL", "https://api.siliconflow.cn/v1")  # OpenAI兼容接口
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")  # 向量模型
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")    # 问答模型
# 若使用 Azure OpenAI，改为：
# BASE_URL="https://你的资源名.openai.azure.com/openai/deployments/你的部署名/api-version=2024-02-01"
CHUNK_SIZE = 400      # 每段约400字
OVERLAP = 60          # 段间重叠，保证语义完整
TOP_K = 3             # 每次检索最相关的3段
STORE_PATH = "vector_store.pkl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------------- 文档读取与切片 ----------------
def read_file(path):
    """读取 txt / md / pdf，返回纯文本"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"不支持的文件格式: {ext}")


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """按固定长度切片（带重叠），真实项目中会用语义/递归切片"""
    text = text.replace("\n", " ")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 20]


# ---------------- 向量化 ----------------
def get_embeddings(texts):
    """调用Embedding API把文本转成向量"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_store(files):
    """上传文件后：读取->切片->向量化->本地保存"""
    all_chunks, all_sources = [], []
    for path in files:
        text = read_file(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(path)] * len(chunks))
    if not all_chunks:
        return "没有解析到有效文本内容。"
    vecs = get_embeddings(all_chunks)
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources,
                     "vecs": np.array(vecs)}, f)
    return f"入库完成：共 {len(all_chunks)} 个文本块，来自 {len(files)} 个文件。"


def load_store():
    if not os.path.exists(STORE_PATH):
        raise gr.Error("请先上传文献构建知识库。")
    with open(STORE_PATH, "rb") as f:
        return pickle.load(f)


# ---------------- 检索 + 生成 ----------------
def retrieve(query, k=TOP_K):
    """余弦相似度检索最相关的k个文本块"""
    store = load_store()
    q_vec = np.array(get_embeddings([query])[0])
    vecs = store["vecs"]
    sims = vecs @ q_vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9)
    top_idx = sims.argsort()[::-1][:k]
    return [(store["chunks"][i], store["sources"][i], float(sims[i])) for i in top_idx]


def build_prompt(question, contexts):
    ctx_text = "\n\n".join(
        f"[资料{i+1}《{src}》]\n{chunk}" for i, (chunk, src, _) in enumerate(contexts))
    return f"""你是医疗文献问答助手。请严格根据以下资料回答问题；
如果资料不足以回答，请明确说"根据已有资料无法回答"，不要编造。
回答末尾请用 [1][2] 的格式标注答案出自哪些资料。

{ctx_text}

问题：{question}"""


def answer(question, history):
    contexts = retrieve(question)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        temperature=0.2,
    )
    reply = resp.choices[0].message.content
    sources = "\n".join(f"[{i+1}] 《{src}》 相关度 {score:.2f}" for i, (_, src, score) in enumerate(contexts))
    return reply, f"参考来源：\n{sources}"


# ---------------- 界面 ----------------
with gr.Blocks(title="MedLit-QA 医疗文献问答") as demo:
    gr.Markdown("# MedLit-QA 医疗文献问答 Demo\n"
                "上传医疗文献构建知识库，然后基于文献内容提问（RAG 检索增强生成最小原型）")
    with gr.Row():
        upload = gr.File(file_count="multiple", label="上传文献（txt/md/pdf）")
        build_btn = gr.Button("构建知识库", variant="primary")
    build_out = gr.Textbox(label="构建状态")
    gr.Markdown("---")
    question = gr.Textbox(label="你的问题", placeholder="例如：二甲双胍的常见不良反应有哪些？")
    ask_btn = gr.Button("提问", variant="primary")
    answer_out = gr.Textbox(label="回答")
    source_out = gr.Textbox(label="检索到的参考来源")

    build_btn.click(build_store, inputs=upload, outputs=build_out)
    ask_btn.click(answer, inputs=question, outputs=[answer_out, source_out])

if __name__ == "__main__":
    demo.launch()
