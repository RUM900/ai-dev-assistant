"""
AI Dev Assistant — Streamlit Cloud 独立部署版
==============================================
直接运行: streamlit run app.py
部署后获得: https://xxx.streamlit.app 公开链接
"""
import json
import math
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 配置（本地 .env / Streamlit Cloud Secrets）
# ============================================================

# 优先读 Streamlit Cloud 的 secrets，fallback 到本地 .env
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        return os.getenv("OPENAI_API_KEY")

DASHSCOPE_API_KEY = get_api_key()
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-flash"
EMBEDDING_MODEL = "text-embedding-v2"

# ============================================================
# Agent 核心（从 agent_core.py 内嵌）
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中语义搜索。当用户问技术问题、概念、最佳实践时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。支持基本运算、sqrt、sin、cos、log 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]

# 向量存储（session_state 模式，支持上传）
def _create_embedding_fn():
    class DashScopeEF(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
    return DashScopeEF()


def init_vector_store():
    """初始化向量库——用 session_state 维持，支持动态添加文档"""
    if "vector_store" in st.session_state:
        return st.session_state.vector_store

    chroma_client = chromadb.Client(chromadb.config.Settings(anonymized_telemetry=False))
    try:
        chroma_client.delete_collection("kb_streamlit")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="kb_streamlit",
        embedding_function=_create_embedding_fn(),
        metadata={"hnsw:space": "cosine"}
    )

    # 预置种子知识
    seed_docs = [
        ("Python asyncio", "asyncio 是 Python 的异步编程库。核心概念：async def 定义协程，await 等待执行，asyncio.gather() 并发执行。与 Java 多线程不同，asyncio 是单线程协程模型。"),
        ("RAG（检索增强生成）", "RAG 是 Retrieval-Augmented Generation 的缩写。流程：文档分块→Embedding向量化→存入向量库→用户提问向量化→余弦相似度检索Top-K→拼接Prompt→LLM生成答案。优化方向：chunk_size调整、混合检索、Re-rank。"),
        ("LangChain vs 手写", "LangChain 统一了 LLM 调用接口，用 LCEL（| 管道）串联 Prompt→LLM→Parser。优点：快速原型、生态丰富。缺点：封装过度、调试困难、版本混乱。建议理解底层原理后选择性使用。"),
        ("AI Agent 概念", "Agent 是能自主感知、决策、行动的 AI 系统。ReAct 模式：思考(Thought)→行动(Action)→观察(Observation)循环。核心组件：Tool Use（工具调用）、Memory（记忆管理）、Planning（任务规划）。"),
    ]
    for i, (title, content) in enumerate(seed_docs):
        collection.add(
            ids=[f"seed_{i}"],
            documents=[f"# {title}\n{content}"],
            metadatas=[{"source": "内置知识库", "type": "seed"}]
        )

    # 追踪上传文件
    st.session_state.uploaded_files = []
    st.session_state.vector_store = collection
    return collection


def add_to_kb(filename: str, content: str):
    """向知识库添加文档（自动分块）"""
    collection = init_vector_store()

    # 简易分块：按 ## 标题切分
    sections = [s.strip() for s in content.split("\n## ") if s.strip()]
    if not sections:
        sections = [content.strip()]

    for section in sections:
        chunk_id = f"upload_{uuid.uuid4().hex[:8]}"
        collection.add(
            ids=[chunk_id],
            documents=[section],
            metadatas=[{"source": filename, "type": "upload"}]
        )

    if filename not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(filename)


def remove_from_kb(filename: str):
    """从知识库删除指定文件的所有文档"""
    collection = init_vector_store()
    results = collection.get(where={"source": filename})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    if filename in st.session_state.uploaded_files:
        st.session_state.uploaded_files.remove(filename)


def search_knowledge(query: str) -> str:
    collection = init_vector_store()
    if collection.count() == 0:
        return "知识库为空，请先上传文档。"
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    if not docs:
        return "知识库中未找到相关信息。"
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, results["metadatas"][0]), 1):
        parts.append(f"[{meta.get('source','?')}]\n{doc[:500]}")
    return "\n\n---\n\n".join(parts)


def calculate(expression: str) -> str:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed.update({"abs": abs, "round": round, "sqrt": math.sqrt,
                    "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except NameError as e:
        var = str(e).split("'")[1] if "'" in str(e) else str(e)
        return f"变量 '{var}' 未定义。此工具只能计算纯数学表达式。"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    now = datetime.now()
    wd = ['一','二','三','四','五','六','日']
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} 星期{wd[now.weekday()]}"


TOOL_FUNCTIONS = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


def run_agent_stream(messages: list[dict], max_steps: int = 5):
    """Agent 执行器 - 生成器，逐 token yield 给 Streamlit"""
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    called_tools = set()

    full_messages = [
        {"role": "system", "content": "你是 AI 开发助手。用中文回复。优先使用 search_knowledge 查找技术答案。需要计算时用 calculate。"}
    ] + messages

    for step in range(max_steps):
        stream = client.chat.completions.create(
            model=MODEL, max_tokens=1024,
            messages=full_messages, tools=TOOLS,
            tool_choice="auto", stream=True,
        )

        content_parts = []
        tool_calls_map = {}
        finish_reason = None

        for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "text", "content": delta.content}

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    e = tool_calls_map[idx]
                    if tc.id: e["id"] = tc.id
                    if tc.function:
                        if tc.function.name: e["function"]["name"] += tc.function.name
                        if tc.function.arguments: e["function"]["arguments"] += tc.function.arguments

        full_content = "".join(content_parts) if content_parts else None
        raw_tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else None

        if not raw_tool_calls:
            full_messages.append({"role": "assistant", "content": full_content or ""})
            yield {"type": "done"}
            return

        full_messages.append({"role": "assistant", "content": full_content, "tool_calls": raw_tool_calls})

        for tc in raw_tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            args_key = f"{name}:{json.dumps(args, sort_keys=True)}"

            yield {"type": "tool_call", "name": name, "args": args}

            if args_key in called_tools:
                result = f"已调用过 {name}。"
            else:
                called_tools.add(args_key)
                func = TOOL_FUNCTIONS.get(name)
                try:
                    result = func(**args) if args else func()
                except Exception as e:
                    result = f"执行失败：{e}"

            yield {"type": "tool_result", "name": name, "result": str(result)[:300]}
            full_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    yield {"type": "error", "message": "推理步数超限"}


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="AI Dev Assistant",
    page_icon="🤖",
    layout="centered",
)

# 样式
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .tool-hint {
        color: #888;
        font-size: 0.85em;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Dev Assistant")
st.caption("基于 Qwen-Flash · RAG 检索增强 · Agent 工具调用")

# 检查 API Key
if not DASHSCOPE_API_KEY:
    st.error("⚙️ 未配置 API Key。请在 Streamlit Cloud 的 Settings → Secrets 中设置 `OPENAI_API_KEY`，或本地创建 `.env` 文件。")
    st.stop()

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("📚 知识库管理")

    # 文件上传
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    uploaded_file = st.file_uploader(
        "上传文档（.md / .txt）",
        type=["md", "txt"],
        key="kb_uploader",
        help="支持 Markdown 和纯文本，自动分块并向量化"
    )
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner("处理中..."):
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                add_to_kb(uploaded_file.name, content)
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ 已添加: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"上传失败: {e}")

    # 已上传文档列表
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if st.session_state.uploaded_files:
        st.caption(f"已上传 {len(st.session_state.uploaded_files)} 个文档：")
        for fname in st.session_state.uploaded_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {fname}")
            with col2:
                if st.button("🗑️", key=f"del_{fname}", help="删除"):
                    remove_from_kb(fname)
                    st.rerun()

    st.divider()

    # 内置知识概览
    with st.expander("📖 内置知识库"):
        st.markdown("""
        - Python asyncio（异步编程）
        - RAG 检索增强生成（原理）
        - LangChain vs 手写（对比）
        - AI Agent 概念（ReAct）
        """)

    st.divider()

    # 统计
    try:
        collection = init_vector_store()
        st.caption(f"📊 共 {collection.count()} 条知识片段")
    except Exception:
        pass

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_hint"):
            st.markdown(f'<div class="tool-hint">{msg["tool_hint"]}</div>', unsafe_allow_html=True)

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 回复（流式）
    with st.chat_message("assistant"):
        placeholder = st.empty()

        full_content = ""
        tool_calls_made = []

        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            history.append({"role": "user", "content": prompt})

            for event in run_agent_stream(history):
                etype = event.get("type")

                if etype == "text":
                    full_content += event.get("content", "")
                    placeholder.markdown(full_content + "▌")

                elif etype == "tool_call":
                    name = event.get("name", "?")
                    tool_calls_made.append(name)
                    placeholder.markdown(full_content + f"\n\n⚙️ *正在执行 {name}...*")

                elif etype == "tool_result":
                    placeholder.markdown(full_content + "▌")

                elif etype == "done":
                    display = full_content
                    if tool_calls_made:
                        display += f"\n\n---\n*🤖 调用了：{', '.join(tool_calls_made)}*"
                    placeholder.markdown(display)

                elif etype == "error":
                    placeholder.error(event.get("message", "未知错误"))

        except Exception as e:
            placeholder.error(f"错误: {e}")
            full_content = "(发生错误)"

        # 保存到历史
        tool_hint = f"🤖 调用了：{', '.join(tool_calls_made)}" if tool_calls_made else None
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_content or "(无响应)",
            "tool_hint": tool_hint,
        })
