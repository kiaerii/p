import os
import re
import pickle
import base64
import time
from typing import TypedDict, List, Literal

import streamlit as st
import pandas as pd
from pyaspeller import YandexSpeller
from rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- НАСТРОЙКИ И КЛЮЧИ ---
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-6d5c4faa-0e9b-4c41-b8be-8b41b76c3314"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-5ac98f2d-724a-48d3-966c-564ce41a6022"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["TAVILY_API_KEY"] = "tvly-..."  # Вставьте ваш ключ Tavily, если нужен веб-поиск

try:
    from langfuse.langchain import CallbackHandler
    from langfuse import Langfuse

    langfuse_handler = CallbackHandler()
    langfuse_client = Langfuse()
    LANGFUSE_ENABLED = True
except Exception as e:
    print(f"Ошибка инициализации Langfuse: {e}")
    LANGFUSE_ENABLED = False
    langfuse_handler = None
    langfuse_client = None

FAISS_DIR = "faiss_index"
CHUNKS_PATH = "chunks.pkl"
BM25_PATH = "bm25.pkl"
RRF_K = 60

_STEMMER = SnowballStemmer("russian")
STOPWORDS = {"почему", "какие", "какой", "какая", "что", "как", "это", "для", "или", "про",
             "есть", "при", "над", "под", "если", "чем", "где", "когда", "кто"}


def tokenize(text: str) -> list:
    tokens = re.findall(r"\w+", text.lower())
    return [_STEMMER.stem(w) for w in tokens if len(w) >= 3 and w not in STOPWORDS]


@st.cache_resource
def load_indexes():
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2",
                                       encode_kwargs={"normalize_embeddings": True})
    faiss_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    with open(CHUNKS_PATH, "rb") as f: chunks = pickle.load(f)
    with open(BM25_PATH, "rb") as f: bm25 = pickle.load(f)
    return faiss_db, bm25, chunks


class GraphState(TypedDict):
    original_question: str
    fixed_question: str
    sub_questions: List[str]
    query_type: str
    documents: List[Document]
    generation: str
    retry_count: int
    steps_log: List[str]
    used_chunks: List[dict]
    cost: float


def get_cost(model_name, input_tokens, output_tokens):
    if "120b" in model_name:
        return (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
    else:
        return (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)


# --- УЗЛЫ ГРАФА ---
def fix_spelling(state: GraphState):
    try:
        speller = YandexSpeller()
        fixed = speller.spelled(state["original_question"])
    except Exception:
        fixed = state["original_question"]
    log = f"🔤 Спелчекер: '{state['original_question']}' ➔ '{fixed}'" if fixed != state[
        "original_question"] else "🔤 Опечаток нет"
    return {"fixed_question": fixed, "steps_log": state.get("steps_log", []) + [log]}


def analyze_query(state: GraphState):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Определи, к какому источнику относится вопрос. Варианты:\n"
        "- 'habr': IT-статьи, программирование.\n"
        "- 'news': новости, политика.\n"
        "- 'science': научные исследования.\n"
        "- 'general': приветствие (например, 'как дела').\n"
        "Вопрос: {q}\nОтвет (ровно одно слово из списка):"
    )
    res = (prompt | llm | StrOutputParser()).invoke({"q": state["fixed_question"]}).strip().lower()

    q_type = "general"
    for t in ["habr", "science", "news"]:
        if t in res:
            q_type = t
            break

    cost = get_cost("openai/gpt-oss-20b", 150, 5)
    return {"query_type": q_type, "steps_log": state["steps_log"] + [f"🔍 Определен источник: {q_type}"],
            "cost": state.get("cost", 0.0) + cost}


def decompose_query(state: GraphState):
    if state["query_type"] == "general":
        return {"sub_questions": [state["fixed_question"]]}

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Разбей этот вопрос на 1-3 простых подвопроса. Верни только подвопросы через запятую.\n"
        "Вопрос: {q}"
    )
    res = (prompt | llm | StrOutputParser()).invoke({"q": state["fixed_question"]})
    subs = [s.strip() for s in res.split(",") if s.strip()]
    if not subs: subs = [state["fixed_question"]]

    cost = get_cost("openai/gpt-oss-20b", 100, 30)
    return {"sub_questions": subs, "steps_log": state["steps_log"] + [f"🧩 Декомпозиция: {len(subs)} подвопрос(а)"],
            "cost": state.get("cost", 0.0) + cost}


def retrieve_docs(state: GraphState):
    faiss_db, bm25, chunks = load_indexes()
    all_docs = []
    source_filter = state["query_type"] if state["query_type"] in ["habr", "science", "news"] else None

    for query in state["sub_questions"]:
        scores = {}
        filter_dict = {"source": source_filter} if source_filter else None
        faiss_results = faiss_db.similarity_search(query, k=10, filter=filter_dict)

        for rank, doc in enumerate(faiss_results, 1):
            e = scores.setdefault(doc.metadata["chunk_id"], {"rrf": 0.0, "doc": doc})
            e["rrf"] += 1.0 / (RRF_K + rank)

        q_tokens = tokenize(query)
        if q_tokens:
            bm = bm25.get_scores(q_tokens)
            top = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)
            rank = 0
            for idx in top:
                if bm[idx] <= 0: break
                doc = chunks[idx]
                if source_filter and doc.metadata.get("source") != source_filter:
                    continue
                rank += 1
                e = scores.setdefault(doc.metadata["chunk_id"], {"rrf": 0.0, "doc": doc})
                e["rrf"] += 1.0 / (RRF_K + rank)
                if rank >= 10: break

        ranked = sorted(scores.values(), key=lambda e: e["rrf"], reverse=True)[:3]
        all_docs.extend([e["doc"] for e in ranked])

    unique_docs = {d.metadata["chunk_id"]: d for d in all_docs}.values()
    log_msg = f"📚 Поиск в '{source_filter or 'все'}': найдено {len(unique_docs)} фрагментов"

    chunks_for_ui = []
    for d in unique_docs:
        chunks_for_ui.append({
            "text": d.page_content,
            "title": d.metadata.get("title", "Без названия"),
            "source": d.metadata.get("source", "unknown")
        })

    return {"documents": list(unique_docs), "steps_log": state["steps_log"] + [log_msg], "used_chunks": chunks_for_ui,
            "cost": state.get("cost", 0.0)}


def grade_docs(state: GraphState):
    if not state["documents"]:
        return {"retry_count": state["retry_count"] + 1, "steps_log": state["steps_log"] + ["❌ Документы не найдены"],
                "cost": state.get("cost", 0.0)}

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    context = "\n\n".join([d.page_content[:500] for d in state["documents"]])
    prompt = ChatPromptTemplate.from_template("Вопрос: {q}\nКонтекст:\n{c}\n\nСодержит ли контекст ответ? (yes/no):")
    res = (prompt | llm | StrOutputParser()).invoke({"q": state["fixed_question"], "c": context}).lower()

    cost = get_cost("openai/gpt-oss-20b", 500, 5)
    if "yes" in res:
        return {"steps_log": state["steps_log"] + ["✅ Оценка: документы релевантны"],
                "cost": state.get("cost", 0.0) + cost}
    else:
        return {"retry_count": state["retry_count"] + 1, "steps_log": state["steps_log"] + ["⚠️ Оценка: нерелевантно"],
                "cost": state.get("cost", 0.0) + cost}


def rewrite_query(state: GraphState):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5)
    prompt = ChatPromptTemplate.from_template("Переформулируй вопрос для лучшего поиска. Только новый вопрос:\n{q}")
    new_q = (prompt | llm | StrOutputParser()).invoke({"q": state["fixed_question"]})

    cost = get_cost("openai/gpt-oss-20b", 50, 20)
    return {"fixed_question": new_q, "sub_questions": [new_q],
            "steps_log": state["steps_log"] + [f"🔄 Переписывание: '{new_q}'"], "cost": state.get("cost", 0.0) + cost}


def web_search(state: GraphState):
    try:
        tool = TavilySearchResults(max_results=3)
        results = tool.invoke({"query": state["fixed_question"]})
        text = "\n".join([f"[Web]: {r['content']}" for r in results])
    except Exception as e:
        text = f"Веб-поиск недоступен: {e}"

    docs = [
        Document(page_content=text, metadata={"title": "Результаты из интернета", "source": "web", "chunk_id": "web"})]

    # Очищаем старые мусорные чанки и записываем результат веб-поиска
    chunks_for_ui = [{"text": text, "title": "Веб-поиск (Tavily)", "source": "web"}]

    return {"documents": docs, "steps_log": state["steps_log"] + ["🌐 Веб-поиск выполнен"], "used_chunks": chunks_for_ui,
            "cost": state.get("cost", 0.0)}


def generate_answer(state: GraphState):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    context = "\n\n".join([f"[Источник {i + 1}]: {d.page_content}" for i, d in enumerate(state["documents"])])
    prompt = ChatPromptTemplate.from_template(
        "Ответь на вопрос, опираясь на контекст.\n\nКонтекст:\n{c}\n\nВопрос: {q}\n\nОтвет:"
    )
    ans = (prompt | llm | StrOutputParser()).invoke({"q": state["fixed_question"], "c": context})

    cost = get_cost("openai/gpt-oss-120b", len(context) // 4, len(ans) // 4)
    return {"generation": ans, "steps_log": state["steps_log"] + ["✍️ Генерация ответа завершена"],
            "cost": state.get("cost", 0.0) + cost}


def generate_general(state: GraphState):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    ans = llm.invoke(state["fixed_question"]).content

    cost = get_cost("openai/gpt-oss-120b", 50, len(ans) // 4)
    return {"generation": ans, "steps_log": state["steps_log"] + ["💬 Разговорный ответ"],
            "cost": state.get("cost", 0.0) + cost}


def route_after_analyze(state: GraphState) -> Literal["decompose_query", "generate_general"]:
    return "decompose_query" if state["query_type"] != "general" else "generate_general"


def route_after_grade(state: GraphState) -> Literal["generate_answer", "rewrite_query", "web_search"]:
    if "✅" in state["steps_log"][-1]:
        return "generate_answer"
    if "🌐" in state["steps_log"][-1]:
        return "generate_answer"
    if state["retry_count"] >= 2:
        return "web_search"
    return "rewrite_query"


workflow = StateGraph(GraphState)
workflow.add_node("fix_spelling", fix_spelling)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("grade_docs", grade_docs)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("generate_general", generate_general)

workflow.set_entry_point("fix_spelling")
workflow.add_edge("fix_spelling", "analyze_query")
workflow.add_conditional_edges("analyze_query", route_after_analyze)
workflow.add_edge("decompose_query", "retrieve_docs")
workflow.add_edge("retrieve_docs", "grade_docs")
workflow.add_conditional_edges("grade_docs", route_after_grade)
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("generate_general", END)

app_graph = workflow.compile()

# ==========================================
# ИНТЕРФЕЙС STREAMLIT
# ==========================================

st.set_page_config(page_title="Habr RAG | Dollhouse", page_icon="🎀", layout="wide", initial_sidebar_state="expanded")


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


BG_FILE = "background.jpg"

try:
    bg_base64 = get_base64_of_bin_file(BG_FILE)
    bg_style = f"background-image: url('data:image/jpeg;base64,{bg_base64}');"
except FileNotFoundError:
    bg_style = "background: linear-gradient(135deg, #ffe6f2 0%, #e0f7fa 100%);"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    .stApp {{
        {bg_style}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
        color: #333;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255, 255, 255, 0.4);
        z-index: -1;
        pointer-events: none;
    }}
    .main-title {{
        font-family: 'VT323', monospace;
        font-size: 4.5rem;
        font-weight: bold;
        color: #ff3399;
        text-shadow: 2px 2px 0px #fff, 4px 4px 0px #000;
        margin-bottom: 0px;
        text-align: center;
        letter-spacing: 2px;
    }}
    .sub-title {{
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-align: center;
        font-style: italic;
        background: rgba(255, 255, 255, 0.8);
        padding: 5px;
        border-radius: 5px;
        display: inline-block;
        width: 100%;
    }}
    .glass-card {{
        background: #f0f0f0;
        border: 2px solid #fff;
        border-right: 2px solid #888;
        border-bottom: 2px solid #888;
        padding: 10px 15px;
        margin-bottom: 10px;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        color: #000;
        position: relative;
    }}
    .glass-card::before {{
        content: "🖥️ system_log.exe";
        display: block;
        background: linear-gradient(90deg, #ff99cc, #ffccff);
        color: #000;
        font-size: 0.8rem;
        font-weight: bold;
        padding: 3px 5px;
        margin: -10px -15px 10px -15px;
        border-bottom: 2px solid #888;
    }}
    .answer-card {{
        background: #ffffff;
        border: 3px dashed #ff66b2;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 5px 5px 0px rgba(255, 102, 178, 0.3);
    }}
    .stButton>button {{
        background: #ff66b2;
        color: white;
        border-top: 2px solid #ff99cc;
        border-left: 2px solid #ff99cc;
        border-right: 2px solid #cc0066;
        border-bottom: 2px solid #cc0066;
        border-radius: 0px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        font-family: 'Courier New', monospace;
        transition: all 0.1s;
        width: 100%;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.2);
    }}
    .stButton>button:hover {{ background: #ff3385; color: #fff; }}
    .stButton>button:active {{
        border-top: 2px solid #cc0066; border-left: 2px solid #cc0066;
        border-right: 2px solid #ff99cc; border-bottom: 2px solid #ff99cc;
        transform: translate(2px, 2px); box-shadow: 1px 1px 0px rgba(0,0,0,0.2);
    }}
    .stTextInput>div>div>input {{
        background-color: #fff; color: #000; border: 2px solid #888;
        border-right: 2px solid #fff; border-bottom: 2px solid #fff;
        border-radius: 0px; padding: 10px; font-family: 'Courier New', monospace;
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.9); border-right: 2px dashed #ff66b2;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f0f0; border: 2px solid #fff;
        border-right: 2px solid #888; border-bottom: 2px solid #888;
        border-radius: 0px; padding: 5px 15px; font-weight: bold;
        color: #333; font-family: 'Courier New', monospace;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #ff99cc !important; color: #fff !important;
        border: 2px solid #fff !important; border-right: 2px solid #cc0066 !important;
        border-bottom: 2px solid #cc0066 !important;
    }}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []
if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0

with st.sidebar:
    st.markdown(
        '<div style="text-align: center;"><img src="https://media.tenor.com/9X6Vz3F8q0AAAAAM/hello-kitty.gif" width="100"></div>',
        unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: center; color: #ff66b2; font-family: \"VT323\", monospace; font-size: 2.5rem;'>🎀 Dollhouse RAG</h2>",
        unsafe_allow_html=True)
    st.markdown("### ⚙️ Настройки")

    raw_key = st.text_input("Ключ API Groq", type="password", help="Введите ключ gsk_...")
    if raw_key:
        clean_key = re.sub(r'[^a-zA-Z0-9\-_]', '', raw_key)
        os.environ["GROQ_API_KEY"] = clean_key

    st.divider()
    st.markdown("#### 📊 Статистика сессии")
    st.metric("Запросов сделано", len(st.session_state.history))
    st.metric("Общие затраты ($)", f"${st.session_state.total_cost:.6f}")
    st.divider()
    st.caption("сделано для интернета • v6.0")

st.markdown('<div class="main-title">Habr RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Мода нулевых, цифровые мечты и только хорошее настроение.</div>',
            unsafe_allow_html=True)

tab_chat, tab_cost, tab_graph = st.tabs(["💬 Диалог", "💰 Панель управления затратами", "🗺 Схема графа"])

with tab_chat:
    st.markdown("### Задайте свой вопрос")
    query = st.text_input("Введите вопрос (можно с опечатками или разговорный):", label_visibility="collapsed")

    if not os.environ.get("GROQ_API_KEY"):
        st.info("👉 Пожалуйста, введите ваш ключ API Groq в боковой панели слева.")
    elif st.button("Отправить запрос 🚀") and query:
        st.markdown("### 🪵 Процесс выполнения")

        steps_container = st.container()
        state = {"original_question": query, "fixed_question": query, "sub_questions": [],
                 "query_type": "", "documents": [], "generation": "", "retry_count": 0, "steps_log": [],
                 "used_chunks": [], "cost": 0.0}

        config = {}
        if LANGFUSE_ENABLED:
            trace_id = langfuse_client.create_trace_id()
            config = {
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_trace_id": trace_id}
            }

        with steps_container:
            for output in app_graph.stream(state, config=config):
                for node_name, update in output.items():
                    if "steps_log" in update:
                        state["steps_log"] = update["steps_log"]
                        for log in state["steps_log"]:
                            st.markdown(f'<div class="glass-card">{log}</div>', unsafe_allow_html=True)
                    if "documents" in update: state["documents"] = update["documents"]
                    if "generation" in update: state["generation"] = update["generation"]
                    if "used_chunks" in update: state["used_chunks"] = update["used_chunks"]
                    if "cost" in update: state["cost"] = update["cost"]

        # Итоговая стоимость: пытаемся взять из Langfuse, иначе локально
        final_cost = state["cost"]
        if LANGFUSE_ENABLED:
            try:
                time.sleep(1.5)
                trace = langfuse_client.get_trace(trace_id)
                if trace and trace.total_cost is not None:
                    final_cost = trace.total_cost
            except Exception:
                pass

        st.session_state.total_cost += final_cost

        st.markdown("### 💡 Ответ системы")
        st.markdown(f'<div class="answer-card">{state["generation"]}</div>', unsafe_allow_html=True)

        if state.get("used_chunks"):
            st.markdown("---")
            st.markdown("### 📌 Использованные фрагменты (чанки)")
            for i, chunk in enumerate(state["used_chunks"], 1):
                with st.container():
                    st.markdown(f"""
                    <div style="background: #fff; border: 2px solid #ff99cc; padding: 15px; border-radius: 5px; margin-bottom: 10px; box-shadow: 3px 3px 0px rgba(0,0,0,0.1);">
                        <div style="color: #ff3399; font-weight: bold; margin-bottom: 5px; font-family: 'Courier New', monospace;">
                            🎀 Фрагмент {i} • <span style="color: #666; font-size: 0.9em;">{chunk['source'].upper()}</span>
                        </div>
                        <div style="color: #333; font-size: 0.95em; margin-bottom: 8px;"><b>{chunk['title']}</b></div>
                        <div style="font-family: 'Courier New', monospace; font-size: 0.85em; color: #555; background: #f9f9f9; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">{chunk['text'][:400]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.session_state.history.append({
            "Вопрос": query,
            "Тип": state.get("query_type", "general"),
            "Стоимость ($)": round(final_cost, 6),
            "Длина ответа": len(state.get("generation", ""))
        })

with tab_cost:
    st.markdown("### 📊 Мониторинг затрат")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Общие затраты за сессию", f"${st.session_state.total_cost:.6f}")

        if LANGFUSE_ENABLED:
            st.success("✅ Интеграция с Langfuse активна")
            st.markdown("👉 [**Открыть Langfuse Dashboard**](https://cloud.langfuse.com/)")
        else:
            st.warning("⚠️ Langfuse не подключен. Затраты считаются локально (приблизительно).")

    with col2:
        st.markdown("#### Затраты по моделям")
        st.markdown("* **openai/gpt-oss-20b:** ~$0.075 / 1M токенов\n* **openai/gpt-oss-120b:** ~$0.15 / 1M токенов")
        st.caption("Стоимость указана приблизительно.")

    st.divider()
    st.markdown("#### История сессии")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Пока нет запросов в этой сессии.")

with tab_graph:
    st.markdown("### 🗺 Архитектура графа")
    st.markdown("Этот граф управляет процессом ответа: от исправления опечаток до веб-поиска и генерации.")
    try:
        graph_png = app_graph.get_graph().draw_mermaid_png()
        st.image(graph_png, use_container_width=True)
    except Exception as e:
        st.warning(f"Не удалось отрисовать граф: {e}. Убедитесь, что установлены `pygraphviz` и `graphviz`.")
