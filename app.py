#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — ENADE Extractor · Interface Streamlit
===============================================
Execute localmente:
    streamlit run app.py

Deploy no Streamlit Cloud:
    1. Faça push deste repo para o GitHub
    2. Em share.streamlit.io, aponte para app.py
    3. Em Settings → Secrets, adicione:
         SUPABASE_URL = "https://xyz.supabase.co"
         SUPABASE_KEY = "eyJ..."
         OPENAI_API_KEY = "sk-..."
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Garante imports do mesmo diretório
sys.path.insert(0, str(Path(__file__).parent))

from enade_extractor import (
    PILLOW_AVAILABLE,
    extract_page_media,
    extract_pdf_text_with_page_markers,
    group_questions_for_output,
    llm_clean_question,
    llm_classify_subarea,
    parse_answer_key_from_text,
    remove_questionario_percepcao,
    split_into_questions,
)
from db_manager import make_backend

# ─────────────────────────────────────────────
# Página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ENADE Extractor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Background ── */
.stApp { background: #080c14; color: #d4dae8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1a2236 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #8a96b0 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: #131929 !important;
    border-color: #1e2d45 !important;
    color: #c5cfe0 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 4px;
}
.app-logo {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    color: #fff;
    line-height: 1;
}
.app-logo span {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-badge {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b82f6;
    border: 1px solid #1e3a6e;
    background: #0a1628;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 4px;
}
.app-sub {
    font-size: 0.88rem;
    color: #4b5a75;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #0d1220;
    border: 1px solid #1a2236;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
}

/* ── Métricas custom ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.2rem 0;
}
.metric-card {
    background: #0d1220;
    border: 1px solid #1a2236;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.metric-num {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-lbl {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5a75;
}
.c-blue   { color: #3b82f6; }
.c-purple { color: #8b5cf6; }
.c-green  { color: #10b981; }
.c-amber  { color: #f59e0b; }
.c-red    { color: #ef4444; }

/* ── Status pill ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 4px 12px;
    border-radius: 999px;
}
.pill-ok   { background:#052e16; color:#34d399; border:1px solid #065f46; }
.pill-err  { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }
.pill-warn { background:#1c1506; color:#fbbf24; border:1px solid #78350f; }
.pill-info { background:#0a1628; color:#60a5fa; border:1px solid #1e3a8a; }

/* ── Log ── */
.logbox {
    background: #060912;
    border: 1px solid #141e30;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #34d399;
    max-height: 320px;
    overflow-y: auto;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ── Config row ── */
.cfg-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.42rem 0;
    border-bottom: 1px solid #111827;
    font-size: 0.84rem;
}
.cfg-key   { color: #4b5a75; }
.cfg-val   { font-weight: 500; color: #c5cfe0; }
.cfg-val.on  { color: #34d399; }
.cfg-val.off { color: #6b7280; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    border-radius: 7px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    color: #fff !important;
    padding: 0.6rem 1.6rem !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    filter: brightness(1.12) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #0d1220 !important;
    border: 1px solid #1e2d45 !important;
    color: #8a96b0 !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #4b5a75 !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
}
[data-testid="stTabsContent"] {
    padding-top: 1.2rem !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border-radius: 999px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d1220 !important;
    border: 1.5px dashed #1e2d45 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] p { color: #4b5a75 !important; font-size: 0.82rem !important; }

/* ── Inputs main area ── */
.stTextArea textarea,
.stTextInput input,
.stNumberInput input {
    background: #0d1220 !important;
    border-color: #1a2236 !important;
    color: #c5cfe0 !important;
    border-radius: 7px !important;
    font-size: 0.87rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1220 !important;
    border: 1px solid #1a2236 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: #8a96b0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #111827 !important; margin: 1.2rem 0 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "log_lines":    [],
    "questions":    [],
    "grouped":      {},
    "backend":      None,
    "db_ok":        None,       # None | True | False
    "db_msg":       "",
    "run_done":     False,
    "stats":        {},
    "db_upsert":    {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────
# Logging → session state
# ─────────────────────────────────────────────
class _STHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        st.session_state.log_lines.append(msg)
        if len(st.session_state.log_lines) > 400:
            st.session_state.log_lines = st.session_state.log_lines[-400:]

_handler = _STHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
_root_log = logging.getLogger()
if not any(isinstance(h, _STHandler) for h in _root_log.handlers):
    _root_log.addHandler(_handler)
_root_log.setLevel(logging.INFO)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _tmp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    t.write(uploaded.read())
    t.flush()
    return t.name


def _flatten(grouped: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out.extend(grouped.get("formacao_geral", []))
    for qs in grouped.get("componente_especifico", {}).values():
        out.extend(qs)
    out.extend(grouped.get("nao_classificadas", []))
    return out


def _pill(ok: Optional[bool], msg_ok="", msg_err="") -> str:
    if ok is True:
        return f"<span class='pill pill-ok'>● {msg_ok}</span>"
    if ok is False:
        return f"<span class='pill pill-err'>● {msg_err}</span>"
    return "<span class='pill pill-warn'>● Não testado</span>"


def _try_connect(btype: str, url: str, key: str, path: str):
    try:
        b = make_backend(btype, supabase_url=url, supabase_key=key, sqlite_path=path)
        ok, msg = b.test_connection()
        st.session_state.backend = b if ok else None
        st.session_state.db_ok   = ok
        st.session_state.db_msg  = msg
    except Exception as e:
        st.session_state.backend = None
        st.session_state.db_ok   = False
        st.session_state.db_msg  = str(e)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Configurações")

    # ── Exame ──
    st.markdown("---")
    st.markdown("**📌 Exame**")
    exam_name = st.text_input("Nome", value="ENADE", key="cfg_exam")
    c1, c2    = st.columns(2)
    exam_year = c1.number_input("Ano", 2000, 2099, 2022, step=1, key="cfg_year")
    exam_area = c2.text_input("Área", value="Administração", key="cfg_area")

    # ── LLM ──
    st.markdown("---")
    st.markdown("**🤖 OpenAI**")
    oai_key   = st.text_input("API Key", type="password",
                               value=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                               placeholder="sk-...", key="cfg_oai_key")
    llm_model = st.selectbox("Modelo", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], key="cfg_model")
    llm_clean    = st.toggle("Limpeza de texto", value=False, key="cfg_clean")
    llm_classify = st.toggle("Classificar subáreas", value=False, key="cfg_classify")
    llm_sleep    = st.slider("Pausa entre chamadas (s)", 0.0, 5.0, 0.5, 0.5, key="cfg_sleep")

    # ── Banco ──
    st.markdown("---")
    st.markdown("**🗄️ Banco de Dados**")
    db_type = st.selectbox(
        "Backend",
        ["supabase", "sqlite"],
        format_func=lambda x: {"supabase": "☁️ Supabase (produção)", "sqlite": "💾 SQLite (local)"}[x],
        key="cfg_db_type",
    )

    if db_type == "supabase":
        sb_url = st.text_input("Supabase URL",
                                value=st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", "")),
                                placeholder="https://xyz.supabase.co", key="cfg_sb_url")
        sb_key = st.text_input("Supabase Key", type="password",
                                value=st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", "")),
                                placeholder="eyJ...", key="cfg_sb_key")
        sq_path = "enade_dev.db"
    else:
        sb_url = sb_key = ""
        sq_path = st.text_input("Arquivo SQLite", value="enade_dev.db", key="cfg_sq")

    if st.button("🔌 Testar conexão", use_container_width=True, key="btn_test_db"):
        with st.spinner("Conectando…"):
            _try_connect(db_type, sb_url, sb_key, sq_path)

    st.markdown(
        _pill(st.session_state.db_ok,
              st.session_state.db_msg,
              st.session_state.db_msg),
        unsafe_allow_html=True,
    )

    # ── Mídia ──
    st.markdown("---")
    st.markdown("**🖼️ Figuras & Tabelas**")
    extract_media = st.toggle(
        "Extrair figuras e tabelas",
        value=False,
        disabled=not PILLOW_AVAILABLE,
        key="cfg_media",
        help="Requer Pillow. `pip install Pillow`" if not PILLOW_AVAILABLE else "",
    )
    if not PILLOW_AVAILABLE:
        st.caption("⚠️ Pillow não instalado.")
    figures_dir   = st.text_input("Pasta das figuras", "figures",
                                   disabled=not extract_media, key="cfg_figdir")
    resolution    = st.select_slider("DPI", [72, 100, 150, 200, 300], 150,
                                     disabled=not extract_media, key="cfg_dpi")
    min_fig_sz    = st.slider("Tamanho mín. (pt)", 20.0, 200.0, 80.0, 10.0,
                               disabled=not extract_media, key="cfg_figmin")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">ENADE<span>Extractor</span></div>
  <div class="app-badge">v2.0 · Produção</div>
</div>
<div class="app-sub">
  Pipeline de extração, limpeza e armazenamento de questões do ENADE · Supabase + GPT-4o
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_ext, tab_res, tab_db, tab_log = st.tabs([
    "📄  Extração",
    "📊  Resultados",
    "🗄️  Banco de Dados",
    "🖥️  Log",
])


# ══════════════════════════════════════════════
# TAB 1 – EXTRAÇÃO
# ══════════════════════════════════════════════
with tab_ext:

    col_up, col_cfg = st.columns([1, 1], gap="large")

    # ── Upload ──────────────────────────────
    with col_up:
        st.markdown("<div class='card-title'>📂 Arquivos PDF</div>", unsafe_allow_html=True)
        pdf_prova    = st.file_uploader("📘 Prova (obrigatório)", type=["pdf"], key="up_prova")
        pdf_gabarito = st.file_uploader("📋 Gabarito (opcional)", type=["pdf"], key="up_gab")

        st.markdown("<br><div class='card-title'>🏷️ Subáreas do Componente Específico</div>",
                    unsafe_allow_html=True)
        subareas_raw = st.text_area(
            "Subáreas",
            value="Finanças,Marketing,Recursos Humanos,Produção,Estratégia",
            height=72,
            label_visibility="collapsed",
            placeholder="Finanças,Marketing,Recursos Humanos,Produção",
            key="cfg_subareas",
        )
        subareas_list = [s.strip() for s in subareas_raw.split(",") if s.strip()]
        if subareas_list:
            pills_html = " ".join(
                f"<span class='pill pill-info' style='margin:2px;font-size:0.72rem;'>{s}</span>"
                for s in subareas_list
            )
            st.markdown(pills_html, unsafe_allow_html=True)

    # ── Resumo config ────────────────────────
    with col_cfg:
        st.markdown("<div class='card-title'>⚙️ Configuração Ativa</div>", unsafe_allow_html=True)

        def _row(label: str, val: str, cls: str = ""):
            return (f"<div class='cfg-row'>"
                    f"<span class='cfg-key'>{label}</span>"
                    f"<span class='cfg-val {cls}'>{val}</span></div>")

        rows_html = "".join([
            _row("Exame",            f"{exam_name} · {int(exam_year)} · {exam_area}"),
            _row("LLM modelo",       llm_model if (llm_clean or llm_classify) else "Desativado",
                 "on" if (llm_clean or llm_classify) else "off"),
            _row("Limpeza LLM",      "✓ Ativa" if llm_clean    else "✗ Inativa",
                 "on" if llm_clean    else "off"),
            _row("Classificação LLM","✓ Ativa" if llm_classify else "✗ Inativa",
                 "on" if llm_classify else "off"),
            _row("Extrair mídia",    "✓ Ativa" if extract_media else "✗ Inativa",
                 "on" if extract_media else "off"),
            _row("Subáreas",         f"{len(subareas_list)} definida(s)"),
            _row("Banco",            db_type.upper()),
            _row("DB status",        "✓ Conectado" if st.session_state.db_ok else "✗ Não testado",
                 "on" if st.session_state.db_ok else "off"),
        ])
        st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Botão de execução ────────────────────
    col_btn, col_warn = st.columns([1, 2])
    with col_btn:
        run = st.button("▶  Iniciar Extração", type="primary",
                        disabled=pdf_prova is None, use_container_width=True, key="btn_run")
    with col_warn:
        if pdf_prova is None:
            st.warning("Selecione o PDF da prova para continuar.")
        elif (llm_clean or llm_classify) and not oai_key:
            st.error("API Key da OpenAI não configurada.")

    # ══════════════════════════
    # Pipeline
    # ══════════════════════════
    if run and pdf_prova:

        # Reset
        st.session_state.log_lines = []
        st.session_state.run_done  = False
        st.session_state.questions = []
        st.session_state.grouped   = {}
        st.session_state.stats     = {}
        st.session_state.db_upsert = {}

        if oai_key:
            os.environ["OPENAI_API_KEY"] = oai_key

        bar    = st.progress(0, text="Iniciando…")
        status = st.empty()

        def _upd(pct: int, msg: str):
            bar.progress(pct, text=msg)
            status.markdown(
                f"<span class='pill pill-info'>⚙ {msg}</span>",
                unsafe_allow_html=True,
            )

        try:
            # 1 · Salvar temporários
            _upd(3, "Salvando arquivos…")
            prova_tmp = _tmp(pdf_prova)
            gab_tmp   = _tmp(pdf_gabarito) if pdf_gabarito else None

            # 2 · Gabarito
            answer_key: Dict[int, str] = {}
            if gab_tmp:
                _upd(10, "Lendo gabarito…")
                key_text   = extract_pdf_text_with_page_markers(gab_tmp)
                answer_key = parse_answer_key_from_text(key_text)
                logging.info("Gabarito: %d resposta(s) lida(s).", len(answer_key))

            # 3 · Texto da prova
            _upd(20, "Extraindo texto da prova…")
            full_text = extract_pdf_text_with_page_markers(prova_tmp)
            full_text = remove_questionario_percepcao(full_text)

            # 4 · Segmentar questões
            _upd(30, "Segmentando questões…")
            questions = split_into_questions(
                full_text  = full_text,
                exam_name  = exam_name,
                year       = int(exam_year),
                area       = exam_area,
                answer_key = answer_key,
                only_objective = True,
            )
            if not questions:
                status.error("Nenhuma questão objetiva encontrada. Verifique o PDF.")
                bar.empty()
                st.stop()

            logging.info("%d questões objetivas encontradas.", len(questions))

            # 5 · LLM + mídia por questão
            client = None
            if (llm_clean or llm_classify) and oai_key:
                from openai import OpenAI
                client = OpenAI(api_key=oai_key)

            n = len(questions)
            for i, q in enumerate(questions):
                pct = 30 + int((i / n) * 55)
                _upd(pct, f"Q{q.q_number} ({i+1}/{n}) — {q.section_type}")
                logging.info("Q%d (%d/%d) pág. %s–%s", q.q_number, i+1, n,
                             q.page_start, q.page_end)

                tables_md:   List[str] = []
                figures_b64: List[str] = []

                # Extração de mídia
                if extract_media and q.page_start and q.page_end:
                    tables_md, fig_paths, figures_b64 = extract_page_media(
                        pdf_path      = prova_tmp,
                        page_start    = q.page_start,
                        page_end      = q.page_end,
                        q_number      = q.q_number,
                        figures_dir   = figures_dir if PILLOW_AVAILABLE else None,
                        resolution    = resolution,
                        min_img_width = min_fig_sz,
                        min_img_height= min_fig_sz,
                    )
                    q.tables  = tables_md  or None
                    q.figures = fig_paths  or None
                    if tables_md:
                        logging.info("  Q%d → %d tabela(s)", q.q_number, len(tables_md))
                    if fig_paths:
                        logging.info("  Q%d → %d figura(s)", q.q_number, len(fig_paths))

                # Limpeza LLM
                if llm_clean and client:
                    try:
                        cleaned = llm_clean_question(
                            client       = client,
                            statement_raw= q.statement,
                            figures_b64  = figures_b64,
                            tables_md    = tables_md,
                            model        = llm_model,
                        )
                        q.statement_clean = cleaned.get("statement_clean")
                        q.prompt          = cleaned.get("prompt")
                        q.sections        = cleaned.get("sections")
                        q.alternatives    = cleaned.get("alternatives")
                        q.llm_clean_failed = False
                        if q.statement_clean:
                            q.statement = q.statement_clean
                    except Exception as exc:
                        logging.error("Limpeza Q%d: %s", q.q_number, exc)
                        q.llm_clean_failed = True

                # Classificação subárea
                if q.section_type == "Componente Específico":
                    if llm_classify and client and subareas_list:
                        try:
                            res = llm_classify_subarea(
                                client        = client,
                                question_text = q.statement,
                                subareas      = subareas_list,
                                model         = llm_model,
                            )
                            q.subarea = res.get("subarea", "Não classificada")
                            q.llm_classify_failed = (
                                q.subarea == "Não classificada"
                                and res.get("confidence", 1.0) == 0.0
                            )
                        except Exception as exc:
                            logging.error("Classif. Q%d: %s", q.q_number, exc)
                            q.subarea = "Não classificada"
                            q.llm_classify_failed = True
                    else:
                        q.subarea = "Não classificada"

                if llm_sleep > 0:
                    time.sleep(llm_sleep)

            # 6 · Agrupar
            _upd(87, "Agrupando resultados…")
            effective_subareas = subareas_list or ["Não classificada"]
            grouped = group_questions_for_output(questions, effective_subareas)

            n_fg = len(grouped["formacao_geral"])
            # CE: conta todas as questões em todos os buckets (incluindo "Não classificada")
            n_ce = sum(len(v) for v in grouped["componente_especifico"].values())
            # nao_classificadas: apenas questões sem section_type reconhecido
            n_nc = len(grouped["nao_classificadas"])

            st.session_state.questions = questions
            st.session_state.grouped   = grouped
            st.session_state.stats     = {
                "total": n, "fg": n_fg, "ce": n_ce, "nc": n_nc
            }

            # 7 · Salvar no banco
            _upd(93, "Salvando no banco de dados…")
            if st.session_state.backend is None:
                _try_connect(db_type, sb_url, sb_key, sq_path)

            db_res: Dict[str, int] = {"inserted": 0, "errors": 0}
            if st.session_state.backend:
                flat   = _flatten(grouped)
                db_res = st.session_state.backend.upsert_questions(flat)
                logging.info(
                    "Banco → %d inseridas | %d erros",
                    db_res.get("inserted", 0), db_res.get("errors", 0),
                )
            elif st.session_state.db_ok is None:
                logging.warning(
                    "Banco não testado — clique em 'Testar conexão' na barra lateral "
                    "e execute novamente, ou verifique se o DDL foi criado no Supabase."
                )

            st.session_state.db_upsert = db_res
            st.session_state.run_done  = True

            # Limpeza dos temporários
            for p in [prova_tmp, gab_tmp]:
                if p:
                    Path(p).unlink(missing_ok=True)

            bar.progress(100, text="Concluído!")
            status.empty()

            # Banner de sucesso
            ins_n = db_res.get("inserted", 0)
            err_n = db_res.get("errors", 0)
            success_msg = (
                f"✅ **{n} questões** extraídas — "
                f"Formação Geral: **{n_fg}** | Comp. Específico: **{n_ce}** | Sem seção: **{n_nc}**\n\n"
                f"🗄️ Banco: **{ins_n}** questão(ões) salvas"
                + (f" · ⚠️ {err_n} erro(s) — veja o **Log** para detalhes" if err_n else "")
            )
            if err_n and ins_n == 0:
                success_msg += (
                    "\n\n💡 **Dica:** Se todos falharam, verifique se o DDL foi executado "
                    "no Supabase (aba 🗄️ Banco de Dados → Ver DDL completo)."
                )
            st.success(success_msg)

        except Exception as exc:
            bar.empty()
            status.empty()
            st.error(f"❌ Erro durante a extração: {exc}")
            logging.exception("Erro na pipeline")


# ══════════════════════════════════════════════
# TAB 2 – RESULTADOS
# ══════════════════════════════════════════════
with tab_res:
    if not st.session_state.run_done:
        st.info("Execute a extração na aba **📄 Extração** para visualizar os resultados.")
    else:
        stats   = st.session_state.stats
        grouped = st.session_state.grouped

        # Métricas
        st.markdown(f"""
        <div class="metrics-row">
          <div class="metric-card">
            <div class="metric-num c-blue">{stats.get("total", 0)}</div>
            <div class="metric-lbl">Total extraídas</div>
          </div>
          <div class="metric-card">
            <div class="metric-num c-purple">{stats.get("fg", 0)}</div>
            <div class="metric-lbl">Formação Geral</div>
          </div>
          <div class="metric-card">
            <div class="metric-num c-green">{stats.get("ce", 0)}</div>
            <div class="metric-lbl">Comp. Específico</div>
          </div>
          <div class="metric-card">
            <div class="metric-num c-amber">{stats.get("nc", 0)}</div>
            <div class="metric-lbl">Não classificadas</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Download
        json_bytes = json.dumps(grouped, ensure_ascii=False, indent=2).encode()
        st.download_button(
            "⬇️  Baixar JSON completo",
            data     = json_bytes,
            file_name= f"enade_{exam_name}_{int(exam_year)}.json",
            mime     = "application/json",
        )

        st.markdown("---")

        def _show_questions(qs: List[Dict], max_show: int = 15):
            if not qs:
                st.caption("Nenhuma questão nesta categoria.")
                return
            for q in qs[:max_show]:
                ans  = q.get("answer") or "—"
                sub  = f" · {q['subarea']}" if q.get("subarea") else ""
                with st.expander(
                    f"Q{q['q_number']}  ·  Resp: {ans}  ·  "
                    f"Págs {q.get('page_start')}–{q.get('page_end')}{sub}"
                ):
                    stmt = q.get("statement_clean") or q.get("statement", "")
                    st.markdown(
                        f"<div style='font-size:0.87rem;color:#c5cfe0;line-height:1.7;"
                        f"white-space:pre-wrap;'>{stmt}</div>",
                        unsafe_allow_html=True,
                    )
                    alts = q.get("alternatives")
                    if alts and isinstance(alts, dict):
                        st.markdown("<br>**Alternativas:**", unsafe_allow_html=True)
                        for letter, text in alts.items():
                            color = "#34d399" if letter == ans else "#4b5a75"
                            st.markdown(
                                f"<div style='color:{color};font-size:0.84rem;margin:2px 0;'>"
                                f"<b>{letter})</b> {text}</div>",
                                unsafe_allow_html=True,
                            )
                    tabs_q = q.get("tables") or []
                    figs_q = q.get("figures") or []
                    if tabs_q:
                        st.caption(f"📊 {len(tabs_q)} tabela(s) extraída(s)")
                    if figs_q:
                        st.caption(f"🖼️ {len(figs_q)} figura(s) extraída(s)")
                        for fp in figs_q:
                            if Path(fp).exists():
                                st.image(fp, width=380)
            if len(qs) > max_show:
                st.caption(f"… e mais {len(qs)-max_show} questão(ões) no JSON completo.")

        tab_fg, tab_ce, tab_nc = st.tabs(["Formação Geral", "Comp. Específico", "Não Classificadas"])

        with tab_fg:
            fg = grouped.get("formacao_geral", [])
            st.caption(f"{len(fg)} questão(ões)")
            _show_questions(fg)

        with tab_ce:
            ce = grouped.get("componente_especifico", {})
            for sub, qs in ce.items():
                if qs:
                    st.markdown(
                        f"<span class='pill pill-info'>{sub} ({len(qs)})</span>",
                        unsafe_allow_html=True,
                    )
                    _show_questions(qs, max_show=5)
                    st.markdown("<br>", unsafe_allow_html=True)

        with tab_nc:
            nc = grouped.get("nao_classificadas", [])
            st.caption(f"{len(nc)} questão(ões)")
            _show_questions(nc)


# ══════════════════════════════════════════════
# TAB 3 – BANCO DE DADOS
# ══════════════════════════════════════════════
with tab_db:
    col_status, col_preview = st.columns([1, 2], gap="large")

    with col_status:
        st.markdown("<div class='card-title'>Status da Conexão</div>", unsafe_allow_html=True)
        st.markdown(
            _pill(st.session_state.db_ok,
                  st.session_state.db_msg,
                  st.session_state.db_msg),
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.backend and st.session_state.db_ok:
            b      = st.session_state.backend
            total  = b.count_questions()
            n_year = b.count_questions(year=int(exam_year))
            sec    = b.fetch_stats()

            st.markdown(f"""
            <div class="metrics-row" style="grid-template-columns:1fr 1fr;">
              <div class="metric-card">
                <div class="metric-num c-blue">{total}</div>
                <div class="metric-lbl">Total no banco</div>
              </div>
              <div class="metric-card">
                <div class="metric-num c-purple">{n_year}</div>
                <div class="metric-lbl">Ano {int(exam_year)}</div>
              </div>
              <div class="metric-card">
                <div class="metric-num c-green">{sec.get("Formação Geral", 0)}</div>
                <div class="metric-lbl">Form. Geral</div>
              </div>
              <div class="metric-card">
                <div class="metric-num c-amber">{sec.get("Componente Específico", 0)}</div>
                <div class="metric-lbl">Comp. Espec.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.db_upsert:
                ins = st.session_state.db_upsert.get("inserted", 0)
                err = st.session_state.db_upsert.get("errors", 0)
                st.markdown(
                    f"<span class='pill pill-ok'>↑ {ins} inseridas</span> "
                    + (f"<span class='pill pill-err'>⚠ {err} erros</span>" if err else ""),
                    unsafe_allow_html=True,
                )

            if st.button("🔄 Atualizar", key="btn_refresh_db"):
                st.rerun()

        st.markdown("---")
        st.markdown("<div class='card-title'>DDL — Supabase SQL Editor</div>",
                    unsafe_allow_html=True)
        st.caption("Execute este SQL no Supabase → SQL Editor antes do primeiro uso.")
        from db_manager import DDL_SUPABASE
        with st.expander("Ver DDL completo"):
            st.code(DDL_SUPABASE.strip(), language="sql")

    with col_preview:
        st.markdown("<div class='card-title'>Últimas Inserções</div>", unsafe_allow_html=True)
        if st.session_state.backend and st.session_state.db_ok:
            sample = st.session_state.backend.fetch_recent(limit=12)
            if sample:
                cols_show = ["id", "q_number", "year", "area", "section_type",
                             "subarea", "answer", "inserted_at"]
                rows_disp = [{c: row.get(c) for c in cols_show} for row in sample]
                st.dataframe(rows_disp, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma questão no banco ainda.")
        else:
            st.info("Conecte ao banco na barra lateral para ver as inserções.")


# ══════════════════════════════════════════════
# TAB 4 – LOG
# ══════════════════════════════════════════════
with tab_log:
    c_log, c_clr = st.columns([6, 1])
    with c_clr:
        if st.button("🗑️ Limpar", key="btn_clr_log"):
            st.session_state.log_lines = []
            st.rerun()

    log_content = "\n".join(st.session_state.log_lines) or "— Nenhum log ainda. Execute a extração. —"
    st.markdown(
        f"<div class='logbox'>{log_content}</div>",
        unsafe_allow_html=True,
    )
