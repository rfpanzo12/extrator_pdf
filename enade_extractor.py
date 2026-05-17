#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enade_extractor.py
==================
Pipeline de extração de questões objetivas do ENADE a partir de PDFs.

Funcionalidades:
  - Extração de texto com marcadores de página
  - Detecção automática de seções (Formação Geral / Componente Específico)
  - Leitura de gabarito
  - Extração de tabelas via pdfplumber (salvas como Markdown)
  - Extração de figuras via renderização de página (salvas como PNG)
  - Limpeza de questões com LLM multimodal (GPT-4o com visão)
  - Classificação de subáreas com LLM
  - Saída agrupada em JSON

Dependências:
  pip install pdfplumber openai Pillow

Para extração de figuras, o pdfplumber usa pdf2image ou pypdfium2 internamente.
Se houver erro de renderização, instale: pip install pypdfium2
"""

import os
import re
import time
import argparse
import json
import bisect
import base64
import logging
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pdfplumber
from openai import OpenAI

# Pillow é necessário apenas para extração de figuras
try:
    from PIL import Image as PILImage  # noqa: F401 (usado via pdfplumber)
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

PDF_DEFAULT = "enade.pdf"

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -----------------------------
# Modelos de dados
# -----------------------------
@dataclass
class Question:
    exam: str
    year: Optional[int]
    area: Optional[str]
    q_number: int
    q_type: str          # "objetiva" | "discursiva"
    section_type: str    # "Formação Geral" | "Componente Específico"
    subarea: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    statement: str
    answer: Optional[str]

    # Mídia extraída do PDF
    tables: Optional[List[str]] = None    # tabelas em Markdown
    figures: Optional[List[str]] = None   # caminhos para arquivos PNG

    # Campos de limpeza (preenchidos pelo LLM)
    statement_clean: Optional[str] = None
    prompt: Optional[str] = None
    sections: Optional[List[Dict[str, str]]] = None
    alternatives: Optional[Dict[str, str]] = None

    # Rastreamento de falhas
    llm_clean_failed: Optional[bool] = None
    llm_classify_failed: Optional[bool] = None


# -----------------------------
# Utilitários de texto / PDF
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")       # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s


def extract_pdf_text_with_page_markers(pdf_path: str) -> str:
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = normalize_text(text)
            chunks.append(f"\n<<PAGE:{i}>>\n{text}\n")
    return "\n".join(chunks)


def guess_page_range(fragment: str) -> Tuple[Optional[int], Optional[int]]:
    pages = [int(p) for p in re.findall(r"<<PAGE:(\d+)>>", fragment)]
    if not pages:
        return None, None
    return min(pages), max(pages)


def remove_questionario_percepcao(full_text: str) -> str:
    patterns = [
        r"QUESTION[AÁ]RIO\s+DE\s+PERCEP[CÇ][AÃ]O\s+DA\s+PROVA",
        r"AVALIA[CÇ][AÃ]O\s+GLOBAL\s+DA\s+PROVA",
    ]
    cut_index = len(full_text)
    found = False
    for pat in patterns:
        for m in re.finditer(pat, full_text, flags=re.IGNORECASE | re.DOTALL):
            if m.start() > len(full_text) * 0.5:
                cut_index = min(cut_index, m.start())
                found = True
    return full_text[:cut_index] if found else full_text


# -----------------------------
# Gabarito
# -----------------------------
def parse_answer_key_from_text(
    key_text: str,
    min_q: int = 1,
    max_q: int = 200,
) -> Dict[int, str]:
    t = re.sub(r"<<PAGE:\d+>>", "\n", key_text)
    t = normalize_text(t)
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    answer_map: Dict[int, str] = {}

    patterns = [
        re.compile(r"\bQUESTÃO\s*(\d{1,2})\s*[-–:]\s*([A-E])\b", re.IGNORECASE),
        re.compile(r"^\s*(\d{1,2})\s*[-–:]\s*([A-E])\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*(\d{1,2})\s+([A-E])\b", re.IGNORECASE | re.MULTILINE),
    ]
    fallback_pat = re.compile(r"\b(\d{1,2})\b\s{0,10}([A-E])\b", re.IGNORECASE)

    for ln in lines:
        if len(ln) <= 30:
            for pat in patterns:
                m = pat.search(ln)
                if m:
                    q, a = int(m.group(1)), m.group(2).upper()
                    if min_q <= q <= max_q:
                        answer_map[q] = a
                    break

    if len(answer_map) < 5:
        log.warning("Poucos itens no gabarito (%d); tentando fallback.", len(answer_map))
        for ln in lines:
            if len(ln) <= 15:
                for m in fallback_pat.finditer(ln):
                    q, a = int(m.group(1)), m.group(2).upper()
                    if min_q <= q <= max_q:
                        answer_map.setdefault(q, a)

    return answer_map


# -----------------------------
# Detecção de seções do exame
# -----------------------------
def detect_exam_sections(full_text: str) -> List[Tuple[int, str]]:
    markers: List[Tuple[int, str]] = []
    for pat, label in [
        (r"FORMA[CÇ][AÃ]O\s+GERAL", "Formação Geral"),
        (r"COMPONENTE\s+ESPEC[ÍI]FIC[OA]", "Componente Específico"),
    ]:
        for m in re.finditer(pat, full_text, flags=re.IGNORECASE):
            markers.append((m.start(), label))
    markers.sort(key=lambda x: x[0])
    if not markers:
        log.warning("Nenhuma seção encontrada; assumindo tudo como Formação Geral.")
        markers = [(0, "Formação Geral")]
    return markers


def get_section_for_position(pos: int, markers: List[Tuple[int, str]]) -> str:
    positions = [p for p, _ in markers]
    idx = bisect.bisect_right(positions, pos) - 1
    return markers[max(idx, 0)][1]


# -----------------------------
# Extração de questões
# -----------------------------
def split_into_questions(
    full_text: str,
    exam_name: str,
    year: Optional[int],
    area: Optional[str],
    answer_key: Optional[Dict[int, str]] = None,
    only_objective: bool = True,
) -> List[Question]:
    header_re = re.compile(
        r"^QUEST[AÃÁ]O\s+(DISCURSIVA\s+)?(?:N[º°]\s*)?(\d{1,2})\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(header_re.finditer(full_text))
    if not matches:
        log.warning("Nenhum cabeçalho de questão encontrado.")
        return []

    section_markers = detect_exam_sections(full_text)
    questions: List[Question] = []
    answer_key = answer_key or {}

    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()

        disc_flag = m.group(1)
        qnum = int(m.group(2))
        q_type = "discursiva" if disc_flag else "objetiva"

        if only_objective and q_type != "objetiva":
            continue

        page_start, page_end = guess_page_range(block)
        section_type = get_section_for_position(start, section_markers)

        statement = header_re.sub("", block, count=1).strip()
        statement = re.sub(r"<<PAGE:\d+>>", "", statement).strip()
        statement = normalize_text(statement)

        if len(statement) < 30:
            log.debug("Q%d ignorada — enunciado muito curto (%d chars).", qnum, len(statement))
            continue

        questions.append(
            Question(
                exam=exam_name,
                year=year,
                area=area,
                q_number=qnum,
                q_type=q_type,
                section_type=section_type,
                subarea=None,
                page_start=page_start,
                page_end=page_end,
                statement=statement,
                answer=answer_key.get(qnum),
            )
        )
    return questions


# ==========================================
# Extração de TABELAS e FIGURAS
# ==========================================

def table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """Converte uma tabela retornada pelo pdfplumber em Markdown."""
    if not table:
        return ""
    rows: List[str] = []
    for i, row in enumerate(table):
        cells = [
            (str(c).replace("|", "\\|").replace("\n", " ").strip() if c is not None else "")
            for c in row
        ]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:  # separador após cabeçalho
            rows.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(rows)


def _render_page_pil(page: Any, resolution: int) -> Optional[Any]:
    """Renderiza uma página pdfplumber como imagem PIL. Retorna None em caso de erro."""
    try:
        return page.to_image(resolution=resolution).original
    except Exception as e:
        log.warning("Falha ao renderizar página %d: %s", page.page_number, e)
        return None


def _pil_to_png_bytes(img: Any) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_page_media(
    pdf_path: str,
    page_start: int,
    page_end: int,
    q_number: int,
    figures_dir: Optional[str] = None,
    resolution: int = 150,
    min_img_width: float = 80.0,
    min_img_height: float = 80.0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extrai tabelas e figuras das páginas indicadas de um PDF.

    Parâmetros
    ----------
    pdf_path       : caminho do PDF da prova
    page_start     : primeira página da questão (1-based)
    page_end       : última página da questão (1-based)
    q_number       : número da questão (para nomear arquivos de saída)
    figures_dir    : diretório onde salvar os PNGs das figuras; None = não salvar
    resolution     : DPI para renderização das páginas
    min_img_width  : largura mínima (pontos PDF) para aceitar uma figura
    min_img_height : altura mínima (pontos PDF) para aceitar uma figura

    Retorna
    -------
    tables_md    : tabelas formatadas em Markdown
    figure_paths : caminhos dos PNGs salvos em disco
    figures_b64  : strings base64 de cada figura (usadas pelo LLM via vision)
    """
    tables_md: List[str] = []
    figure_paths: List[str] = []
    figures_b64: List[str] = []

    save_figs = figures_dir is not None and PILLOW_AVAILABLE
    if figures_dir and not PILLOW_AVAILABLE:
        log.warning(
            "Pillow não instalado — figuras não serão salvas. "
            "Instale com: pip install Pillow"
        )
    if save_figs and figures_dir:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        p_start = max(1, page_start) - 1      # converte para índice 0-based
        p_end   = min(page_end, n_pages)       # inclusive 1-based

        for page_idx in range(p_start, p_end):
            page = pdf.pages[page_idx]
            page_label = page_idx + 1

            # --------------------------------------------------
            # TABELAS
            # pdfplumber detecta automaticamente regiões tabulares
            # e retorna uma lista de tabelas, cada uma como lista de linhas.
            # --------------------------------------------------
            for raw_table in page.extract_tables() or []:
                md = table_to_markdown(raw_table)
                if md:
                    tables_md.append(md)
                    log.debug(
                        "  Tabela na pág.%d (Q%d): %d lin × %d col",
                        page_label, q_number,
                        len(raw_table),
                        len(raw_table[0]) if raw_table else 0,
                    )

            # --------------------------------------------------
            # FIGURAS
            # page.images retorna metadados de cada objeto de imagem
            # embutido no PDF (posição, dimensão, etc.).
            # Renderizamos a página inteira como PIL e recortamos
            # cada região de imagem individualmente.
            # --------------------------------------------------
            img_metas = page.images or []
            if not img_metas or not PILLOW_AVAILABLE:
                continue

            # Renderizar a página uma única vez (custo fixo por página)
            page_pil = _render_page_pil(page, resolution)
            if page_pil is None:
                continue

            # Fatores de escala: pontos PDF → pixels da imagem renderizada
            scale_x = page_pil.width  / float(page.width)
            scale_y = page_pil.height / float(page.height)

            seen_bboxes: set = set()  # deduplicação de regiões idênticas

            for img_meta in img_metas:
                x0  = img_meta.get("x0",     0.0)
                top = img_meta.get("top",    0.0)
                x1  = img_meta.get("x1",     0.0)
                bot = img_meta.get("bottom", 0.0)

                # Descartar objetos muito pequenos (marcadores, ícones, etc.)
                if (x1 - x0) < min_img_width or (bot - top) < min_img_height:
                    continue

                # Evitar duplicatas (mesmo bbox na página)
                bbox_key = (round(x0, 1), round(top, 1), round(x1, 1), round(bot, 1))
                if bbox_key in seen_bboxes:
                    continue
                seen_bboxes.add(bbox_key)

                # Converter coordenadas PDF → pixels, com clamp nos limites da imagem
                x0_px  = max(0,               int(x0  * scale_x))
                top_px = max(0,               int(top * scale_y))
                x1_px  = min(page_pil.width,  int(x1  * scale_x))
                bot_px = min(page_pil.height, int(bot * scale_y))

                if x1_px <= x0_px or bot_px <= top_px:
                    continue

                try:
                    cropped   = page_pil.crop((x0_px, top_px, x1_px, bot_px))
                    img_bytes = _pil_to_png_bytes(cropped)

                    # Base64 sempre gerado (necessário para envio ao LLM)
                    figures_b64.append(base64.b64encode(img_bytes).decode("utf-8"))

                    # Salvar PNG em disco (somente se figures_dir foi especificado)
                    if save_figs and figures_dir:
                        fig_idx = len(figure_paths) + 1
                        fname   = f"q{q_number:02d}_p{page_label}_fig{fig_idx}.png"
                        fpath   = str(Path(figures_dir) / fname)
                        Path(fpath).write_bytes(img_bytes)
                        figure_paths.append(fpath)
                        log.debug("  Figura salva: %s", fpath)

                except Exception as e:
                    log.warning(
                        "Falha ao recortar figura na pág.%d (Q%d): %s",
                        page_label, q_number, e,
                    )

    return tables_md, figure_paths, figures_b64


# ==========================================
# LLM — schemas JSON
# ==========================================

QUESTION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "enade_question_cleaning",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement_clean": {"type": "string"},
            "prompt": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["texto_base", "tabela", "figura", "outro"],
                        },
                        "label":   {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["type", "label", "content"],
                },
            },
            "alternatives": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                            "E": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D", "E"],
                    },
                ]
            },
        },
        "required": ["statement_clean", "prompt", "sections", "alternatives"],
    },
}

SUBAREA_SCHEMA: Dict[str, Any] = {
    "name": "enade_subarea_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subarea":    {"type": "string"},
            "confidence": {"type": "number"},
            "reason":     {"type": "string"},
        },
        "required": ["subarea", "confidence", "reason"],
    },
}


# ==========================================
# LLM — helper de chamada (multimodal)
# ==========================================

def _build_multimodal_content(
    text: str,
    figures_b64: List[str],
    tables_md: Optional[List[str]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """
    Monta o campo `content` da mensagem do usuário.

    - Tabelas Markdown são acrescentadas ao bloco de texto.
    - Figuras são anexadas como `image_url` (GPT-4o / vision).
    - Se não houver figuras, retorna str simples (sem custo de vision).
    """
    if tables_md:
        tables_block = (
            "\n\n--- TABELAS EXTRAÍDAS DO PDF ---\n"
            + "\n\n".join(tables_md)
        )
        text = text + tables_block

    if not figures_b64:
        return text  # chamada de texto puro

    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for b64 in figures_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    return content


def _call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: Union[str, List[Dict[str, Any]]],
    response_format: Dict[str, Any],
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Chama client.chat.completions com suporte a conteúdo multimodal.
    Faz retry com backoff exponencial simples.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                response_format=response_format,
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            return json.loads(raw)
        except Exception as e:
            last_err = e
            log.warning("Tentativa %d/%d falhou: %s", attempt + 1, retries + 1, e)
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Todas as tentativas LLM falharam: {last_err}")


# ==========================================
# LLM — limpeza de questão (multimodal)
# ==========================================

def llm_clean_question(
    client: OpenAI,
    statement_raw: str,
    figures_b64: Optional[List[str]] = None,
    tables_md: Optional[List[str]] = None,
    model: str = "gpt-4o",
    max_input_chars: int = 12_000,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Limpa e estrutura o enunciado de uma questão via LLM.

    Quando `figures_b64` é fornecido, a chamada é multimodal:
    o modelo recebe as imagens (via vision) e deve descrevê-las
    em 'sections' com type='figura'.

    Quando `tables_md` é fornecido, as tabelas são acrescentadas
    ao prompt de texto para que o LLM possa referenciá-las.
    """
    instructions = (
        "Você vai limpar e organizar o texto de UMA questão objetiva do ENADE.\n"
        "Regras:\n"
        "1) Não invente conteúdo — use apenas o texto e as imagens recebidas.\n"
        "2) Corrija quebras de linha, espaços e palavras quebradas pelo PDF.\n"
        "3) Separe TEXTO 1, TEXTO 2, tabelas e figuras em 'sections':\n"
        "   - Para CADA imagem recebida, crie uma section type='figura' com\n"
        "     descrição objetiva do conteúdo (gráfico, mapa, diagrama, foto…).\n"
        "   - Para CADA tabela em '--- TABELAS EXTRAÍDAS DO PDF ---', crie uma\n"
        "     section type='tabela' contendo o Markdown original.\n"
        "4) Em 'prompt', coloque apenas o enunciado/pergunta principal.\n"
        "5) Extraia as alternativas A–E em 'alternatives' quando existirem.\n"
        "6) Em 'statement_clean', devolva o texto completo e legível (sem figuras).\n"
    )

    text_in = statement_raw.strip()
    if len(text_in) > max_input_chars:
        text_in = text_in[:max_input_chars] + "\n\n[TRUNCADO]\n"

    user_content = _build_multimodal_content(
        text=text_in,
        figures_b64=figures_b64 or [],
        tables_md=tables_md,
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name":   QUESTION_JSON_SCHEMA["name"],
            "schema": QUESTION_JSON_SCHEMA["schema"],
            "strict": True,
        },
    }
    return _call_llm(
        client=client,
        model=model,
        system_prompt=instructions,
        user_content=user_content,
        response_format=response_format,
        retries=retries,
    )


# ==========================================
# LLM — classificação de subárea
# ==========================================

def llm_classify_subarea(
    client: OpenAI,
    question_text: str,
    subareas: List[str],
    model: str = "gpt-4o",
    retries: int = 2,
) -> Dict[str, Any]:
    allowed = ", ".join(subareas)
    instructions = (
        "Você vai classificar uma questão objetiva do ENADE em apenas UMA subárea.\n"
        f"Subáreas permitidas: {allowed}\n"
        "Regras:\n"
        "1) Escolha somente uma subárea da lista.\n"
        "2) Não invente nova subárea.\n"
        "3) Baseie-se no conteúdo principal da questão.\n"
        "4) Retorne a confiança entre 0 e 1.\n"
        "5) Se estiver em dúvida, escolha a mais provável entre as disponíveis.\n"
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name":   SUBAREA_SCHEMA["name"],
            "schema": SUBAREA_SCHEMA["schema"],
            "strict": True,
        },
    }
    try:
        result = _call_llm(
            client=client,
            model=model,
            system_prompt=instructions,
            user_content=question_text,
            response_format=response_format,
            retries=retries,
        )
        if result.get("subarea") not in subareas:
            log.warning(
                "Subárea '%s' fora da lista; marcando como Não classificada.",
                result.get("subarea"),
            )
            result["subarea"] = "Não classificada"
        return result
    except RuntimeError as e:
        return {"subarea": "Não classificada", "confidence": 0.0, "reason": str(e)}


# ==========================================
# Agrupamento e persistência
# ==========================================

def group_questions_for_output(
    questions: List[Question],
    subareas: List[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "formacao_geral": [],
        "componente_especifico": {sub: [] for sub in subareas},
        "nao_classificadas": [],
    }
    for q in questions:
        q_dict = asdict(q)
        if q.section_type == "Formação Geral":
            output["formacao_geral"].append(q_dict)
        elif q.section_type == "Componente Específico":
            if q.subarea and q.subarea in output["componente_especifico"]:
                output["componente_especifico"][q.subarea].append(q_dict)
            else:
                output["nao_classificadas"].append(q_dict)
    return output


def save_grouped_json(grouped_data: Dict[str, Any], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, ensure_ascii=False, indent=4)
    log.info("JSON salvo em: %s", output_file)


# ==========================================
# Main
# ==========================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extrai questões objetivas do ENADE (com tabelas e figuras) "
            "e divide em Formação Geral / Componente Específico."
        )
    )

    # --- Arquivos ---
    ap.add_argument("--pdf",  default=PDF_DEFAULT,  help="Caminho do PDF da prova.")
    ap.add_argument("--key",  default=None,          help="Caminho do PDF do gabarito.")
    ap.add_argument("--out",  default="questoes_objetivas_agrupadas.json", help="JSON de saída.")
    ap.add_argument("--exam", default="ENADE",        help="Nome do exame.")
    ap.add_argument("--year", type=int, default=2022, help="Ano do exame.")
    ap.add_argument("--area", default="Administração", help="Área/Curso.")

    # --- Extração de mídia ---
    ap.add_argument(
        "--extract-media",
        action="store_true",
        help="Ativa a extração de tabelas e figuras por questão.",
    )
    ap.add_argument(
        "--figures-dir",
        default="figures",
        help=(
            "Diretório onde os PNGs das figuras serão salvos "
            "(requer --extract-media e Pillow). Padrão: 'figures'."
        ),
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=150,
        help="DPI para renderização de páginas ao extrair figuras. Padrão: 150.",
    )
    ap.add_argument(
        "--min-figure-size",
        type=float,
        default=80.0,
        help=(
            "Tamanho mínimo em pontos PDF (largura E altura) para aceitar "
            "um objeto como figura válida. Padrão: 80."
        ),
    )

    # --- LLM ---
    ap.add_argument("--llm-clean",            action="store_true", help="Limpa questões com LLM.")
    ap.add_argument("--llm-classify-subareas", action="store_true", help="Classifica subáreas com LLM.")
    ap.add_argument(
        "--llm-model",
        default="gpt-4o",
        help="Modelo OpenAI (deve suportar visão quando há figuras). Padrão: gpt-4o.",
    )
    ap.add_argument("--llm-sleep", type=float, default=0.0, help="Pausa entre chamadas LLM (segundos).")
    ap.add_argument(
        "--subareas",
        default="",
        help='Subáreas separadas por vírgula. Ex.: "Finanças,Marketing,RH,Produção"',
    )

    # --- Debug ---
    ap.add_argument("--debug", action="store_true", help="Logging em nível DEBUG.")

    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    subareas = [s.strip() for s in args.subareas.split(",") if s.strip()] or ["Não classificada"]

    # --------------------------------------------------
    # Gabarito
    # --------------------------------------------------
    answer_key: Dict[int, str] = {}
    if args.key:
        log.info("Lendo GABARITO: %s", args.key)
        key_text = extract_pdf_text_with_page_markers(args.key)
        answer_key = parse_answer_key_from_text(key_text, min_q=1, max_q=200)
        log.info("Gabarito: %d resposta(s) detectada(s).", len(answer_key))

    # --------------------------------------------------
    # Extração de texto
    # --------------------------------------------------
    log.info("Lendo PROVA: %s", args.pdf)
    full_text = extract_pdf_text_with_page_markers(args.pdf)
    full_text = remove_questionario_percepcao(full_text)

    questions = split_into_questions(
        full_text=full_text,
        exam_name=args.exam,
        year=args.year,
        area=args.area,
        answer_key=answer_key,
        only_objective=True,
    )

    if not questions:
        log.error("Nenhuma questão objetiva encontrada.")
        return

    log.info("Questões objetivas extraídas: %d", len(questions))

    if args.extract_media and not PILLOW_AVAILABLE:
        log.warning(
            "Pillow não instalado — figuras serão ignoradas. "
            "Instale com: pip install Pillow"
        )

    # --------------------------------------------------
    # Cliente LLM
    # --------------------------------------------------
    client: Optional[OpenAI] = None
    if args.llm_clean or args.llm_classify_subareas:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("ERRO: defina OPENAI_API_KEY no ambiente.")
        client = OpenAI()

    # --------------------------------------------------
    # Loop de processamento
    # --------------------------------------------------
    for i, q in enumerate(questions, start=1):
        log.info(
            "Q%d (%d/%d) — %s | págs. %s–%s",
            q.q_number, i, len(questions),
            q.section_type, q.page_start, q.page_end,
        )

        tables_md: List[str] = []
        figures_b64: List[str] = []

        # --- Extração de tabelas e figuras ---
        if args.extract_media and q.page_start is not None and q.page_end is not None:
            tables_md, figure_paths, figures_b64 = extract_page_media(
                pdf_path=args.pdf,
                page_start=q.page_start,
                page_end=q.page_end,
                q_number=q.q_number,
                figures_dir=args.figures_dir if PILLOW_AVAILABLE else None,
                resolution=args.resolution,
                min_img_width=args.min_figure_size,
                min_img_height=args.min_figure_size,
            )
            q.tables  = tables_md    or None
            q.figures = figure_paths or None

            if tables_md:
                log.info("  → %d tabela(s)", len(tables_md))
            if figure_paths:
                log.info("  → %d figura(s) salva(s) em '%s'", len(figure_paths), args.figures_dir)
            elif figures_b64:
                log.info("  → %d figura(s) detectada(s) (sem salvar em disco)", len(figures_b64))

        # --- Limpeza com LLM (multimodal se houver figuras) ---
        if args.llm_clean and client is not None:
            try:
                cleaned = llm_clean_question(
                    client=client,
                    statement_raw=q.statement,
                    figures_b64=figures_b64,   # vision: envia imagens ao LLM
                    tables_md=tables_md,       # tabelas no prompt de texto
                    model=args.llm_model,
                )
                q.statement_clean  = cleaned.get("statement_clean")
                q.prompt           = cleaned.get("prompt")
                q.sections         = cleaned.get("sections")
                q.alternatives     = cleaned.get("alternatives")
                q.llm_clean_failed = False

                if q.statement_clean:
                    q.statement = q.statement_clean

            except Exception as e:
                log.error("Falha na limpeza da Q%d: %s", q.q_number, e)
                q.llm_clean_failed = True

        # --- Classificação de subárea ---
        if q.section_type == "Componente Específico":
            if (
                args.llm_classify_subareas
                and client is not None
                and subareas != ["Não classificada"]
            ):
                result = llm_classify_subarea(
                    client=client,
                    question_text=q.statement,
                    subareas=subareas,
                    model=args.llm_model,
                )
                q.subarea = result.get("subarea", "Não classificada")
                q.llm_classify_failed = (
                    q.subarea == "Não classificada"
                    and result.get("confidence", 1.0) == 0.0
                )
            else:
                q.subarea = "Não classificada"

        if args.llm_sleep > 0:
            time.sleep(args.llm_sleep)

    # --------------------------------------------------
    # Saída JSON
    # --------------------------------------------------
    grouped = group_questions_for_output(questions, subareas)
    save_grouped_json(grouped, args.out)

    n_fg = len(grouped["formacao_geral"])
    n_ce = sum(len(v) for v in grouped["componente_especifico"].values())
    n_nc = len(grouped["nao_classificadas"])
    log.info(
        "Concluído — Formação Geral: %d | Comp. Específico: %d | Não classificadas: %d",
        n_fg, n_ce, n_nc,
    )


if __name__ == "__main__":
    main()
