#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db_manager.py
=============
Abstração de banco de dados para o ENADE Extractor.

Backends suportados:
  - Supabase  → PostgreSQL gerenciado via REST API (recomendado para produção)
  - SQLite    → fallback local / desenvolvimento

Variáveis de ambiente (opcionais):
  SUPABASE_URL   URL do projeto  (ex: https://xyz.supabase.co)
  SUPABASE_KEY   Chave anon ou service_role
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# DDL                                                                 #
# ------------------------------------------------------------------ #

# Execute este SQL no Supabase → SQL Editor antes de usar o app
DDL_SUPABASE = """
-- Execute no Supabase → SQL Editor
CREATE TABLE IF NOT EXISTS enade_questions (
    id                  BIGSERIAL PRIMARY KEY,
    exam                TEXT        NOT NULL,
    year                INTEGER     NOT NULL,
    area                TEXT        NOT NULL,
    q_number            INTEGER     NOT NULL,
    q_type              TEXT,
    section_type        TEXT,
    subarea             TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    statement           TEXT,
    answer              TEXT,
    statement_clean     TEXT,
    prompt              TEXT,
    sections            JSONB,
    alternatives        JSONB,
    tables              JSONB,
    figures             JSONB,
    llm_clean_failed    BOOLEAN     DEFAULT FALSE,
    llm_classify_failed BOOLEAN     DEFAULT FALSE,
    inserted_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (exam, year, area, q_number)
);

-- Índices úteis para consultas
CREATE INDEX IF NOT EXISTS idx_eq_year_area   ON enade_questions (year, area);
CREATE INDEX IF NOT EXISTS idx_eq_section     ON enade_questions (section_type);
CREATE INDEX IF NOT EXISTS idx_eq_subarea     ON enade_questions (subarea);
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS enade_questions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    exam                TEXT    NOT NULL,
    year                INTEGER NOT NULL,
    area                TEXT    NOT NULL,
    q_number            INTEGER NOT NULL,
    q_type              TEXT,
    section_type        TEXT,
    subarea             TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    statement           TEXT,
    answer              TEXT,
    statement_clean     TEXT,
    prompt              TEXT,
    sections            TEXT,
    alternatives        TEXT,
    tables              TEXT,
    figures             TEXT,
    llm_clean_failed    INTEGER DEFAULT 0,
    llm_classify_failed INTEGER DEFAULT 0,
    inserted_at         TEXT,
    UNIQUE (exam, year, area, q_number)
);
"""

# ------------------------------------------------------------------ #
# Serialização                                                        #
# ------------------------------------------------------------------ #

def _prepare_row(q: Dict[str, Any], json_as_str: bool = False) -> Dict[str, Any]:
    """
    Normaliza um dict de questão para inserção.
    json_as_str=True  → SQLite (campos JSON viram string)
    json_as_str=False → Supabase/Postgres (campos JSON ficam como dict/list)
    """
    row = {k: v for k, v in q.items() if k != "id"}

    # Campos JSON
    json_fields = ("sections", "alternatives", "tables", "figures")
    for f in json_fields:
        val = row.get(f)
        if val is None:
            row[f] = None
        elif json_as_str and not isinstance(val, str):
            row[f] = json.dumps(val, ensure_ascii=False)
        elif not json_as_str and isinstance(val, str):
            try:
                row[f] = json.loads(val)
            except Exception:
                row[f] = val

    # Campos BOOLEAN — None causaria erro no Postgres (NOT NULL DEFAULT FALSE)
    for f in ("llm_clean_failed", "llm_classify_failed"):
        if row.get(f) is None:
            row[f] = False

    # inserted_at: deixa o Supabase usar DEFAULT NOW(); para SQLite, define manualmente
    if json_as_str:
        row.setdefault("inserted_at", datetime.now(timezone.utc).isoformat())
    else:
        row.pop("inserted_at", None)   # Supabase: deixa o DEFAULT agir

    return row


# ================================================================== #
# Supabase Backend                                                    #
# ================================================================== #

class SupabaseBackend:
    """
    Usa a PostgREST REST API do Supabase.
    Não depende da lib supabase-py — apenas de httpx.
    """

    TABLE = "enade_questions"

    def __init__(self, url: str, key: str):
        try:
            import httpx
            self._http = httpx
        except ImportError:
            raise ImportError("Instale httpx: pip install httpx")

        self.base    = url.rstrip("/")
        self.key     = key
        self._hdrs   = {
            "apikey":        key,
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }

    def _url(self) -> str:
        return f"{self.base}/rest/v1/{self.TABLE}"

    # ---- public API ----

    def test_connection(self) -> Tuple[bool, str]:
        """Retorna (ok, mensagem)."""
        try:
            r = self._http.get(
                self._url(),
                headers={**self._hdrs, "Prefer": "count=exact"},
                params={"select": "id", "limit": "1"},
                timeout=10,
            )
            r.raise_for_status()
            total = int(r.headers.get("content-range", "*/0").split("/")[-1])
            return True, f"Conectado — {total} questão(ões) no banco"
        except Exception as e:
            return False, str(e)

    def count_questions(self, year: Optional[int] = None, area: Optional[str] = None) -> int:
        params: Dict[str, str] = {"select": "id"}
        if year:
            params["year"] = f"eq.{year}"
        if area:
            params["area"] = f"eq.{area}"
        try:
            r = self._http.get(
                self._url(),
                headers={**self._hdrs, "Prefer": "count=exact"},
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            return int(r.headers.get("content-range", "*/0").split("/")[-1])
        except Exception:
            return -1

    def upsert_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        rows = [_prepare_row(q, json_as_str=False) for q in questions]

        # Tentativa em lote primeiro
        try:
            r = self._http.post(
                self._url(),
                headers={
                    **self._hdrs,
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                content=json.dumps(rows, ensure_ascii=False, default=str),
                timeout=60,
            )
            if r.is_success:
                return {"inserted": len(rows), "errors": 0}
            # Supabase retornou erro HTTP — loga o body para diagnóstico
            log.error(
                "Supabase batch upsert HTTP %d: %s",
                r.status_code, r.text[:500],
            )
        except Exception as e:
            log.error("Supabase batch upsert exception: %s", e)

        # Fallback: um a um para identificar qual questão falha
        ok = err = 0
        for row in rows:
            try:
                r = self._http.post(
                    self._url(),
                    headers={
                        **self._hdrs,
                        "Prefer": "resolution=merge-duplicates,return=minimal",
                    },
                    content=json.dumps([row], ensure_ascii=False, default=str),
                    timeout=30,
                )
                if r.is_success:
                    ok += 1
                else:
                    log.error(
                        "Supabase Q%s HTTP %d: %s",
                        row.get("q_number"), r.status_code, r.text[:300],
                    )
                    err += 1
            except Exception as e2:
                log.error("Supabase Q%s exception: %s", row.get("q_number"), e2)
                err += 1
        return {"inserted": ok, "errors": err}

    def fetch_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            r = self._http.get(
                self._url(),
                headers=self._hdrs,
                params={"order": "id.desc", "limit": str(limit)},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return []

    def fetch_stats(self) -> Dict[str, Any]:
        """Retorna contagens agrupadas por section_type."""
        stats: Dict[str, Any] = {}
        for section in ("Formação Geral", "Componente Específico"):
            try:
                r = self._http.get(
                    self._url(),
                    headers={**self._hdrs, "Prefer": "count=exact"},
                    params={"select": "id", "section_type": f"eq.{section}", "limit": "1"},
                    timeout=10,
                )
                r.raise_for_status()
                stats[section] = int(r.headers.get("content-range", "*/0").split("/")[-1])
            except Exception:
                stats[section] = -1
        return stats

    @property
    def ddl(self) -> str:
        return DDL_SUPABASE


# ================================================================== #
# SQLite Backend (desenvolvimento / fallback)                         #
# ================================================================== #

class SQLiteBackend:
    TABLE = "enade_questions"

    def __init__(self, path: str = "enade_dev.db"):
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute(DDL_SQLITE)
            conn.commit()
        log.info("SQLite inicializado: %s", self.path)

    def test_connection(self) -> Tuple[bool, str]:
        try:
            n = self.count_questions()
            return True, f"SQLite OK — {n} questão(ões)"
        except Exception as e:
            return False, str(e)

    def count_questions(self, year: Optional[int] = None, area: Optional[str] = None) -> int:
        sql  = "SELECT COUNT(*) FROM enade_questions"
        cond = []
        vals = []
        if year:
            cond.append("year=?"); vals.append(year)
        if area:
            cond.append("area=?"); vals.append(area)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        with sqlite3.connect(self.path) as conn:
            return conn.execute(sql, vals).fetchone()[0]

    def upsert_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        ok = err = 0
        with sqlite3.connect(self.path) as conn:
            for q in questions:
                row  = _prepare_row(q, json_as_str=True)
                cols = list(row.keys())
                ph   = ",".join(["?"] * len(cols))
                upd  = ",".join(
                    f"{c}=excluded.{c}"
                    for c in cols
                    if c not in ("exam", "year", "area", "q_number")
                )
                sql = (
                    f"INSERT INTO enade_questions ({','.join(cols)}) VALUES ({ph}) "
                    f"ON CONFLICT(exam,year,area,q_number) DO UPDATE SET {upd}"
                )
                try:
                    conn.execute(sql, list(row.values()))
                    ok += 1
                except Exception as e:
                    log.error("SQLite Q%s: %s", q.get("q_number"), e)
                    err += 1
            conn.commit()
        return {"inserted": ok, "errors": err}

    def fetch_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM enade_questions ORDER BY id DESC LIMIT {limit}"
            ).fetchall()
            return [dict(r) for r in rows]

    def fetch_stats(self) -> Dict[str, Any]:
        stats = {}
        with sqlite3.connect(self.path) as conn:
            for section in ("Formação Geral", "Componente Específico"):
                n = conn.execute(
                    "SELECT COUNT(*) FROM enade_questions WHERE section_type=?", (section,)
                ).fetchone()[0]
                stats[section] = n
        return stats

    @property
    def ddl(self) -> str:
        return DDL_SQLITE


# ================================================================== #
# Factory                                                             #
# ================================================================== #

def make_backend(
    backend_type: str,
    supabase_url: str = "",
    supabase_key: str = "",
    sqlite_path:  str = "enade_dev.db",
):
    if backend_type == "supabase":
        url = supabase_url or os.getenv("SUPABASE_URL", "")
        key = supabase_key or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise ValueError("Supabase URL e Key são obrigatórios.")
        return SupabaseBackend(url, key)
    return SQLiteBackend(sqlite_path)
