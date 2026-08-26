import os
import time

import pandas as pd
import streamlit as st

from databricks.sdk.service.sql import StatementState

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "5804cb6f6345868a")

# Tipos numéricos y de fecha según la API de Databricks Statement Execution
_NUMERIC_TYPES = {"INT", "INTEGER", "LONG", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "SHORT", "BYTE"}
_DATE_TYPES = {"DATE"}
_TIMESTAMP_TYPES = {"TIMESTAMP", "TIMESTAMP_NTZ"}

_TERMINAL_STATES = {
    StatementState.SUCCEEDED,
    StatementState.FAILED,
    StatementState.CANCELED,
    StatementState.CLOSED,
}


@st.cache_resource
def _get_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def ejecutar_sql(sql: str) -> pd.DataFrame:
    """
    Ejecuta una sentencia SQL en el SQL Warehouse configurado y devuelve un DataFrame.

    - Cliente perezoso: se instancia solo cuando se llama por primera vez.
    - Polling: espera hasta que la consulta termine (estados PENDING/RUNNING).
    - Paginación: descarga todos los chunks del resultado.
    - Casteo de tipos: convierte columnas numéricas y de fecha según el esquema.
    - Mensajes de error claros cuando la tabla no existe o faltan permisos.
    """
    w = _get_client()

    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="50s",
    )

    # Polling hasta estado terminal (máximo ~2 minutos adicionales)
    stmt_id = response.statement_id
    max_polls = 60
    for _ in range(max_polls):
        if response.status and response.status.state in _TERMINAL_STATES:
            break
        time.sleep(2)
        response = w.statement_execution.get_statement(statement_id=stmt_id)
    else:
        # El bucle terminó sin alcanzar un estado terminal
        raise TimeoutError(
            f"La consulta '{stmt_id}' no finalizó tras {max_polls * 2}s adicionales. "
            "Considera simplificar la consulta o aumentar el warehouse."
        )

    if response.status.state != StatementState.SUCCEEDED:
        error_msg = ""
        if response.status and response.status.error:
            error_msg = response.status.error.message or str(response.status.error)
        raise RuntimeError(
            f"La consulta terminó con estado {response.status.state}. "
            f"Detalle: {error_msg or 'sin mensaje de error'}"
        )

    # Schema de columnas
    schema_cols = response.manifest.schema.columns if response.manifest and response.manifest.schema else []
    cols = [c.name for c in schema_cols]
    type_names = [str(c.type_name).upper() if c.type_name else "" for c in schema_cols]

    # Recolectar todos los chunks
    all_rows: list = []
    chunk = response.result
    if chunk and chunk.data_array:
        all_rows.extend(chunk.data_array)

    next_index = chunk.next_chunk_index if chunk else None
    while next_index is not None:
        chunk = w.statement_execution.get_statement_result_chunk_n(
            statement_id=stmt_id,
            chunk_index=next_index,
        )
        if chunk and chunk.data_array:
            all_rows.extend(chunk.data_array)
        next_index = chunk.next_chunk_index if chunk else None

    df = pd.DataFrame(all_rows, columns=cols) if all_rows else pd.DataFrame(columns=cols)

    # Casteo de tipos
    for col, tname in zip(cols, type_names):
        base = tname.split("(")[0].strip()
        if base in _NUMERIC_TYPES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif base in _DATE_TYPES:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        elif base in _TIMESTAMP_TYPES:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalizar nombres de columnas a minúsculas para evitar problemas de case
    df.columns = [c.lower() for c in df.columns]

    return df


# Alias para compatibilidad con imports existentes
query = ejecutar_sql