from databricks.sdk import WorkspaceClient
import pandas as pd

WAREHOUSE_ID = "5804cb6f6345868a"

w = WorkspaceClient()

def ejecutar_sql(query):

    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query,
        wait_timeout="30s"
    )

    if response.status:
        if str(response.status.state) != "StatementState.SUCCEEDED":
            raise Exception(response.status)

    cols = [
        col.name
        for col in response.manifest.schema.columns
    ]

    rows = response.result.data_array

    return pd.DataFrame(
        rows,
        columns=cols
    )