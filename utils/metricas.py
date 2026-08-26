import pandas as pd

def calcular_nps(df):

    promotores = (df["nps_categoria"]=="Promotor").sum()
    detractores = (df["nps_categoria"]=="Detractor").sum()

    total = len(df)

    if total == 0:
        return 0

    return round(
        (
            (promotores/total)
            -
            (detractores/total)
        ) * 100,
        1
    )


def calcular_csat(df):

    total = len(df)

    if total == 0:
        return 0

    satisfechos = (
        df["csat_categoria"]=="Satisfecho"
    ).sum()

    return round(
        (satisfechos/total)*100,
        1
    )