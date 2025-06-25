from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
import pandas as pd

class AgentState(TypedDict, total=False):
    archivo_input: str
    df: pd.DataFrame
    estructura: dict
    resumen: dict
    insights: str
    limpieza: str
    historial_limpieza: list
    visualizaciones: list
    graficos_generados: list
    graficos: list
    insights_graficos: list
    errores_graficos: list
    refinamiento_intentos: int
    estrategia_negocio: dict
    decision_usuario: str
    modelo_autogl: Any
    leaderboard: dict
    metricas_modelo: dict
    predicciones: pd.DataFrame
    modelo_interpretacion: dict
    reporte_ml: str
    reporte_final: str
    errores: list
    messages: Annotated[list[AnyMessage], add_messages]
