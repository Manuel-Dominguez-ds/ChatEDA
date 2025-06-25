from agent.nodos.load_data import *
from agent.nodos.verificar_estructura import *
from agent.nodos.resumen_estadistico import *
from agent.nodos.limpieza import *
from agent.nodos.visualizacion import *
from agent.nodos.insights import *
from agent.nodos.machine_learning import *
from agent.nodos.reporte import * 
from agent.nodos.routers import *
from langgraph.graph import START, END, StateGraph

graph = StateGraph(AgentState)

graph.add_node("load_data", load_data)
graph.add_node("verificar_estructura", verificar_estructura)
graph.add_node("resumen_estadistico", resumen_estadistico)
graph.add_node("analisis_limpieza", analisis_limpieza)
graph.add_node("aplicar_tool_limpieza", aplicar_tool_limpieza)
graph.add_node('sugerir_graficos_llm', sugerir_graficos_llm)
graph.add_node('generar_codigo_grafico_llm', generar_codigo_grafico_llm)
graph.add_node('ejecutar_graficos', ejecutar_graficos)
graph.add_node('refinar_codigo_grafico_llm', refinar_codigo_grafico_llm)
graph.add_node('insights_graficos_llm', insights_graficos_llm)
graph.add_node("insights_llm", insights_llm)
graph.add_node('reporte_final_llm', reporte_final_llm_mejorado)
graph.add_node("validar_problema_negocio_llm", validar_problema_negocio_llm)
graph.add_node("confirmar_estrategia_ml", confirmar_estrategia_ml)
graph.add_node("aplicar_automl", aplicar_automl)

graph.add_edge(START, "load_data")
graph.add_edge("load_data", "verificar_estructura")
graph.add_edge("verificar_estructura", "resumen_estadistico")
graph.add_edge("resumen_estadistico", "analisis_limpieza")
graph.add_conditional_edges(
    "analisis_limpieza",
    route_analisis_limpieza,
    {"No hace falta limpieza":"sugerir_graficos_llm",
     "Tool limpieza":"aplicar_tool_limpieza"})
graph.add_edge("aplicar_tool_limpieza", "analisis_limpieza")
graph.add_edge("sugerir_graficos_llm", "generar_codigo_grafico_llm")
graph.add_conditional_edges(
    "generar_codigo_grafico_llm",
    routing_graficos,
    {"pendientes": "generar_codigo_grafico_llm", 
     "completo": "ejecutar_graficos"}
)
graph.add_conditional_edges(
    'ejecutar_graficos',
    route_grafico_error,
    {"refinar": "refinar_codigo_grafico_llm", 
     "continuar": "insights_graficos_llm"}
)
graph.add_edge("refinar_codigo_grafico_llm", "ejecutar_graficos")
graph.add_edge("insights_graficos_llm", "insights_llm")
graph.add_edge("insights_llm", "validar_problema_negocio_llm")
graph.add_edge('validar_problema_negocio_llm', 'confirmar_estrategia_ml')
graph.add_conditional_edges(
    "confirmar_estrategia_ml",
    router_decision_usuario,
    {"modelado_automatico": "aplicar_automl", 
     "fin": 'reporte_final_llm', 
     "reformular_problema_negocio": "validar_problema_negocio_llm"}
)
graph.add_edge("aplicar_automl", "reporte_final_llm")
graph.add_edge("reporte_final_llm", END)


react_graph = graph.compile()
