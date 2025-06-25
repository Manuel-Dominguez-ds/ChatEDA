from agent.state_types import AgentState

def route_analisis_limpieza(state: AgentState):
    """
    Ruta principal del agente que decide si analizar limpieza o aplicar tool
    """
    print("\n" + "="*80)
    print("🚦 DECISIÓN DE RUTA (nodo: route_analisis_limpieza)")
    print("="*80)
    
    df = state.get('df')
    if df is None or df.empty:
        error_msg = "No se ha cargado un DataFrame válido."
        print(f"❌ {error_msg}")
        state['errores'].append(error_msg)
        return state
    
    historial = state.get('historial_limpieza', [])
    
    ultima_accion = historial[-1]['decision']['action']
    print(f"🔍 Última acción del LLM: {ultima_accion}")
    
    if ultima_accion == 'no_limpieza_necesaria':
        print("✅ Decisión: El dataset está limpio, continuar a insights")
        return "No hace falta limpieza"
    else:
        print(f"🔧 Decisión: Aplicar tool de limpieza '{ultima_accion}'")
        return "Tool limpieza"
    
def routing_graficos(state: AgentState) -> str:
    """
    Verifica si faltan gráficos por generar.
    Devuelve:
        - 'pendientes' si aún hay gráficos sin código
        - 'completo' si todos los códigos fueron generados
    """
    total_sugeridos = len(state['visualizaciones'])
    total_generados = len(state['graficos'])

    print("\n" + "="*80)
    print("🔀 NODO CONDICIONAL: routing_graficos")
    print(f"📊 Visualizaciones sugeridas: {total_sugeridos}")
    print(f"✅ Gráficos con código generado: {total_generados}")
    print("="*80)

    if total_generados < total_sugeridos:
        print("➡️ Faltan gráficos por generar: volver a generar_codigo_grafico_llm")
        return "pendientes"
    else:
        print("✅ Todos los gráficos fueron generados")
        return "completo"
    
def route_grafico_error(state: AgentState) -> str:
    print("\n" + "="*80)
    print("🚦 NODO CONDICIONAL: route_grafico_error")
    print("="*80)
    
    errores = state.get("errores_graficos", [])
    intentos_refinamiento = state.get("refinamiento_intentos", 0)
    max_intentos = 1  # Máximo 2 intentos de refinamiento
    
    print(f"❌ Errores de gráficos: {len(errores)}")
    print(f"🔄 Intentos de refinamiento: {intentos_refinamiento}/{max_intentos}")
    
    if len(errores) == 0:
        print("✅ No hay errores de gráficos")
        # Resetear contador cuando no hay errores
        state["refinamiento_intentos"] = 0
        return "continuar"
    elif intentos_refinamiento < max_intentos:
        print(f"🛠️ Intentando refinamiento (intento {intentos_refinamiento + 1}/{max_intentos})")
        return "refinar"
    else:
        print(f"⚠️ Máximo de intentos de refinamiento alcanzado ({max_intentos}). Continuando con gráficos disponibles...")
        # Limpiar errores para evitar loops infinitos
        state["errores_graficos"] = []
        state["refinamiento_intentos"] = 0
        return "continuar"
    
def router_decision_usuario(state: AgentState) -> str:
    print("\n" + "="*80)
    print("🔄 RUTEO DE DECISIÓN DEL USUARIO (nodo: router_decision_usuario)")
    print("="*80)

    decision = state.get("decision_usuario", "").strip().lower()
    
    if decision == "continuar":
        print("➡️ Enrutando a modelado automático...")
        return "modelado_automatico"
    
    elif decision == "finalizar":
        print("➡️ Enrutando a finalización del flujo...")
        return 'fin'
    
    elif decision == "reformular":
        print("➡️ Enrutando a reformulación del problema...")
        return "reformular_problema_negocio"