from agent.state import AgentState


SYSTEM_PROMPT = '''
Sos un Data Scientist senior especializado en resolver problemas de negocio reales para empresas pequeñas y medianas, incluso cuando no tienen expertos en datos.

Tu objetivo es transformar datasets crudos en valor, siguiendo un flujo completo de análisis y modelado. En cada paso vas a recibir información estructurada y actualizada sobre el dataset y el contexto del negocio. Tu comportamiento debe adaptarse según el tipo de análisis que se espera realizar.

Debés asistir en tareas como:

1. **Preprocesamiento de datos**  
   - Detectar problemas de calidad (nulos, duplicados, valores atípicos, tipos incorrectos, etc.).  
   - Elegir la mejor herramienta de limpieza disponible, de la lista proporcionada.  
   - Actuar de forma gradual, justificando tus elecciones.

2. **Exploración de datos e insights**  
   - Interpretar estadísticas descriptivas, estructuras, y visualizaciones.  
   - Detectar patrones, tendencias, relaciones y anomalías.  
   - Generar insights accionables (marketing, ventas, operaciones, etc.).

3. **Visualizaciones**  
   - Sugerir gráficos útiles según el contexto del negocio.  
   - Generar descripciones claras o código Python para representar los datos visualmente.

4. **Modelado predictivo**  
   - Sugerir si un problema es de clasificación o regresión.  
   - Preparar el dataset para modelado (encoding, imputación, etc.).  
   - Elegir variables objetivo y útiles.  
   - Proponer técnicas AutoML cuando sea relevante.

5. **Reportes finales**  
   - Comunicar hallazgos, gráficos y modelos de forma clara y útil para un usuario no técnico.  
   - Enfocarte en explicar “qué significa” cada resultado para el negocio.

Recibís siempre:
- La estructura del dataset (forma, tipos, nulos).
- Estadísticas numéricas y categóricas.
- El historial de pasos previos.
- Visualizaciones, insights, decisiones anteriores, y objetivos del negocio.

Respondé de forma clara, concreta y accionable. No repitas el input. Usá lenguaje profesional pero accesible. Pensá como un analista que entrega valor y toma decisiones inteligentes en cada etapa.
'''

def dataset_info_prompt(input_llm, historial_limpieza, textual_description_of_tools):
    prompt = f"""
        INFORMACIÓN DEL DATASET:
        {input_llm}
        
        HISTORIAL DE LIMPIEZA APLICADA:
        {[decision['decision'] for decision in historial_limpieza]}
        
        AVAILABLE TOOLS:
        {textual_description_of_tools}
        
        Analiza esta información y decide si el dataset necesita limpieza.
        Si necesita limpieza, selecciona UNA tool específica para aplicar. Solo podes elegir una tool de la lista proporcionada.
        Si consideras que el dataset ya está limpio y listo para análisis, indica 'no_limpieza_necesaria'.
        Si no encuentras la tool adecuada, indica 'generar_tool' con la descripción de lo que necesitas.
        
        Responde ÚNICAMENTE en formato JSON:
        {{
            "action": "nombre_de_tool_o_no_limpieza_necesaria_o_generar_tool",
            "params": {{"param1": "value1"}},
            "message": "Descripción de la acción"
        }}
        """
    return prompt


# ----------- CONSTRUCCIÓN INPUT ----------- #
def construir_input_llm(state: AgentState) -> dict:
    """
    Construye el input estructurado para el LLM con toda la información relevante
    """
    estructura = state.get("estructura", {})
    resumen = state.get("resumen", {})
    historial = state.get("historial_limpieza", [])
    
    # Información de calidad de datos
    nulls_info = estructura.get("nulls", {})
    total_nulls = sum(nulls_info.values())
    columnas_con_nulls = {k: v for k, v in nulls_info.items() if v > 0}
    
    # Análisis de tipos de datos
    tipos = estructura.get("tipos", {})
    tipos_problematicos = []
    for col, tipo in tipos.items():
        if str(tipo) == 'object' and col in resumen.get('categoricas', {}):
            if resumen['categoricas'][col].get('nunique', 0) > 50:
                tipos_problematicos.append(f"{col}: posible texto libre (demasiadas categorías únicas)")
    
    input_llm = {
        "estructura_dataset": {
            "filas": estructura.get("cant_filas"),
            "columnas": estructura.get("cant_columnas"),
            "tipos_por_columna": {k: str(v) for k, v in tipos.items()},
            "calidad_datos": {
                "total_valores_nulos": total_nulls,
                "columnas_con_nulos": columnas_con_nulls,
                "porcentaje_nulos_global": round((total_nulls / (estructura.get("cant_filas", 1) * estructura.get("cant_columnas", 1))) * 100, 2)
            }
        },
        "resumen_estadistico": {
            "variables_numericas": resumen.get("numericas", {}),
            "variables_categoricas": resumen.get("categoricas", {}),
            "posibles_problemas": tipos_problematicos
        },
        "historial_limpieza_aplicada": [
            {
                "paso": item.get("paso"),
                "accion": item.get("decision", {}).get("action"),
                "parametros": item.get("decision", {}).get("params"),
                "descripcion": item.get("decision", {}).get("message")
            }
            for item in historial
        ]
    }
    
    return input_llm