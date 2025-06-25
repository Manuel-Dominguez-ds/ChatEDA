from agent.state_types import AgentState

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

insights_grafico_system_prompt = '''Eres un experto en análisis de gráficos y visualización de datos.
Tu tarea es analizar gráficos generados y extraer insights relevantes sobre las relaciones entre las variables representadas.
Debes proporcionar un análisis claro y conciso, destacando patrones, tendencias y cualquier hallazgo importante.
Evitá explicaciones largas o genéricas. Sé directo, claro y orientado a negocio.
Si encuentras algún problema para analizar el gráfico, indícalo claramente.'''

insights_llm_prompt = """
        Teniendo en cuenta el dataset proporcionado, la estructura, el resumen, el proceso de limpieza y las visualizaciones, genera insights valiosos y accionables.
        
        INSTRUCCIONES:
        1. Analiza este dataset de forma integral y profesional
        2. Genera insights valiosos y accionables basados en los datos
        3. Identifica patrones, anomalías y oportunidades
        4. Sugiere próximos pasos para análisis o modelado
        5. Considera el tipo de problema detectado y las variables disponibles
        6. Proporciona recomendaciones específicas para mejorar el análisis
        
        Estructura tu respuesta de manera clara y organizada con secciones bien definidas.
        """

sugerir_graficos_prompt = f"""
Teniendo en cuenta el dataset proporcionado y el objetivo de análisis sugerí visualizaciones útiles para entender las relaciones entre variables y la distribución del target.
Basado en esto, sugerí entre 3 y 6 gráficos útiles para entender las relaciones importantes entre variables y la distribución del target. Para cada gráfico, devolvé un JSON con el siguiente formato:

{{
  "id": "grafico_1",
  "tipo": "scatterplot" | "boxplot" | "histograma" | "heatmap" | "barplot",
  "columnas": ["col1", "col2"],
  "descripcion": "Relación entre col1 y col2"
}}

Solo devolvé una lista JSON con estos objetos. Nada más. No uses etiquetas de código como "```json" o "```python".
"""

system_prompt_refinamiento= """
Sos un asistente experto en Python. Vas a recibir una función ya generada que puede tener errores como:
- strings sin cerrar
- llamadas comentadas
- falta de ejecución de la función

Tu tarea es corregir SOLO esa función y su posible llamada al final. No agregues ejemplos, ni declares DataFrames ni imports innecesarios. No incluyas explicaciones. Devolvé SOLO el código corregido y ejecutable.
Si el código no tiene errores de sintaxis, devolvé el código tal cual, sin cambios.
"""

user_prompt_validacion_problema_negocio = f"""
Teniendo en cuenta los insights obtenidos del dataset y los requerimientos del usuario, por favor analiza el problema de negocio que podría resolverse con Machine Learning.

Con base en esta información, responde:
1. ¿Cuál parece ser el problema de negocio más relevante?
2. ¿Es útil aplicar Machine Learning? ¿Por qué?
3. Si es útil, ¿qué tipo de modelo (clasificación, regresión, clustering) se debería usar?
4. ¿Qué variable parece ser la más adecuada como objetivo (target)?

Responde ÚNICAMENTE con un JSON válido usando exactamente estos campos:

{{
  "problema_negocio": "descripción clara del problema que se quiere resolver",
  "usar_ml": true,
  "justificacion_ml": "explicación breve de por qué es útil aplicar Machine Learning en este caso",
  "tipo_modelo": "regresion",
  "variable_target": "nombre_exacto_de_la_columna",
  "comentarios_adicionales": "cualquier otro dato relevante para el modelado"
}}

IMPORTANTE: Responde SOLO con el JSON, sin texto adicional, sin markdown, sin explicaciones extra.
"""
# ------------- FUNCIONES DE PROMPTS ------------- #

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

def generar_codigo_grafico_prompt(grafico_info, contexto):
    prompt =f"""
Generá código Python usando matplotlib o seaborn para construir un gráfico de tipo {grafico_info["tipo"]}, 
que analice las columnas: {', '.join(grafico_info["columnas"])}.

Descripción del gráfico: {grafico_info["descripcion"]}

Contexto del análisis: {contexto}

No expliques nada, solo devolvé el código limpio en Python, listo para ejecutarse.
Usá como variable de entrada un DataFrame llamado `df`.
"""
    return prompt

def construir_prompt_refinamiento(error, original_code):
    prompt =f"""
            Este es el código generado para crear un gráfico. Al ejecutarlo, se produjo el siguiente error:

            >>> {error['error']}

            Revisá el código y corregí el problema. Recordá:
            - No agregues DataFrames ni imports nuevos.
            - No reescribas el código completo si no es necesario.
            - Solo corregí lo justo y necesario para que funcione.

            Código original:
            {original_code}
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