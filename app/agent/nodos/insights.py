from agent.state_types import AgentState
import PIL
import google.generativeai as genai
from agent.state import api_key_multimodal, llm
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.utils.prompts import insights_grafico_system_prompt, insights_llm_prompt, construir_input_llm
import json


def insights_graficos_llm(state : AgentState) -> AgentState:
    """
    Genera insights a partir de los gráficos generados, utilizando LLM para interpretar los resultados.
    """
    print("\n" + "="*80)
    print("🔍 NODO: insights_graficos_llm")
    print("="*80)

    try:
        if not state.get('graficos'):
            raise ValueError("No hay gráficos generados para analizar.")
        genai.configure(api_key=api_key_multimodal)

        model = genai.GenerativeModel("gemini-2.0-flash")
        
        model_system_prompt = insights_grafico_system_prompt
        
        insights_graficos = []
        for graf in state['graficos']:
            graf_id = graf['id']
            ruta = graf.get('ruta', None)
            if not ruta or not os.path.exists(ruta):
                raise ValueError(f"Gráfico {graf_id} no tiene ruta válida o no fue generado correctamente.")
            
            prompt = f"""{state['messages'][1].content}.\n\nAnalizá el gráfico generado en {ruta} y extraé insights relevantes sobre la relación entre las variables representadas. Si tenes algun problema para analizar el gráfico, indicá que no se puede analizar."""
            img = PIL.Image.open(ruta)
            
            response = model.generate_content([model_system_prompt,prompt,img])
            insights = response.text.strip() 
            insights_graficos.append({
                "id": graf_id,
                "ruta": ruta,
                "insights": insights
            })
            print(f"✅ Insights generados para el gráfico {graf_id}")
            message = AIMessage(
                content=f"Insights para el gráfico {graf_id}:\n{insights}",
                name="insights_graficos_llm"
            )
            state['messages'].append(message)
            print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        state['insights_graficos'] = insights_graficos
        return state
    except Exception as e:
        msg = f"❌ Error en insights_graficos_llm: {str(e)}"
        print(msg)
        state['errores'].append(msg)
        return state
            
            
def insights_llm(state: AgentState) -> AgentState:
    """
    Genera insights del dataset utilizando el LLM
    """
    print("\n" + "="*80)
    print("💡 GENERANDO INSIGHTS CON LLM (nodo: insights_llm)")
    print("="*80)
    
    try:
        df = state.get('df')
        historial_limpieza = state.get('historial_limpieza', [])
        
        print(f"📊 Dataset final para insights: {df.shape}")
        print(f"🧹 Pasos de limpieza aplicados: {len(historial_limpieza)}")
        
        # Construir el input estructurado para el LLM
        input_llm = construir_input_llm(state)
        
        # Generar estadísticas adicionales para insights
        print("📈 Generando estadísticas adicionales...")
        
        # Análisis de correlaciones (solo para variables numéricas)
        correlaciones = {}
        numericas = df.select_dtypes(include=['number'])
        if len(numericas.columns) > 1:
            corr_matrix = numericas.corr()
            correlaciones_fuertes = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        correlaciones_fuertes.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlacion': round(corr_val, 3)
                        })
            correlaciones['fuertes'] = correlaciones_fuertes
            print(f"🔗 Correlaciones fuertes encontradas: {len(correlaciones_fuertes)}")
        
        # Análisis de distribución de variables categóricas
        analisis_categoricas = {}
        categoricas = df.select_dtypes(include=['object', 'category'])
        for col in categoricas.columns:
            value_counts = df[col].value_counts()
            analisis_categoricas[col] = {
                'categorias_unicas': len(value_counts),
                'distribucion_top5': value_counts.head().to_dict(),
                'concentracion': round(value_counts.iloc[0] / len(df) * 100, 2) if len(value_counts) > 0 else 0
            }
        
        print(f"🏷️  Variables categóricas analizadas: {len(categoricas.columns)}")
        
        # Análisis de calidad final de datos
        calidad_final = {
            'completitud': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            'filas_completas': len(df.dropna()),
            'duplicados_restantes': df.duplicated().sum(),
        }
        
        print(f"✅ Completitud final: {calidad_final['completitud']}%")
        print(f"📋 Filas completas: {calidad_final['filas_completas']:,}")
        
        print("🤖 Consultando al LLM para generar insights...")
        
        message_correlaciones = AIMessage(
            content=f"Análisis de correlaciones:\n{json.dumps(correlaciones, indent=2, ensure_ascii=False, default=str)}",
            name="analisis_correlaciones")
        
        message_calidad_final = AIMessage(
            content=f"Calidad final de datos:\n{json.dumps(calidad_final, indent=2, ensure_ascii=False, default=str)}",
            name="calidad_final_datos")
        
        state['messages'].append(message_correlaciones)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        state['messages'].append(message_calidad_final)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        # Crear prompt para insights
        insights_prompt = insights_llm_prompt
        state['messages'].append(HumanMessage(content=insights_prompt))
        print("✅ Mensaje de Humano registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        # Invocar al LLM para generar insights
        insights_response = llm.invoke(state['messages'])
        
        insights_generados = insights_response.content
        
        print(f"💬 Insights generados: {len(insights_generados)} caracteres")

        visualizaciones_texto = "\n".join([
            f"### 📊 Gráfico {item['id']}\n"
            f"- **Ruta**: {item['ruta']}\n"
            f"- **Insight**: {item['insights']}"
            for item in state.get("insights_graficos", [])
        ]) or "No se generaron visualizaciones."

        
        # Crear insights finales estructurados
        insights_finales = f"""
# 📊 ANÁLISIS COMPLETO DEL DATASET

## 📈 MÉTRICAS CLAVE
- **Filas procesadas**: {len(df):,}
- **Columnas analizadas**: {len(df.columns)}
- **Completitud de datos**: {calidad_final['completitud']}%
- **Variables numéricas**: {len(numericas.columns)}
- **Variables categóricas**: {len(categoricas.columns)}
- **Pasos de limpieza aplicados**: {len(historial_limpieza)}


## VISUALIZACIONES GENERADAS
{visualizaciones_texto}
---

{insights_generados}

---

## 🛠️ PROCESO DE LIMPIEZA APLICADO:
{chr(10).join([f"✅ **Paso {item['paso']}**: {item['decision'].get('action')} - {item['decision'].get('message')}" 
               for item in historial_limpieza if item.get('decision', {}).get('action')])}
        """
        
        state['insights'] = insights_finales
        
        print("✅ Insights generados y guardados en el estado")
        print("📝 Resumen de insights:")
        print(f"   • Pasos de limpieza documentados: {len(historial_limpieza)}")
        print(f"   • Correlaciones detectadas: {len(correlaciones.get('fuertes', []))}")
        print(f"   • Variables categóricas analizadas: {len(analisis_categoricas)}")
        
        state['messages'].append(AIMessage(
            content=insights_finales,
            name="insights_llm"
        ))
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
    except Exception as e:
        error_msg = f"Error generando insights: {str(e)}"
        print(f"❌ {error_msg}")
        state['errores'].append(error_msg)
        state['insights'] = f"❌ Error al generar insights: {error_msg}"
    
    return state