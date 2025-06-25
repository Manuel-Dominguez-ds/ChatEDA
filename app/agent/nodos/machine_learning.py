import json
from agent.state_types import AgentState
from agent.state import llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import re
from agent.utils.prompts import user_prompt_validacion_problema_negocio
from autogluon.tabular import TabularPredictor
import shutil
from datetime import datetime
import os

def validar_problema_negocio_llm(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("🧠 VALIDANDO PROBLEMA DE NEGOCIO (nodo: validar_problema_negocio_llm)")
    print("="*80)

    # Recuperar el último mensaje de insights
    insights = None
    indice_insights_llm = 0
    print(len(state["messages"]))
    for i,msg in enumerate(state["messages"]):
        if isinstance(msg, AIMessage) and msg.name == "insights_llm":
            insights = msg.content
            indice_insights_llm = i
            print(f"🔍 Insights encontrados")
            break

    if not insights:
        state["errores"].append("No se encontraron insights para analizar el problema de negocio.")
        return state
    
    user_prompt = user_prompt_validacion_problema_negocio
    
    if state['decision_usuario'] != 'reformular':
        state["messages"].append(HumanMessage(content=user_prompt, name="validar_problema_negocio_llm"))
        print("✅ Mensaje de Humano registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
    try:
        response = llm.invoke([
            state['messages'][0],
            state['messages'][1],
            *state["messages"][indice_insights_llm:],
        ])
        
        print("✅ Respuesta generada por LLM:")
        print(response.content)

        # 🔧 PARSEO MEJORADO DE LA RESPUESTA
        response_content = response.content.strip()
        
        # Limpiar la respuesta de posibles marcadores de markdown
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
            print("🧹 Eliminando marcadores de markdown")
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
            print("🧹 Eliminando bloques de código")
        
        # Buscar el JSON válido usando regex como respaldo
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
            print(f"🔍 JSON extraído mediante regex")
        else:
            json_content = response_content
            print("⚠️ Usando respuesta completa como JSON")
        
        estrategia_dict = json.loads(json_content)
        
        state["estrategia_negocio"] = estrategia_dict  # ✅ Ahora es un dict
            
        print("✅ Estrategia de negocio parseada correctamente:")
        print(f"   • Problema: {estrategia_dict['problema_negocio']}")
        print(f"   • Usar ML: {estrategia_dict['usar_ml']}")
        print(f"   • Tipo modelo: {estrategia_dict['tipo_modelo']}")
        print(f"   • Variable target: {estrategia_dict['variable_target']}")
            
            # Agregar mensaje al historial con el JSON parseado
        state["messages"].append(AIMessage(
                content=json.dumps(estrategia_dict, indent=2, ensure_ascii=False), 
                name="validar_problema_negocio_llm"
            ))
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        state["decision_usuario"] = ""
        return state
            
    except Exception as e:
        error_msg = f"Error al validar problema de negocio: {str(e)}"
        print(f"❌ {error_msg}")
        state["errores"].append(error_msg)
        return state
    
    
def confirmar_estrategia_ml(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("📣 CONFIRMACIÓN DE ESTRATEGIA (nodo: confirmar_estrategia_ml)")
    print("="*80)

    estrategia = state.get("estrategia_negocio")
    if not estrategia:
        msg = "No hay estrategia de negocio para confirmar."
        print(f"❌ {msg}")
        state["errores"].append(msg)
        return state

    # Mostrar estrategia sugerida
    problema_negocio = estrategia['problema_negocio']
    usar_ml = estrategia['usar_ml']
    justificacion_ml= estrategia["justificacion_ml"]
    tipo_modelo = estrategia["tipo_modelo"]
    variable_target = estrategia["variable_target"]
    comentarios_adicionales = estrategia["comentarios_adicionales"]
    
    print("\n💡 Estrategia sugerida por la IA:\n")
    print(f"- Problema de negocio: {problema_negocio}")
    print(f"- ¿Usar ML?: {'Sí' if usar_ml else 'No'}")
    print(f"- Tipo de modelo sugerido: {tipo_modelo}")
    print(f"- Variable objetivo: {variable_target}")
    print(f"- Justificación: {justificacion_ml}")
    print(f"- Comentarios adicionales: {comentarios_adicionales}\n")


    # Bucle de validación de entrada del usuario
    decision = ""
    while decision not in ["continuar", "finalizar", "reformular"]:
        print("\n🧭 ¿Qué te gustaría hacer?")
        print("1. Continuar con el modelado automático (escribí: continuar)")
        print("2. Finalizar el análisis (escribí: finalizar)")
        print("3. Aportar más detalles sobre el problema (escribí: reformular)\n")

        #decision = input("👉 Escribí tu decisión: ").strip().lower() 
        decision = 'continuar'
        if decision not in ["continuar", "finalizar", "reformular"]:
            print("❌ Opción no válida. Por favor, escribí una de: continuar / finalizar / reformular.")

    # Procesar la decisión del usuario
    state["decision_usuario"] = decision

    if decision == "continuar":
        print("✅ Continuando con el flujo de Machine Learning.")
        return state

    elif decision == "finalizar":
        print("🛑 El análisis ha sido finalizado por decisión del usuario.")
        return state

    elif decision == "reformular":
        print("✍️ Por favor, ingresá más contexto sobre el problema de negocio:")
        detalles_extra = input("📝 Nuevos detalles: ").strip()
        state["messages"].append(HumanMessage(content=detalles_extra, name="contexto_extra_usuario"))
        print("✅ Mensaje de Humano registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        print("🔁 Volviendo a validar el problema con la nueva información provista...")
        return state

def aplicar_automl(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("🤖 MODELADO AUTOMÁTICO CON AUTOGLUON (nodo: aplicar_automl)")
    print("="*80)

    try:
        # Validaciones básicas
        df = state.get("df")
        if df is None or df.empty:
            msg = "DataFrame no disponible o vacío."
            print(f"❌ {msg}")
            state["errores"].append(msg)
            return state

        estrategia_negocio = state.get("estrategia_negocio")
        if not estrategia_negocio:
            msg = "No se encontró estrategia de negocio definida."
            print(f"❌ {msg}")
            state["errores"].append(msg)
            return state

        tipo_modelo =estrategia_negocio["tipo_modelo"]
        target = estrategia_negocio["variable_target"]

        if not tipo_modelo or not target:
            msg = "Faltan datos requeridos: tipo_modelo o variable_target."
            print(f"❌ {msg}")
            state["errores"].append(msg)
            return state

        if target not in df.columns:
            msg = f"La variable objetivo '{target}' no existe en el DataFrame."
            print(f"❌ {msg}")
            state["errores"].append(msg)
            return state

        print(f"📊 Dataset: {df.shape}")
        print(f"🎯 Variable objetivo: {target}")
        print(f"🔧 Tipo de modelo: {tipo_modelo}")

        # Análisis básico de la variable objetivo
        target_info = {
            "valores_unicos": df[target].nunique(),
            "valores_nulos": df[target].isnull().sum(),
            "tipo_datos": str(df[target].dtype)
        }
        
        print(f"🔍 Variable objetivo:")
        print(f"   • Valores únicos: {target_info['valores_unicos']}")
        print(f"   • Valores nulos: {target_info['valores_nulos']}")
        print(f"   • Tipo: {target_info['tipo_datos']}")

        # Mapeo de tipos de problema para AutoGluon
        problem_type_map = {
            "clasificacion": "binary" if target_info["valores_unicos"] == 2 else "multiclass",
            "regresion": "regression"
        }

        problem_type = problem_type_map.get(tipo_modelo)
        
        if problem_type is None:
            msg = f"Tipo de modelo '{tipo_modelo}' no soportado por AutoGluon."
            print(f"❌ {msg}")
            state["errores"].append(msg)
            return state

        print(f"🧠 Problema AutoGluon: {problem_type}")

        # Validaciones específicas por tipo de problema
        if problem_type in ["binary", "multiclass"] and target_info["valores_unicos"] > 100:
            msg = f"Demasiadas clases ({target_info['valores_unicos']}) para clasificación."
            print(f"⚠️ {msg}")
            state["errores"].append(msg)
            return state

        # Configuración de AutoGluon basada en el tamaño del dataset
        dataset_size = len(df)
        if dataset_size < 1000:
            time_limit = 300  # 5 minutos
            presets = 'medium_quality'
        elif dataset_size < 10000:
            time_limit = 600  # 10 minutos  
            presets = 'good_quality'
        else:
            time_limit = 1200  # 20 minutos
            presets = 'best_quality'
        time_limit = 120
        presets = 'medium_quality'
        print(f"⏱️ Tiempo límite: {time_limit // 60} minutos")
        print(f"🎯 Preset: {presets}")

        # Preparar directorio para el modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"modelo_autogl_{timestamp}"
        
        # Limpiar modelo anterior si existe
        if os.path.exists("modelo_autogl"):
            try:
                shutil.rmtree("modelo_autogl")
                print("🗑️ Modelo anterior eliminado")
            except Exception as e:
                print(f"⚠️ No se pudo eliminar modelo anterior: {e}")

        print(f"\n🚀 Iniciando entrenamiento AutoGluon...")
        print(f"📁 Modelo se guardará en: {save_path}")

        # Crear y entrenar predictor - AutoGluon maneja todo el preprocesamiento
        predictor = TabularPredictor(
            label=target, 
            path=save_path, 
            problem_type=problem_type,
            verbosity=2
        )

        # Entrenamiento - AutoGluon se encarga de todo automáticamente
        predictor.fit(
            df, 
            time_limit=time_limit,
            presets=presets,
            holdout_frac=0.2  # AutoGluon maneja la división automáticamente
        )

        print("✅ Entrenamiento completado")

        # Obtener resultados
        leaderboard_df = predictor.leaderboard(extra_info=True)
        best_model = leaderboard_df.iloc[0]["model"]
        best_score = leaderboard_df.iloc[0]["score_val"]
        metric_name = leaderboard_df.iloc[0]["eval_metric"]

        print(f"\n📊 Resultados del entrenamiento:")
        print(f"🏆 Mejor modelo: {best_model}")
        print(f" Metrica de validación: {metric_name}")
        print(f"📈 Score de validación: {best_score:.4f}")
        print(f"🔢 Modelos entrenados: {len(leaderboard_df)}")

        # Feature importance (si está disponible)
        feature_importance = None
        try:
            feature_importance = predictor.feature_importance(df, silent=True)
            print(f"\n🔝 Top 10 variables más importantes:")
            for idx, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
                print(f"   {idx:2d}. {feature}: {importance:.4f}")
        except Exception as e:
            print(f"⚠️ Feature importance no disponible: {e}")

        # Información del modelo
        model_info = predictor.info()
        
        # Métricas detalladas para el estado
        metricas_detalladas = {
            "mejor_modelo": best_model,
            "metrica": metric_name,
            "score_validacion": float(best_score),
            "cantidad_modelos": len(leaderboard_df),
            "tipo_problema": problem_type,
            "variable_objetivo": target,
            "tamaño_dataset": len(df),
            "tiempo_entrenamiento": time_limit,
            "preset_calidad": presets,
            "path_modelo": save_path,
            "target_info": target_info,
            "feature_importance": feature_importance.to_dict() if feature_importance is not None else None,
            "timestamp": timestamp,
            "model_info": str(model_info) if model_info else None
        }

        # Actualizar estado
        state["modelo_autogl"] = predictor
        state["leaderboard"] = leaderboard_df.to_dict()
        state["metricas_modelo"] = metricas_detalladas
        
        # Mensaje para el historial
        summary_message = f"""✅ Modelo AutoML entrenado exitosamente:

🏆 **Mejor modelo**: {best_model}
📈 **Score de validación**: {best_score:.4f}
🔢 **Modelos evaluados**: {len(leaderboard_df)}
🎯 **Tipo de problema**: {problem_type}
📊 **Dataset**: {len(df):,} filas
⏱️ **Tiempo**: {time_limit//60} minutos
🎚️ **Preset**: {presets}

AutoGluon se encargó automáticamente de:
- Limpieza y preprocesamiento de datos
- Selección de features
- Validación cruzada
- Optimización de hiperparámetros
- Ensembling de modelos"""

        state["messages"].append(AIMessage(
            content=summary_message,
            name="aplicar_automl"
        ))

        print(f"\n✅ Modelado automático completado exitosamente")
        print(f"📁 Modelo guardado en: {save_path}")
        
        return state

    except Exception as e:
        msg = f"Error durante el entrenamiento con AutoGluon: {str(e)}"
        print(f"❌ {msg}")
        state["errores"].append(msg)
        
        # Limpiar recursos en caso de error
        try:
            if 'save_path' in locals() and os.path.exists(save_path):
                shutil.rmtree(save_path)
        except:
            pass
            
        return state
