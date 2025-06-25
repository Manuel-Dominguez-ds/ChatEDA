from agent.state_types import AgentState
from agent.utils.limpieza_tools import AVAILABLE_TOOLS,textual_description_of_tools
from agent.utils.prompts import construir_input_llm, SYSTEM_PROMPT, dataset_info_prompt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agent.nodos.verificar_estructura import verificar_estructura
from agent.nodos.resumen_estadistico import resumen_estadistico
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback
import json
import pandas as pd
from agent.state import  llm

tools = AVAILABLE_TOOLS

def analisis_limpieza(state: AgentState) -> AgentState:
    """
    Analiza si el dataset necesita limpieza y decide qué acción tomar
    """
    print("\n" + "="*80)
    print("🧠 ANÁLISIS DE LIMPIEZA CON LLM (nodo: analisis_limpieza)")
    print("="*80)
    
    try:
        # Construir el input estructurado para el LLM
        input_llm = construir_input_llm(state)
        historial_limpieza = state.get('historial_limpieza', [])
        
        print(f"📋 Preparando información para el LLM...")
        print(f"🔍 Pasos de limpieza previos: {len(historial_limpieza)}")
        print(f"🛠️  Tools disponibles: {len(tools)}")
        
        # Crear el mensaje con la información del dataset
        dataset_info = dataset_info_prompt(
            input_llm=input_llm,
            historial_limpieza=historial_limpieza,
            textual_description_of_tools=textual_description_of_tools
        )
        print("🤖 Consultando al LLM para decisión de limpieza...")
        
        
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            state['messages'][1],
            HumanMessage(content=dataset_info)
        ]
        
        # Invocar al LLM
        llm_response = llm.invoke(messages)
        response_content = llm_response.content.strip()
        
        print(f"💬 Respuesta del LLM recibida: {len(response_content)} caracteres")
        
        # Limpiar la respuesta si viene con marcadores de código
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
            print("🧹 Limpiando formato markdown de la respuesta")
        
        try:
            decision = json.loads(response_content)
            state['limpieza'] = decision
            
            print(f"📝 Decisión del LLM: {decision['action']}")
            print(f"💬 Mensaje: {decision.get('message', 'Sin mensaje')}")
            if decision.get('params'):
                print(f"⚙️  Parámetros: {decision['params']}")
            
            # Agregar al historial de decisiones
            historial = state.get('historial_limpieza', [])
            historial.append({
                'paso': len(historial) + 1,
                'decision': decision,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            state['historial_limpieza'] = historial
            if decision['action'] == 'no_limpieza_necesaria':
                print("✅ El LLM determinó que no es necesaria limpieza.")
                message1 = AIMessage(content="Informacion del dataset final:\n\n" + str(input_llm), name="informacion_dataset")
                state['messages'].append(message1)
                print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
                message2 = AIMessage(
                    content=f"Pasos de limpieza:\n\n{[decision['decision'] for decision in historial_limpieza]}\n\nNo se requiere mas limpieza.",
                    name="analisis_limpieza"
                )
                
                state['messages'].append(message2)
                print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
                return state
            else:
                message = AIMessage(
                    content=f"Decisión del LLM: {decision['action']}\n" +
                            f"Parámetros: {decision.get('params', {})}\n" +
                            f"Descripción: {decision.get('message', 'Sin descripción')}",
                    name="analisis_limpieza"
                )
                state['messages'].append(message)
                print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
            print(f"✅ Decisión registrada en el historial (paso {len(historial)})")
            
        except json.JSONDecodeError as e:
            error_msg = f'Respuesta no válida del LLM: {response_content}'
            print(f"❌ Error parseando JSON: {str(e)}")
            print(f"🔍 Respuesta problemática: {response_content[:200]}...")
            
            state['limpieza'] = {
                'action': 'error',
                'message': error_msg
            }
            state['errores'].append(error_msg)
        
        return state
        
    except Exception as e:
        error_msg = f"Error en análisis de limpieza: {str(e)}"
        print(f"❌ {error_msg}")
        state['errores'].append(error_msg)
        state['limpieza'] = {
            'action': 'error',
            'message': f'Error interno: {str(e)}'
        }
        return state
    
def aplicar_tool_limpieza(state: AgentState) -> AgentState:
    """
    Aplica la tool de limpieza seleccionada por el LLM basándose en el último elemento del historial
    """
    print("\n" + "="*80)
    print("🔧 APLICANDO TOOL DE LIMPIEZA (nodo: aplicar_tool_limpieza)")
    print("="*80)
    
    try:
        historial = state.get('historial_limpieza', [])
        if not historial:
            error_msg = "No hay decisiones en el historial para aplicar"
            print(f"❌ {error_msg}")
            state['errores'].append(error_msg)
            return state
            
        # Obtener la última decisión del historial
        ultima_decision = historial[-1]['decision']
        action = ultima_decision.get('action')
        params = ultima_decision.get('params', {})
        
        print(f"🎯 Tool a aplicar: {action}")
        print(f"⚙️  Parámetros: {params}")
        print(f"📊 Shape antes: {state['df'].shape}")
        
        # Buscar la tool por nombre en las tools disponibles
        tool_found = None
        for tool in AVAILABLE_TOOLS:
            if hasattr(tool, '__name__') and tool.__name__ == action:
                tool_found = tool
                break
        
        if not tool_found:
            error_msg = f"Tool '{action}' no encontrada en AVAILABLE_TOOLS"
            print(f"❌ {error_msg}")
            print(f"🔍 Tools disponibles: {[tool.__name__ for tool in AVAILABLE_TOOLS]}")
            state['errores'].append(error_msg)
            # Marcar como fallida en el historial
            historial[-1]['aplicada'] = False
            historial[-1]['resultado'] = f'error: {error_msg}'
            return state
        
        print(f"✅ Tool encontrada: {tool_found.__name__}")
        
        # Preparar parámetros para la tool
        tool_params = params.copy()
        
        # Reemplazar 'df' string con el DataFrame real
        if 'df' in tool_params and tool_params['df'] == 'df':
            tool_params['df'] = state['df']
        elif 'df' not in tool_params:
            tool_params['df'] = state['df']
        
        print(f"🚀 Ejecutando {action}...")
        
        # Ejecutar la tool
        df_limpio = tool_found(**tool_params)
        
        # Actualizar el estado con el DataFrame limpio
        state['df'] = df_limpio
        
        print(f"📊 Shape después: {df_limpio.shape}")
        shape_antes = historial[-1]['decision'].get('shape_antes', 'N/A')
        if shape_antes != 'N/A':
            filas_eliminadas = shape_antes[0] - df_limpio.shape[0] if isinstance(shape_antes, tuple) else 0
            print(f"📉 Filas eliminadas: {filas_eliminadas}")
        
        print("🔄 Actualizando estructura y resumen...")
        
        # Actualizar estructura y resumen con los nuevos datos
        state = verificar_estructura(state)
        state = resumen_estadistico(state)
        
        # Registrar la acción como aplicada exitosamente
        historial[-1]['aplicada'] = True
        historial[-1]['resultado'] = 'exitoso'
        historial[-1]['shape_despues'] = df_limpio.shape
        
        print(f"✅ Tool '{action}' aplicada exitosamente")
        
        message = ToolMessage(
            content=f"Tool '{action}' aplicada exitosamente con parámetros: {tool_params}",
            name=action,
            tool_call_id=ultima_decision.get('tool_call_id', None)
        )
        
        state['messages'].append(message)
        print("✅ Mensaje de Tool registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        
        return state
        
    except Exception as e:
        error_msg = f"Error aplicando tool: {str(e)}"
        print(f"❌ {error_msg}")
        state['errores'].append(error_msg)
        
        # Marcar como fallida en el historial
        historial = state.get('historial_limpieza', [])
        if historial:
            historial[-1]['aplicada'] = False
            historial[-1]['resultado'] = f'error: {error_msg}'
        
        return state
