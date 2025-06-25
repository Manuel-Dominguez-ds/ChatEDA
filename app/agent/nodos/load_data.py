import pandas as pd
from langchain_core.messages import AIMessage
from agent.state_types import AgentState

def load_data(state: AgentState) -> AgentState:
    '''
    Nodo: Carga un archivo CSV o Excel y actualiza el estado del agente con el DataFrame.
    
    Args:
        state (AgentState): Estado actual del agente.
    
    Returns:
        AgentState: Estado actualizado con el DataFrame y mensajes.
    '''
    print("\n" + "="*80)
    print("🔍 INICIANDO CARGA DE DATOS (nodo: load_data)")
    print("="*80)
    print(f"📁 Archivo a cargar: {state['archivo_input']}")
    
    try:
        print("⏳ Cargando archivo...")
        if state['archivo_input'].endswith('.csv'):
            df = pd.read_csv(state['archivo_input'])
        else:
            df = pd.read_excel(state['archivo_input'])

        state['df'] = df
        
        print(f"✅ Archivo cargado correctamente")
        print(f"📊 Dimensiones del dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        print(f"💾 Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"🔤 Columnas disponibles: {list(df.columns)}")
        
        message = AIMessage(content=f"Archivo cargado correctamente con {df.shape[0]:,} filas y {df.shape[1]} columnas. Columnas: {list(df.columns)}")
        state['messages'].append(message)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))

    except Exception as e:
        error_msg = f"❌ Error al cargar el archivo: {str(e)}"
        print(error_msg)
        state.setdefault('errores', []).append(error_msg)

    return state
