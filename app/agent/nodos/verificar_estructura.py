import pandas as pd
from langchain_core.messages import AIMessage
from agent.state_types import AgentState

def verificar_estructura(state: AgentState) -> AgentState:
    print("\n" + "=" * 80)
    print("🔬 VERIFICANDO ESTRUCTURA DEL DATASET (nodo: verificar_estructura)")
    print("=" * 80)
    
    try:
        df = state["df"]
        cant_filas, cant_columnas = df.shape
        print(f"📐 Dimensiones: {cant_filas:,} filas x {cant_columnas} columnas")

        # Tipos de datos
        tipos_conteo = df.dtypes.value_counts()
        print(f"🏷️  Tipos de datos encontrados:")
        for tipo, cantidad in tipos_conteo.items():
            print(f"   • {tipo}: {cantidad} columnas")

        # Nulos
        nulls = df.isnull().sum()
        columnas_con_nulls = nulls[nulls > 0]
        if not columnas_con_nulls.empty:
            print(f"⚠️  Valores nulos detectados en {len(columnas_con_nulls)} columnas:")
            for col, cantidad_nulls in columnas_con_nulls.items():
                porcentaje = (cantidad_nulls / cant_filas) * 100
                print(f"   • {col}: {cantidad_nulls:,} ({porcentaje:.1f}%)")
        else:
            print("✅ No se encontraron valores nulos")

        # Guardar estructura
        estructura = {
            "cant_filas": cant_filas,
            "cant_columnas": cant_columnas,
            "tipos": df.dtypes.astype(str).to_dict(),
            "nulls": nulls.to_dict()
        }
        state["estructura"] = estructura

        message = AIMessage(
            content=(
                f"Estructura verificada: {cant_filas:,} filas, {cant_columnas} columnas. "
                f"Tipos de datos y valores nulos analizados:\n"
                f"Tipos de datos de columnas: {estructura['tipos']}\n"
                f"Valores nulos por columna: {estructura['nulls']}"
            ),
            name="verificar_estructura"
        )
        state["messages"].append(message)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state["messages"]))
        print("✅ Estructura verificada correctamente")

    except Exception as e:
        error_msg = f"Error al verificar la estructura: {str(e)}"
        print(f"❌ {error_msg}")
        state.setdefault("errores", []).append(error_msg)

    return state
