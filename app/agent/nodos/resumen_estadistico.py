import pandas as pd
from langchain_core.messages import AIMessage
from agent.state_types import AgentState

def resumen_estadistico(state: AgentState) -> AgentState:
    print("\n" + "=" * 80)
    print("📊 GENERANDO RESUMEN ESTADÍSTICO (nodo: resumen_estadistico)")
    print("=" * 80)

    df = state.get("df")
    if df is None:
        error_msg = "No se encontró DataFrame en el estado."
        print(f"❌ {error_msg}")
        state.setdefault("errores", []).append(error_msg)
        return state

    resumen = {}

    try:
        # Variables numéricas
        numericas = df.select_dtypes(include=["number"])
        print(f"🔢 Analizando {len(numericas.columns)} variables numéricas:")
        for col in numericas.columns:
            stats = numericas[col].describe()
            print(f"   • {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, media={stats['mean']:.2f}")

        resumen["numericas"] = numericas.describe().to_dict()

        # Variables categóricas
        categoricas = df.select_dtypes(include=["object", "category", "bool"])
        print(f"🏷️  Analizando {len(categoricas.columns)} variables categóricas:")

        resumen_cat = {}
        for col in categoricas.columns:
            nunique = categoricas[col].nunique()
            top_value = categoricas[col].mode().iloc[0] if not categoricas[col].mode().empty else None
            freq = categoricas[col].value_counts().iloc[0] if not categoricas[col].value_counts().empty else None

            print(f"   • {col}: {nunique} valores únicos, más frecuente: '{top_value}' ({freq} veces)")

            resumen_cat[col] = {
                "nunique": nunique,
                "top": top_value,
                "freq": freq
            }

        resumen["categoricas"] = resumen_cat

        # Guardar en el estado
        state["resumen"] = resumen

        message = AIMessage(
            content=(
                f"Resumen estadístico generado:\n"
                f"Variables numéricas:\n{resumen['numericas']}\n\n"
                f"Variables categóricas:\n{resumen['categoricas']}"
            ),
            name="resumen_estadistico"
        )
        state["messages"].append(message)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state["messages"]))
        print("✅ Resumen estadístico generado correctamente")

    except Exception as e:
        error_msg = f"Error al generar resumen estadístico: {str(e)}"
        print(f"❌ {error_msg}")
        state.setdefault("errores", []).append(error_msg)

    return state
