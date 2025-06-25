import re 
from agent.state_types import AgentState
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils.prompts import construir_prompt_refinamiento,generar_codigo_grafico_prompt,sugerir_graficos_prompt,system_prompt_refinamiento
import os
import matplotlib.pyplot as plt
import time
from agent.state import llm

def sugerir_graficos_llm(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("🧠 NODO: sugerir_graficos_llm")
    print("="*80)
    
    try:
        prompt = sugerir_graficos_prompt
        
        state['messages'].append(HumanMessage(content=prompt))
        print("✅ Mensaje de Humano registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        print("🤖 Enviando a LLM...")
        response = llm.invoke(state['messages'])
        
        print("📝 Respuesta del LLM:")
        print(response.content)
        
        # 🆕 LIMPIEZA MEJORADA DE LA RESPUESTA
        response_content = response.content.strip()
        
        # Eliminar bloques de código markdown
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
            print("🧹 Eliminando formato markdown de la respuesta")
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
            print("🧹 Eliminando bloques de código de la respuesta")
        
        # Buscar el JSON válido usando regex
        json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
            print(f"🔍 JSON extraído: {json_content[:100]}...")
        else:
            json_content = response_content
            print("⚠️ No se encontró array JSON, usando respuesta completa")
        
        # Intentar parsear el JSON
        visualizaciones = json.loads(json_content)
        if not isinstance(visualizaciones, list):
            raise ValueError("La respuesta no es una lista JSON válida")
        
        state['visualizaciones'] = visualizaciones
        print(f"✅ Se sugirieron {len(visualizaciones)} visualizaciones")
        
            # Agregar mensaje de AI con las visualizaciones sugeridas
        state['messages'].append(AIMessage(
                content=f"Sugerencias de visualizaciones:\n{json.dumps(visualizaciones, indent=2, ensure_ascii=False, default=str)}",
                name="sugerir_graficos_llm"
        ))
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))
        print(f"📊 Dimensiones del dataset: {state['df'].shape}")

    except Exception as e:
        msg = f"❌ Error en generar_codigo_grafico_llm: {str(e)}"
        print(msg)
        state['errores'].append(msg)

    return state

def generar_codigo_grafico_llm(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("📊 NODO: generar_codigo_grafico_llm")
    print("="*80)

    try:
        print(state['df'].shape)
        df = state['df']
        visualizaciones = state.get('visualizaciones', [])
        generados = state.get('graficos_generados', [])
        contexto = state['messages'][1].content or "Análisis exploratorio para predecir una variable." 

        # Buscar la primera visualización pendiente
        pendiente = next((v for v in visualizaciones if v["id"] not in generados), None)
        if not pendiente:
            print("✅ Todos los gráficos ya fueron generados.")
            return state

        print(f"🛠️ Generando código para: {pendiente['id']} ({pendiente['tipo']})")

        # Preparar prompt
        graf_prompt = generar_codigo_grafico_prompt(pendiente, contexto)

        response = llm.invoke([
            SystemMessage(content="Sos un experto en visualización de datos y generación de gráficos con Python."),
            HumanMessage(content=graf_prompt)
        ])

        # Limpieza del código recibido
        codigo = response.content.strip()
        codigo = re.sub(r'if __name__ == [\'"]__main__[\'"]:(.*?)```', '', codigo, flags=re.DOTALL)
        codigo = codigo.strip('```python').strip('```').strip()
        codigo = codigo.replace("plt.show()", "")

        # Agregar ejecución automática si el código contiene una función
        match = re.search(r'def (\w+)\(df.*?\)', codigo)
        if match:
            nombre_funcion = match.group(1)

            # Buscar una línea comentada que invoque la función, por ejemplo:
            # plot_histogram(df, "col", "titulo")
            ejemplo_match = re.search(rf"#\s*{nombre_funcion}\((.*?)\)", codigo)

            if ejemplo_match:
                argumentos = ejemplo_match.group(1).strip()
                llamada_real = f"{nombre_funcion}({argumentos})"
                codigo += f"\n\n{llamada_real}"
                print(f"🔧 Se usó llamada comentada: {llamada_real}")
            else:
                # No hay llamada comentada, usar genérico
                llamada_generica = f"{nombre_funcion}(df)"
                codigo += f"\n\n{llamada_generica}"
                print(f"⚠️ No se encontró llamada comentada. Usando fallback: {llamada_generica}")
        else:
            print("❌ No se detectó ninguna función definida en el código.")


        # Guardar en el estado
        state['graficos'].append({"id": pendiente["id"], "codigo": codigo})
        state['graficos_generados'].append(pendiente["id"])
        
        print(f"✅ Código generado para {pendiente['id']}, guardado correctamente")

    except Exception as e:
        msg = f"❌ Error en generar_codigo_grafico_llm: {str(e)}"
        print(msg)
        state['errores'].append(msg)

    return state


def ejecutar_graficos(state: AgentState) -> AgentState:
    """
    Ejecuta el código Python generado por la LLM para crear gráficos
    y guarda las imágenes en disco, actualizando el estado.
    """
    print("\n" + "="*80)
    print("🖼️ NODO: ejecutar_graficos")
    print("="*80)

    try:
        df = state['df']
        output_dir = "graficos_generados"
        os.makedirs(output_dir, exist_ok=True)

        nuevos_graficos = {}

        for v in state['graficos']:
            # Saltar si ya es una ruta (ya fue ejecutado)
            graf_id = v['id']
            codigo = v['codigo']
            if isinstance(codigo, str) and os.path.exists(codigo):
                continue

            print(f"🧪 Ejecutando código para: {graf_id}")

            # Agregamos un cierre de figura automático para evitar overlaps
            exec_context = {"df": df, "plt": plt}
            try:
                exec(codigo, exec_context)

                # Guardar imagen
                ruta = os.path.join(output_dir, f"{graf_id}.png")
                plt.savefig(ruta, bbox_inches='tight')
                plt.close()

                nuevos_graficos[graf_id] = ruta
                print(f"✅ Guardado: {ruta}")

            except Exception as e:
                msg = f"❌ Error al ejecutar gráfico {graf_id}: {str(e)}"
                print(msg)
                error = {'grafico_id': graf_id, 'error': str(e)}
                state['errores_graficos'].append(error)
                state['errores'].append(msg)

        # Actualizar state.graficos reemplazando código por la ruta del archivo
        for graf in state['graficos']:
            graf_id = graf['id']
            if graf_id in nuevos_graficos:
                graf['ruta'] = nuevos_graficos[graf_id] # agregamos campo 'ruta'

        message = AIMessage(
            content=f"Gráficos ejecutados y guardados en {output_dir}. Nuevos gráficos generados: {len(nuevos_graficos)}",
            name="ejecutar_graficos"
        )
        state['messages'].append(message)
        print("✅ Mensaje de AI registrado en el historial. Cantidad de mensajes en el historial:", len(state['messages']))

    except Exception as e:
        msg = f"❌ Error general en ejecutar_graficos: {str(e)}"
        print(msg)
        state['errores'].append(msg)

    return state


def refinar_codigo_grafico_llm(state: AgentState) -> AgentState:
    """Refina el código de los gráficos generados que tienen errores de ejecución."""

    print("\n" + "="*80)
    print("🛠️ NODO: refinamiento de código gráfico")
    print("="*80)

    # 🆕 Incrementar contador de intentos
    intentos = state.get("refinamiento_intentos", 0) + 1
    state["refinamiento_intentos"] = intentos
    print(f"🔄 Intento de refinamiento #{intentos}")

    try:
        errores = state.get("errores_graficos", [])
        if not errores:
            print("✅ No hay gráficos para refinar.")
            return state

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        prompt_sistema = system_prompt_refinamiento

        errores_refinados = 0
        graficos_a_eliminar = [] 
        
        for error in errores[:]:  
            graf_id = error["grafico_id"]
            print(f"🔧 Refinando gráfico: {graf_id}")

            # Buscar el gráfico original
            grafico = next((g for g in state["graficos"] if g["id"] == graf_id), None)
            if not grafico:
                print(f"⚠️ No se encontró el gráfico {graf_id} en el estado.")
                continue

            original_code = grafico["codigo"]
            prompt_usuario = construir_prompt_refinamiento(error, original_code)
            
            try:
                # Invocar a la LLM
                print('Esperando')
                time.sleep(60)
                response = model.invoke([
                    SystemMessage(content=prompt_sistema),
                    HumanMessage(content=prompt_usuario)
                ])

                codigo_corregido = response.content.strip().strip("```python").strip("```")

                # Intentar descomentar llamada si es válida
                lineas = codigo_corregido.splitlines()
                for i, linea in enumerate(lineas):
                    if re.match(r"#\s*\w+\(.*\)", linea):  # detecta llamada comentada
                        try:
                            linea_eval = linea.lstrip("# ").strip()
                            compile(linea_eval, "<string>", "exec")
                            lineas[i] = linea_eval
                            print(f"🔧 Línea descomentada: {linea_eval}")
                            break
                        except SyntaxError:
                            continue

                grafico["codigo"] = "\n".join(lineas)
                errores_refinados += 1
                print(f"✅ Gráfico {graf_id} refinado exitosamente")
                
            except Exception as e:
                print(f"❌ Error refinando gráfico {graf_id}: {str(e)}")
                # 🆕 Marcar gráfico para eliminación completa
                graficos_a_eliminar.append(graf_id)
                print(f"🗑️ Gráfico {graf_id} será eliminado del flujo")

        # 🆕 ELIMINAR COMPLETAMENTE los gráficos que no se pudieron refinar
        if graficos_a_eliminar:
            print(f"\n🗑️ Eliminando {len(graficos_a_eliminar)} gráficos problemáticos:")
            
            # Eliminar de la lista de gráficos
            state["graficos"] = [g for g in state["graficos"] if g["id"] not in graficos_a_eliminar]
            
            # Eliminar de la lista de visualizaciones
            state["visualizaciones"] = [v for v in state.get("visualizaciones", []) if v["id"] not in graficos_a_eliminar]
            
            # Eliminar de gráficos generados
            state["graficos_generados"] = [g for g in state.get("graficos_generados", []) if g not in graficos_a_eliminar]
            
            # Eliminar insights de gráficos si existen
            state["insights_graficos"] = [i for i in state.get("insights_graficos", []) if i.get("id") not in graficos_a_eliminar]
            
            for graf_id in graficos_a_eliminar:
                print(f"   • {graf_id} eliminado completamente")

        # 🆕 Limpiar TODOS los errores (ya que los problemáticos fueron eliminados)
        state["errores_graficos"] = []
        
        print(f"\n✅ Refinamiento completado:")
        print(f"   • Gráficos refinados exitosamente: {errores_refinados}")
        print(f"   • Gráficos eliminados por errores: {len(graficos_a_eliminar)}")
        print(f"   • Gráficos restantes: {len(state['graficos'])}")

    except Exception as e:
        error_msg = f"❌ Error refinando código gráfico: {str(e)}"
        print(error_msg)
        state["errores"].append(error_msg)

    return state
