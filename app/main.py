from agent.state_types import AgentState
import pandas as pd 
import json
import os
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from agent.utils.prompts import SYSTEM_PROMPT
from agent.graph import react_graph

def obtener_bibliotecas_mas_usadas(graficos_info):
        """Función auxiliar para contar bibliotecas más utilizadas"""
        contador_bibliotecas = {}
        for grafico in graficos_info:
            bibliotecas = grafico.get('analisis_codigo', {}).get('bibliotecas_utilizadas', [])
            for lib in bibliotecas:
                contador_bibliotecas[lib] = contador_bibliotecas.get(lib, 0) + 1
        
        return sorted(contador_bibliotecas.items(), key=lambda x: x[1], reverse=True)

def guardar_estado_final(state_invoked, output_dir="outputs"):
    """
    Guarda el estado final del agente en formato JSON limpio
    Excluye DataFrames y objetos binarios, solo mantiene información textual
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        estado_limpio = {}
        
        estado_limpio['metadata'] = {
            'archivo_procesado': state_invoked.get('archivo_input', ''),
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_errores': len(state_invoked.get('errores', [])),
            'pasos_limpieza': len(state_invoked.get('historial_limpieza', [])),
            'graficos_generados': len(state_invoked.get('graficos', [])),
            'modelo_ml_entrenado': state_invoked.get('modelo_autogl') is not None
        }
        
        if not state_invoked.get('df', pd.DataFrame()).empty:
            df = state_invoked['df']
            estado_limpio['dataset_info'] = {
                'filas': len(df),
                'columnas': len(df.columns),
                'nombres_columnas': df.columns.tolist(),
                'tipos_datos': df.dtypes.astype(str).to_dict(),
                'valores_nulos': df.isnull().sum().to_dict(),
                'memoria_uso_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        
        estado_limpio['estructura'] = state_invoked.get('estructura', {})
        estado_limpio['resumen'] = state_invoked.get('resumen', {})
        
        estado_limpio['insights'] = state_invoked.get('insights', '')
        
        estado_limpio['historial_limpieza'] = state_invoked.get('historial_limpieza', [])
        
        estado_limpio['visualizaciones'] = state_invoked.get('visualizaciones', [])
        
        graficos_info = []
        for i, grafico in enumerate(state_invoked.get('graficos', [])):
            grafico_limpio = {
                'id': grafico.get('id', f'grafico_{i+1}'),
                'ruta_archivo': grafico.get('ruta', ''),
                'codigo_python': grafico.get('codigo', ''),  # 🆕 CÓDIGO COMPLETO
                'codigo_lineas': len(str(grafico.get('codigo', '')).splitlines()),  # 🆕 NÚMERO DE LÍNEAS
                'tamaño_codigo': len(str(grafico.get('codigo', ''))),  # 🆕 TAMAÑO DEL CÓDIGO
                'archivo_existe': os.path.exists(grafico.get('ruta', '')) if grafico.get('ruta') else False,
                'metadata': {
                    'creado': grafico.get('creado', False),
                    'timestamp': grafico.get('timestamp', ''),
                    'tipo_grafico': grafico.get('tipo', 'unknown')
                }
            }
            
            codigo = str(grafico.get('codigo', ''))
            if codigo:
                bibliotecas_detectadas = []
                if 'import matplotlib' in codigo or 'plt.' in codigo:
                    bibliotecas_detectadas.append('matplotlib')
                if 'import seaborn' in codigo or 'sns.' in codigo:
                    bibliotecas_detectadas.append('seaborn')
                if 'import pandas' in codigo or 'pd.' in codigo:
                    bibliotecas_detectadas.append('pandas')
                if 'import numpy' in codigo or 'np.' in codigo:
                    bibliotecas_detectadas.append('numpy')
                
                grafico_limpio['analisis_codigo'] = {
                    'bibliotecas_utilizadas': bibliotecas_detectadas,
                    'contiene_funciones': 'def ' in codigo,
                    'contiene_plt_show': 'plt.show()' in codigo,
                    'contiene_plt_savefig': 'plt.savefig' in codigo,
                    'lineas_comentarios': len([line for line in codigo.splitlines() if line.strip().startswith('#')]),
                    'lineas_codigo': len([line for line in codigo.splitlines() if line.strip() and not line.strip().startswith('#')])
                }
            
            graficos_info.append(grafico_limpio)
        
        estado_limpio['graficos'] = graficos_info
        
        if graficos_info:
            estado_limpio['resumen_codigos'] = {
                'total_graficos': len(graficos_info),
                'graficos_con_codigo': len([g for g in graficos_info if g['codigo_python']]),
                'total_lineas_codigo': sum([g['codigo_lineas'] for g in graficos_info]),
                'bibliotecas_mas_usadas': obtener_bibliotecas_mas_usadas(graficos_info),
                'graficos_con_funciones': len([g for g in graficos_info if g.get('analisis_codigo', {}).get('contiene_funciones', False)])
            }
        
        estado_limpio['insights_graficos'] = state_invoked.get('insights_graficos', [])
        
        if state_invoked.get('modelo_autogl'):
            estado_limpio['machine_learning'] = {
                'modelo_entrenado': True,
                'metricas_modelo': state_invoked.get('metricas_modelo', {}),
                'leaderboard': state_invoked.get('leaderboard', {}),
                'estrategia_negocio': state_invoked.get('estrategia_negocio', {}),
                'modelo_interpretacion': state_invoked.get('modelo_interpretacion', {}),
                'reporte_ml': state_invoked.get('reporte_ml', '')
            }
        else:
            estado_limpio['machine_learning'] = {
                'modelo_entrenado': False,
                'motivo': 'No se entrenó modelo de ML en este análisis'
            }
        
        estado_limpio['archivos_generados'] = {
            'reporte_final': state_invoked.get('reporte_final', ''),
            'carpeta_graficos': 'graficos_generados/',
            'carpeta_reportes': 'reportes/'
        }
        
        estado_limpio['errores'] = state_invoked.get('errores', [])
        
        estado_limpio['errores_graficos'] = state_invoked.get('errores_graficos', [])
        
        mensajes_texto = []
        for msg in state_invoked.get('messages', []):
            if hasattr(msg, 'content'):
                mensaje_limpio = {
                    'tipo': msg.__class__.__name__,
                    'contenido': msg.content[:1000] + '...' if len(str(msg.content)) > 1000 else str(msg.content),
                    'name': getattr(msg, 'name', None)  # 🆕 INCLUIR NOMBRE SI EXISTE
                }
                mensajes_texto.append(mensaje_limpio)
        estado_limpio['conversacion'] = mensajes_texto
        
        campos_texto = ['limpieza', 'modelo_sugerido', 'decision_usuario', 'refinamiento_intentos']
        for campo in campos_texto:
            if campo in state_invoked:
                estado_limpio[campo] = state_invoked[campo]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo_nombre = f"estado_final_{timestamp}.json"
        ruta_archivo = os.path.join(output_dir, archivo_nombre)
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(estado_limpio, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ Estado final guardado en: {ruta_archivo}")
        print(f"📊 Información incluida:")
        print(f"   • Dataset: {estado_limpio['dataset_info']['filas']} filas, {estado_limpio['dataset_info']['columnas']} columnas")
        print(f"   • Limpieza: {len(estado_limpio['historial_limpieza'])} pasos")
        print(f"   • Gráficos: {len(estado_limpio['graficos'])} visualizaciones")
        if estado_limpio.get('resumen_codigos'):
            print(f"   • Códigos: {estado_limpio['resumen_codigos']['total_lineas_codigo']} líneas totales") # 🆕
        print(f"   • ML: {'✅' if estado_limpio['machine_learning']['modelo_entrenado'] else '❌'}")
        print(f"   • Errores: {len(estado_limpio['errores'])}")
        
        return ruta_archivo
        
    except Exception as e:
        print(f"❌ Error guardando estado final: {e}")
        return None

if __name__ == "__main__":

    state = AgentState(
        archivo_input='train (1).csv',
        df=pd.DataFrame(),
        estructura={},
        resumen={},
        insights="",
        limpieza="",
        historial_limpieza=[], 
        
        visualizaciones=[],
        graficos_generados=[],
        graficos=[],
        insights_graficos=[],
        errores_graficos=[],
        refinamiento_intentos=0,
        
        modelo_sugerido={},
        estrategia_negocio="",
        decision_usuario="", 
            
        modelo_autogl = None,                 # Predictor de AutoGluon entrenado
        leaderboard = {},                  # Resultados de todos los modelos
        metricas_modelo = {},              # Métricas detalladas del mejor modelo
        predicciones = pd.DataFrame(),        # Predicciones del modelo (opcional)
        modelo_interpretacion = {},      # Feature importance y análisis del modelo
        reporte_ml = "",            # Reporte final del modelo
        
        reporte_final="",
        errores=[],  
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="Es un dataset de pozos petroleros, la idea es poder predecir la producción del petróleo, la columna 'production_rate', en base a las otras columnas del dataset. Ayudame a limpiar el dataset y generar insights para realizar un modelo.")
        ]
    )

    print("\n" + "="*100)
    print("🚀 INICIANDO AGENTE DE ANÁLISIS DE DATOS")
    print("="*100)
    print(f"📂 Archivo a procesar: {state['archivo_input']}")
    print(f"🎯 Objetivo: Limpieza y generación de insights")
    print("="*100)

    # Ejecutar el grafo
    state_invoked = react_graph.invoke(state,config={"recursion_limit":50})

    print("\n" + "="*100)
    print("🏁 AGENTE DE ANÁLISIS COMPLETADO")
    print("="*100)
    print(f"❌ Errores encontrados: {len(state_invoked.get('errores', []))}")
    print(f"🧹 Pasos de limpieza aplicados: {len(state_invoked.get('historial_limpieza', []))}")
    print(f"💡 Insights generados: {'✅' if state_invoked.get('insights') else '❌'}")
    print("="*100)
    
    # 🆕 GUARDAR ESTADO FINAL
    print("\n" + "="*50)
    print("💾 GUARDANDO ESTADO FINAL")
    print("="*50)
    archivo_estado = guardar_estado_final(state_invoked)
    if archivo_estado:
        print(f"✅ Estado guardado exitosamente")
    else:
        print("❌ Error al guardar el estado")
    print("="*50)