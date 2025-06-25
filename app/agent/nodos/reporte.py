from datetime import datetime
import os
from agent.state_types  import AgentState
import base64
import re
import pandas as pd

def reporte_final_llm_mejorado(state: AgentState) -> AgentState:
    print("\n" + "="*80)
    print("📄 NODO: reporte_final_llm_mejorado")
    print("="*80)

    try:
        output_dir = "reportes"
        os.makedirs(output_dir, exist_ok=True)

        df = state['df']
        insights_raw = state['insights']
        graficos = state['graficos']
        insights_graficos = state.get('insights_graficos', [])
        limpieza = state.get('historial_limpieza', [])
        
        # 🆕 OBTENER RESULTADOS DE MACHINE LEARNING
        modelo_autogl = state.get('modelo_autogl')
        metricas_modelo = state.get('metricas_modelo', {})
        leaderboard = state.get('leaderboard', {})
        estrategia_negocio = state.get('estrategia_negocio', {})

        nombre_archivo = f"reporte_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        ruta_reporte = os.path.join(output_dir, nombre_archivo)

        print(f"📝 Generando reporte mejorado en: {ruta_reporte}")
        print(f"🖼️ Gráficos disponibles: {len(graficos)}")
        print(f"💡 Insights de gráficos disponibles: {len(insights_graficos)}")
        print(f"🤖 Modelo ML disponible: {'✅' if modelo_autogl else '❌'}")

        # Función para convertir imágenes a base64
        def imagen_a_base64(ruta_imagen):
            try:
                if not os.path.exists(ruta_imagen):
                    return None
                with open(ruta_imagen, "rb") as img_file:
                    img_data = img_file.read()
                    base64_string = base64.b64encode(img_data).decode('utf-8')
                    return base64_string
            except Exception as e:
                print(f"❌ Error convirtiendo imagen {ruta_imagen}: {e}")
                return None

        # Asegurar rutas de gráficos
        for grafico in graficos:
            if 'ruta' not in grafico:
                ruta_esperada = os.path.join("graficos_generados", f"{grafico['id']}.png")
                if os.path.exists(ruta_esperada):
                    grafico['ruta'] = ruta_esperada

        # Función para obtener insight de un gráfico específico
        def obtener_insight_grafico(graf_id):
            """Busca el insight correspondiente a un gráfico específico"""
            for insight in insights_graficos:
                if insight.get('id') == graf_id:
                    return insight.get('insights', 'No se generaron insights para este gráfico.')
            return 'No se generaron insights para este gráfico.'

        # Función mejorada para procesar insights
        def procesar_insights(text):
            if not text:
                return "<p class='no-content'>No hay insights disponibles.</p>"
            
            # Limpiar secciones innecesarias
            text = re.sub(r'# 📊 ANÁLISIS COMPLETO DEL DATASET.*?---', '', text, flags=re.DOTALL)
            text = re.sub(r'## 📈 MÉTRICAS CLAVE.*?---', '', text, flags=re.DOTALL)
            text = re.sub(r'## VISUALIZACIONES GENERADAS.*?---', '', text, flags=re.DOTALL)
            text = re.sub(r'---\s*## 🛠️ PROCESO DE LIMPIEZA APLICADO:.*$', '', text, flags=re.DOTALL)
            
            # Dividir en secciones principales
            secciones = text.split('##')
            html_sections = ""
            
            for i, seccion in enumerate(secciones):
                if seccion.strip():
                    # Procesar título de sección
                    lines = seccion.strip().split('\n')
                    if lines:
                        titulo = lines[0].strip()
                        contenido = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                        
                        # Convertir contenido
                        contenido_html = contenido
                        contenido_html = re.sub(r'\*\*(.*?)\*\*', r'<strong class="highlight">\1</strong>', contenido_html)
                        contenido_html = re.sub(r'^\* (.*)', r'<li class="insight-item">• \1</li>', contenido_html, flags=re.MULTILINE)
                        contenido_html = re.sub(r'^- (.*)', r'<li class="insight-item">• \1</li>', contenido_html, flags=re.MULTILINE)
                        
                        # Agrupar listas
                        lines = contenido_html.split('\n')
                        result = []
                        in_list = False
                        
                        for line in lines:
                            if '<li class="insight-item">' in line:
                                if not in_list:
                                    result.append('<ul class="insight-list">')
                                    in_list = True
                                result.append(line)
                            else:
                                if in_list:
                                    result.append('</ul>')
                                    in_list = False
                                if line.strip():
                                    result.append(f'<p class="insight-text">{line.strip()}</p>')
                        
                        if in_list:
                            result.append('</ul>')
                        
                        contenido_final = '\n'.join(result)
                        
                        html_sections += f"""
                        <div class="insight-section">
                            <h3 class="insight-section-title">{titulo}</h3>
                            <div class="insight-content">
                                {contenido_final}
                            </div>
                        </div>
                        """
            
            return html_sections

        insights_html = procesar_insights(insights_raw)

        # 🆕 FUNCIÓN PARA GENERAR SECCIÓN DE MACHINE LEARNING (CORREGIDA)
        def generar_seccion_ml():
            if not modelo_autogl or not metricas_modelo:
                return '''
                <div class="no-content-card">
                    <div class="no-content-icon">🤖</div>
                    <h3>Sin Modelo de Machine Learning</h3>
                    <p>No se entrenó ningún modelo de Machine Learning en este análisis.</p>
                </div>
                '''
            
            # Extraer métricas principales
            mejor_modelo = metricas_modelo.get('mejor_modelo', 'N/A')
            score_validacion = metricas_modelo.get('score_validacion', 0)
            cantidad_modelos = metricas_modelo.get('cantidad_modelos', 0)
            tipo_problema = metricas_modelo.get('tipo_problema', 'N/A')
            variable_objetivo = metricas_modelo.get('variable_objetivo', 'N/A')
            feature_importance_raw = metricas_modelo.get('feature_importance', {})
            metrica_nombre = metricas_modelo.get('metrica', 'N/A')
            
            # 🆕 PROCESAMIENTO MEJORADO DE FEATURE IMPORTANCE
            feature_importance_html = ""
            if feature_importance_raw:
                try:
                    print(f"🔍 Procesando feature importance: {type(feature_importance_raw)}")
                    
                    # Convertir a diccionario si es necesario
                    if isinstance(feature_importance_raw, dict):
                        feature_importance_dict = feature_importance_raw
                    elif hasattr(feature_importance_raw, 'to_dict'):
                        feature_importance_dict = feature_importance_raw.to_dict()
                    elif isinstance(feature_importance_raw, pd.Series):
                        feature_importance_dict = feature_importance_raw.to_dict()
                    else:
                        print(f"⚠️ Tipo de feature importance no reconocido: {type(feature_importance_raw)}")
                        feature_importance_dict = {}
                    
                    # Filtrar solo valores numéricos válidos
                    valid_features = {}
                    for feature, importance in feature_importance_dict.items():
                        try:
                            # Convertir a float si es posible
                            if isinstance(importance, (int, float)):
                                valid_features[feature] = float(importance)
                            elif isinstance(importance, str):
                                valid_features[feature] = float(importance)
                            else:
                                print(f"⚠️ Valor no numérico ignorado para {feature}: {importance} ({type(importance)})")
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ Error procesando {feature}: {e}")
                            continue
                    
                    if valid_features:
                        # Ordenar por importancia absoluta
                        top_features = sorted(valid_features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                        
                        feature_importance_html = """
                        <div class="feature-importance">
                            <h4 class="section-subtitle">🔝 Variables Más Importantes</h4>
                            <div class="importance-table">
                        """
                        
                        # Calcular el máximo valor para normalizar las barras
                        max_importance = max(abs(importance) for _, importance in top_features) if top_features else 1
                        
                        for i, (feature, importance) in enumerate(top_features, 1):
                            # Normalizar importancia para la barra (0-100%)
                            bar_width = (abs(importance) / max_importance) * 100 if max_importance > 0 else 0
                            
                            feature_importance_html += f"""
                            <div class="importance-row">
                                <div class="importance-rank">{i}</div>
                                <div class="importance-name">{str(feature)[:30]}{'...' if len(str(feature)) > 30 else ''}</div>
                                <div class="importance-bar">
                                    <div class="importance-fill" style="width: {bar_width:.1f}%"></div>
                                </div>
                                <div class="importance-value">{importance:.3f}</div>
                            </div>
                            """
                        
                        feature_importance_html += """
                            </div>
                        </div>
                        """
                    else:
                        print("⚠️ No se encontraron valores válidos de feature importance")
                
                except Exception as e:
                    print(f"❌ Error procesando feature importance: {e}")
                    print(f"📊 Datos recibidos: {feature_importance_raw}")
            
            # Información del problema de negocio
            problema_negocio_html = ""
            if estrategia_negocio:
                problema_negocio_html = f"""
                <div class="business-problem">
                    <h4 class="section-subtitle">🎯 Problema de Negocio</h4>
                    <div class="problem-card">
                        <p><strong>Descripción:</strong> {estrategia_negocio.get('problema_negocio', 'N/A')}</p>
                        <p><strong>Justificación ML:</strong> {estrategia_negocio.get('justificacion_ml', 'N/A')}</p>
                        <p><strong>Comentarios:</strong> {estrategia_negocio.get('comentarios_adicionales', 'N/A')}</p>
                    </div>
                </div>
                """
            
            return f"""
            <div class="ml-results">
                <div class="ml-summary">
                    <div class="ml-metrics-grid">
                        <div class="ml-metric-card primary">
                            <div class="metric-icon">🏆</div>
                            <div class="metric-content">
                                <div class="metric-value">{score_validacion:.3f}</div>
                                <div class="metric-label">{metrica_nombre}</div>
                            </div>
                        </div>
                        <div class="ml-metric-card">
                            <div class="metric-icon">🤖</div>
                            <div class="metric-content">
                                <div class="metric-value">{cantidad_modelos}</div>
                                <div class="metric-label">Modelos Evaluados</div>
                            </div>
                        </div>
                        <div class="ml-metric-card">
                            <div class="metric-icon">🎯</div>
                            <div class="metric-content">
                                <div class="metric-value">{tipo_problema.title()}</div>
                                <div class="metric-label">Tipo de Problema</div>
                            </div>
                        </div>
                        <div class="ml-metric-card">
                            <div class="metric-icon">📊</div>
                            <div class="metric-content">
                                <div class="metric-value">{variable_objetivo}</div>
                                <div class="metric-label">Variable Objetivo</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="ml-details">
                    <div class="model-info">
                        <h4 class="section-subtitle">🥇 Mejor Modelo</h4>
                        <div class="model-card">
                            <h5>{mejor_modelo}</h5>
                            <p>Este modelo fue seleccionado automáticamente por AutoGluon tras evaluar {cantidad_modelos} algoritmos diferentes.</p>
                        </div>
                    </div>
                    
                    {problema_negocio_html}
                    {feature_importance_html}
                </div>
            </div>
            """

        ml_html = generar_seccion_ml()

        # Estadísticas del dataset (incluyendo contadores ML)
        completitud = round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
        tiene_modelo = 1 if modelo_autogl else 0
        
        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-content">
                    <div class="stat-number">{len(df):,}</div>
                    <div class="stat-label">Filas de Datos</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📋</div>
                <div class="stat-content">
                    <div class="stat-number">{len(df.columns)}</div>
                    <div class="stat-label">Variables</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-content">
                    <div class="stat-number">{completitud}%</div>
                    <div class="stat-label">Completitud</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🤖</div>
                <div class="stat-content">
                    <div class="stat-number">{tiene_modelo}</div>
                    <div class="stat-label">Modelo ML</div>
                </div>
            </div>
        </div>
        """

        # Mapeo de nombres de funciones a nombres más descriptivos
        nombres_funciones = {
            'remove_columns': 'Eliminar Columnas',
            'encode_categorical_columns': 'Codificar Variables Categóricas',
            'standardize_numeric_columns': 'Estandarizar Variables Numéricas',
            'remove_duplicates': 'Eliminar Duplicados',
            'handle_missing_values': 'Manejar Valores Faltantes',
            'remove_outliers': 'Eliminar Valores Atípicos',
            'convert_data_types': 'Convertir Tipos de Datos',
            'clean_text_column': 'Limpiar Columnas de Texto'
        }

        # Proceso de limpieza mejorado
        limpieza_html = ""
        pasos_validos = [item for item in limpieza if item.get('decision', {}).get('action') != 'no_limpieza_necesaria']
        
        if pasos_validos:
            for i, item in enumerate(pasos_validos, 1):
                if item.get('decision'):
                    accion = item['decision'].get('action', 'N/A')
                    mensaje = item['decision'].get('message', 'Sin descripción')
                    resultado = item.get('resultado', 'pendiente')
                    
                    # Nombre más descriptivo
                    nombre_descriptivo = nombres_funciones.get(accion, accion.replace('_', ' ').title())
                    
                    status_icon = "✅" if resultado == "exitoso" else "⚠️" if resultado == "pendiente" else "❌"
                    status_class = "success" if resultado == "exitoso" else "warning" if resultado == "pendiente" else "error"
                    
                    limpieza_html += f"""
                    <div class="process-step {status_class}">
                        <div class="step-header">
                            <div class="step-number">
                                <span class="number">{i}</span>
                                <div class="step-line"></div>
                            </div>
                            <div class="step-info">
                                <h4 class="step-title">{nombre_descriptivo}</h4>
                                <div class="step-status">
                                    <span class="status-icon">{status_icon}</span>
                                    <span class="status-text">{'Completado' if resultado == 'exitoso' else 'Error' if 'error' in resultado else 'Pendiente'}</span>
                                </div>
                            </div>
                        </div>
                        <div class="step-description">
                            <p>{mensaje}</p>
                        </div>
                    </div>
                    """
        else:
            limpieza_html = '''
            <div class="no-process">
                <div class="no-process-icon">🎯</div>
                <h3>Dataset Optimizado</h3>
                <p>Los datos ya se encontraban en excelente estado y no requirieron pasos de limpieza adicionales.</p>
            </div>
            '''

        # Generar sección de gráficos CON INSIGHTS
        graficos_html = ""
        if graficos and len(graficos) > 0:
            graficos_procesados = 0
            for i, grafico in enumerate(graficos):
                graf_id = grafico['id']
                ruta_imagen = grafico.get('ruta')
                
                if ruta_imagen and os.path.exists(ruta_imagen):
                    file_size = os.path.getsize(ruta_imagen)
                    if file_size > 0:
                        img_base64 = imagen_a_base64(ruta_imagen)
                        if img_base64:
                            # Obtener descripción
                            descripcion = "Visualización de datos"
                            visualizaciones = state.get('visualizaciones', [])
                            for viz in visualizaciones:
                                if viz.get('id') == graf_id:
                                    descripcion = viz.get('descripcion', descripcion)
                                    break
                            
                            # Obtener insight específico del gráfico
                            insight_grafico = obtener_insight_grafico(graf_id)
                            
                            # Procesar el insight para formato HTML
                            insight_procesado = insight_grafico.replace('\n', '<br>')
                            insight_procesado = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', insight_procesado)
                            insight_procesado = re.sub(r'^\* (.*)', r'<li>\1</li>', insight_procesado, flags=re.MULTILINE)
                            
                            # Si hay elementos de lista, envolverlos en <ul>
                            if '<li>' in insight_procesado:
                                lines = insight_procesado.split('<br>')
                                processed_lines = []
                                in_list = False
                                
                                for line in lines:
                                    if '<li>' in line:
                                        if not in_list:
                                            processed_lines.append('<ul>')
                                            in_list = True
                                        processed_lines.append(line)
                                    else:
                                        if in_list:
                                            processed_lines.append('</ul>')
                                            in_list = False
                                        if line.strip():
                                            processed_lines.append(line)
                                
                                if in_list:
                                    processed_lines.append('</ul>')
                                
                                insight_procesado = '<br>'.join(processed_lines)
                            
                            graficos_html += f"""
                            <div class="chart-container">
                                <div class="chart-header">
                                    <div class="chart-info">
                                        <h3 class="chart-title">{graf_id.replace('_', ' ').title()}</h3>
                                        <p class="chart-description">{descripcion}</p>
                                    </div>
                                    <div class="chart-badge">
                                        <span class="badge-number">{i+1}</span>
                                    </div>
                                </div>
                                <div class="chart-content">
                                    <img src="data:image/png;base64,{img_base64}" alt="{graf_id}" class="chart-image">
                                </div>
                                <div class="chart-insights">
                                    <h4 class="insights-title">💡 Análisis del Gráfico</h4>
                                    <div class="insights-content">
                                        <p>{insight_procesado}</p>
                                    </div>
                                </div>
                            </div>
                            """
                            graficos_procesados += 1
            
            if graficos_procesados == 0:
                graficos_html = '''
                <div class="no-content-card">
                    <div class="no-content-icon">📊</div>
                    <h3>Gráficos en Proceso</h3>
                    <p>Los gráficos están siendo generados. Verifica la carpeta "graficos_generados".</p>
                </div>
                '''
        else:
            graficos_html = '''
            <div class="no-content-card">
                <div class="no-content-icon">📈</div>
                <h3>Sin Visualizaciones</h3>
                <p>No se generaron visualizaciones para este análisis.</p>
            </div>
            '''

        # HTML completo con estilos mejorados (incluyendo estilos para ML)
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Análisis de Datos - DataViz AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --background: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.95);
            --border-radius: 16px;
            --shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.15);
            --insights-bg: #f8fafc;
            --insights-border: #e2e8f0;
            --ml-primary: #8b5cf6;
            --ml-secondary: #a78bfa;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            min-height: 100vh;
            gap: 24px;
            padding: 24px;
        }}

        /* SIDEBAR */
        .sidebar {{
            width: 320px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            padding: 32px 24px;
            box-shadow: var(--shadow-lg);
            position: sticky;
            top: 24px;
            height: fit-content;
        }}

        .sidebar-header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 24px;
            background: var(--primary-gradient);
            border-radius: 12px;
            color: white;
        }}

        .sidebar-header h1 {{
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 8px;
        }}

        .sidebar-header p {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .nav-menu {{
            list-style: none;
        }}

        .nav-item {{
            margin-bottom: 8px;
        }}

        .nav-link {{
            display: flex;
            align-items: center;
            padding: 16px 20px;
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: 12px;
            gap: 16px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}

        .nav-link:hover {{
            background: var(--primary-gradient);
            color: white;
            transform: translateX(4px);
        }}

        .nav-icon {{
            font-size: 20px;
            width: 24px;
            text-align: center;
        }}

        /* MAIN CONTENT */
        .main-content {{
            flex: 1;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            padding: 48px;
            box-shadow: var(--shadow-lg);
            overflow-y: auto;
            max-height: calc(100vh - 48px);
        }}

        /* HEADER */
        .header {{
            text-align: center;
            margin-bottom: 64px;
            padding: 48px 32px;
            background: var(--primary-gradient);
            border-radius: var(--border-radius);
            color: white;
        }}

        .header h1 {{
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 16px;
            background: linear-gradient(45deg, #ffffff, #e2e8f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}

        /* SECTIONS */
        .section {{
            margin-bottom: 80px;
        }}

        .section-title {{
            font-size: 36px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 3px solid transparent;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        /* STATS GRID */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: white;
            padding: 32px;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            display: flex;
            align-items: center;
            gap: 20px;
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-4px);
        }}

        .stat-icon {{
            font-size: 32px;
            width: 64px;
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--primary-gradient);
            border-radius: 12px;
            color: white;
        }}

        .stat-content {{
            flex: 1;
        }}

        .stat-number {{
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 4px;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .stat-label {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* 🆕 ESTILOS PARA MACHINE LEARNING */
        .ml-results {{
            background: white;
            border-radius: var(--border-radius);
            padding: 32px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }}

        .ml-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .ml-metric-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 24px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: transform 0.3s ease;
        }}

        .ml-metric-card.primary {{
            background: linear-gradient(135deg, var(--ml-primary) 0%, var(--ml-secondary) 100%);
            color: white;
            transform: scale(1.02);
        }}

        .ml-metric-card:hover {{
            transform: translateY(-2px);
        }}

        .ml-metric-card.primary:hover {{
            transform: scale(1.02) translateY(-2px);
        }}

        .metric-icon {{
            font-size: 24px;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
        }}

        .ml-metric-card:not(.primary) .metric-icon {{
            background: var(--primary-gradient);
            color: white;
        }}

        .metric-content {{
            flex: 1;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 4px;
        }}

        .ml-metric-card:not(.primary) .metric-value {{
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .metric-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
        }}

        .ml-details {{
            display: grid;
            gap: 24px;
        }}

        .model-info, .business-problem {{
            background: #f8fafc;
            padding: 24px;
            border-radius: 12px;
            border-left: 4px solid var(--ml-primary);
        }}

        .section-subtitle {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .model-card, .problem-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}

        .model-card h5 {{
            font-size: 20px;
            font-weight: 700;
            color: var(--ml-primary);
            margin-bottom: 8px;
        }}

        .feature-importance {{
            background: #f8fafc;
            padding: 24px;
            border-radius: 12px;
            border-left: 4px solid #10b981;
        }}

        .importance-table {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}

        .importance-row {{
            display: grid;
            grid-template-columns: auto 1fr auto auto;
            gap: 16px;
            padding: 12px 16px;
            align-items: center;
            border-bottom: 1px solid #e2e8f0;
        }}

        .importance-row:last-child {{
            border-bottom: none;
        }}

        .importance-rank {{
            width: 24px;
            height: 24px;
            background: var(--primary-gradient);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
        }}

        .importance-name {{
            font-weight: 600;
            color: var(--text-primary);
        }}

        .importance-bar {{
            width: 100px;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }}

        .importance-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            transition: width 0.3s ease;
        }}

        .importance-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            min-width: 60px;
            text-align: right;
        }}

        /* PROCESS STEPS */
        .process-step {{
            background: white;
            border-radius: var(--border-radius);
            padding: 24px;
            margin-bottom: 16px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--success-color);
        }}

        .process-step.warning {{
            border-left-color: var(--warning-color);
        }}

        .process-step.error {{
            border-left-color: var(--error-color);
        }}

        .step-header {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 16px;
        }}

        .step-number {{
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .step-number .number {{
            width: 40px;
            height: 40px;
            background: var(--primary-gradient);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 16px;
        }}

        .step-line {{
            width: 2px;
            height: 20px;
            background: linear-gradient(to bottom, #667eea, transparent);
            margin-top: 8px;
        }}

        .step-info {{
            flex: 1;
        }}

        .step-title {{
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}

        .step-status {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .status-icon {{
            font-size: 18px;
        }}

        .status-text {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        .step-description {{
            margin-left: 60px;
            padding: 16px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 3px solid #e2e8f0;
        }}

        .step-description p {{
            color: var(--text-secondary);
            font-size: 15px;
        }}

        /* NO PROCESS */
        .no-process {{
            text-align: center;
            padding: 48px 32px;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
        }}

        .no-process-icon {{
            font-size: 48px;
            margin-bottom: 16px;
        }}

        .no-process h3 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 12px;
            color: var(--text-primary);
        }}

        .no-process p {{
            color: var(--text-secondary);
            font-size: 16px;
        }}

        /* INSIGHTS */
        .insight-section {{
            background: white;
            border-radius: var(--border-radius);
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: var(--shadow);
        }}

        .insight-section-title {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
        }}

        .insight-content {{
            line-height: 1.7;
        }}

        .insight-text {{
            margin-bottom: 16px;
            color: var(--text-secondary);
            font-size: 16px;
        }}

        .insight-list {{
            margin: 16px 0;
            padding-left: 0;
            list-style: none;
        }}

        .insight-item {{
            padding: 8px 0;
            color: var(--text-secondary);
            font-size: 15px;
        }}

        .highlight {{
            background: linear-gradient(120deg, #667eea20, #764ba220);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        /* CHARTS */
        .chart-container {{
            background: white;
            border-radius: var(--border-radius);
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: var(--shadow);
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 24px;
        }}

        .chart-info {{
            flex: 1;
        }}

        .chart-title {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}

        .chart-description {{
            color: var(--text-secondary);
            font-size: 16px;
        }}

        .chart-badge {{
            background: var(--primary-gradient);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }}

        .chart-content {{
            text-align: center;
            margin-bottom: 24px;
        }}

        .chart-image {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }}

        /* ESTILOS PARA INSIGHTS DE GRÁFICOS */
        .chart-insights {{
            background: var(--insights-bg);
            border: 2px solid var(--insights-border);
            border-radius: 12px;
            padding: 24px;
            margin-top: 20px;
        }}

        .insights-title {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .insights-content {{
            color: var(--text-secondary);
            font-size: 15px;
            line-height: 1.6;
        }}

        .insights-content p {{
            margin-bottom: 12px;
        }}

        .insights-content strong {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .insights-content ul {{
            margin: 12px 0;
            padding-left: 20px;
        }}

        .insights-content li {{
            margin-bottom: 8px;
            color: var(--text-secondary);
        }}

        /* NO CONTENT */
        .no-content-card {{
            text-align: center;
            padding: 48px 32px;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
        }}

        .no-content-icon {{
            font-size: 48px;
            margin-bottom: 16px;
        }}

        .no-content-card h3 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 12px;
            color: var(--text-primary);
        }}

        .no-content-card p {{
            color: var(--text-secondary);
            font-size: 16px;
        }}

        /* RESPONSIVE */
        @media (max-width: 1200px) {{
            .container {{
                flex-direction: column;
            }}
            
            .sidebar {{
                width: 100%;
                position: static;
            }}
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 16px;
                gap: 16px;
            }}
            
            .main-content {{
                padding: 24px;
            }}
            
            .header h1 {{
                font-size: 36px;
            }}
            
            .section-title {{
                font-size: 28px;
            }}
            
            .chart-insights, .ml-results {{
                padding: 16px;
            }}
            
            .ml-metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .importance-row {{
                grid-template-columns: auto 1fr;
                gap: 8px;
            }}
            
            .importance-bar, .importance-value {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <nav class="sidebar">
            <div class="sidebar-header">
                <h1>📊 DataViz AI</h1>
                <p>Análisis Inteligente de Datos</p>
            </div>
            <ul class="nav-menu">
                <li class="nav-item">
                    <a href="#resumen" class="nav-link">
                        <span class="nav-icon">📊</span>
                        <span class="nav-text">Resumen Ejecutivo</span>
                    </a>
                </li>
                <li class="nav-item">
                    <a href="#limpieza" class="nav-link">
                        <span class="nav-icon">🧼</span>
                        <span class="nav-text">Proceso de Limpieza</span>
                    </a>
                </li>
                <li class="nav-item">
                    <a href="#insights" class="nav-link">
                        <span class="nav-icon">💡</span>
                        <span class="nav-text">Análisis e Insights</span>
                    </a>
                </li>
                <li class="nav-item">
                    <a href="#graficos" class="nav-link">
                        <span class="nav-icon">📈</span>
                        <span class="nav-text">Visualizaciones</span>
                    </a>
                </li>
                <li class="nav-item">
                    <a href="#machine-learning" class="nav-link">
                        <span class="nav-icon">🤖</span>
                        <span class="nav-text">Machine Learning</span>
                    </a>
                </li>
            </ul>
        </nav>

        <main class="main-content">
            <div class="header">
                <h1>Reporte de Análisis de Datos</h1>
                <p>Generado el {datetime.now().strftime('%d de %B de %Y a las %H:%M hrs')}</p>
            </div>

            <section id="resumen" class="section">
                <h2 class="section-title">📊 Resumen Ejecutivo</h2>
                {stats_html}
            </section>

            <section id="limpieza" class="section">
                <h2 class="section-title">🧼 Proceso de Limpieza de Datos</h2>
                {limpieza_html}
            </section>

            <section id="insights" class="section">
                <h2 class="section-title">💡 Análisis e Insights</h2>
                {insights_html}
            </section>

            <section id="graficos" class="section">
                <h2 class="section-title">📈 Visualizaciones</h2>
                {graficos_html}
            </section>

            <section id="machine-learning" class="section">
                <h2 class="section-title">🤖 Machine Learning</h2>
                {ml_html}
            </section>
        </main>
    </div>

    <script>
        // Smooth scrolling mejorado
        document.querySelectorAll('.nav-link').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {{
                    targetElement.scrollIntoView({{ 
                        behavior: 'smooth', 
                        block: 'start',
                        inline: 'nearest'
                    }});
                    
                    // Añadir efecto visual de navegación activa
                    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                }}
            }});
        }});

        // Intersection Observer para navegación activa
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.nav-link');

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const id = entry.target.getAttribute('id');
                    navLinks.forEach(link => {{
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${{id}}`) {{
                            link.classList.add('active');
                        }}
                    }});
                }}
            }});
        }}, {{ threshold: 0.1 }});

        sections.forEach(section => {{
            observer.observe(section);
        }});
    </script>
</body>
</html>
        """

        # Guardar el archivo
        with open(ruta_reporte, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n✅ Reporte completo generado: {ruta_reporte}")
        print(f"📊 Gráficos procesados: {len([g for g in graficos if g.get('ruta')])}")
        print(f"🤖 Modelo ML incluido: {'✅' if modelo_autogl else '❌'}")
        state['reporte_final'] = ruta_reporte

    except Exception as e:
        msg = f"❌ Error en reporte_final_llm_mejorado: {str(e)}"
        print(msg)
        state['errores'].append(msg)

    return state