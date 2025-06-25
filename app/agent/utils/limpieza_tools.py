import pandas as pd
import numpy as np
from typing import List, Union, Any, Optional, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# -------- TOOLS -------- #

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Elimina filas duplicadas del DataFrame
    
    Args:
        df: DataFrame a procesar
        subset: Lista de columnas a considerar para duplicados. Si es None, considera todas las columnas
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates(subset=subset)
    final_rows = len(df_cleaned)
    print(f"Filas eliminadas: {initial_rows - final_rows}")
    return df_cleaned

def handle_missing_values(df: pd.DataFrame, columns: Union[str, List[str]], method: str = "drop", fill_value: Any = None) -> pd.DataFrame:
    """
    Maneja valores faltantes en una o múltiples columnas
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas
        method: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'fill_value'
        fill_value: Valor específico para rellenar (solo si method='fill_value')
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    # Verificar que todas las columnas existan
    missing_cols = [col for col in columns if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    for column in columns:
        if method == "drop":
            df_copy = df_copy.dropna(subset=[column])
        elif method == "mean" and df_copy[column].dtype in ['int64', 'float64']:
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        elif method == "median" and df_copy[column].dtype in ['int64', 'float64']:
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        elif method == "mode":
            mode_value = df_copy[column].mode().iloc[0] if not df_copy[column].mode().empty else None
            if mode_value is not None:
                df_copy[column] = df_copy[column].fillna(mode_value)
        elif method == "forward_fill":
            df_copy[column] = df_copy[column].fillna(method='ffill')
        elif method == "backward_fill":
            df_copy[column] = df_copy[column].fillna(method='bfill')
        elif method == "fill_value" and fill_value is not None:
            df_copy[column] = df_copy[column].fillna(fill_value)
    
    return df_copy

def remove_outliers(df: pd.DataFrame, columns: Union[str, List[str]], method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
    """
    Elimina outliers de una o múltiples columnas numéricas
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas numéricas
        method: 'iqr' o 'zscore'
        factor: Factor para el método IQR (default 1.5) o threshold para z-score (default 1.5)
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    # Verificar que todas las columnas existan y sean numéricas
    for column in columns:
        if column not in df_copy.columns:
            raise ValueError(f"Columna '{column}' no encontrada")
        if df_copy[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Columna '{column}' debe ser numérica")
    
    # Crear máscara para filtrar outliers
    mask = pd.Series([True] * len(df_copy), index=df_copy.index)
    
    for column in columns:
        if method == "iqr":
            Q1 = df_copy[column].quantile(0.25)
            Q3 = df_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            column_mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
            mask = mask & column_mask
        
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[column].dropna()))
            # Crear máscara para esta columna considerando NaN
            column_mask = pd.Series([True] * len(df_copy), index=df_copy.index)
            valid_indices = df_copy[column].dropna().index
            column_mask.loc[valid_indices] = z_scores < factor
            mask = mask & column_mask
    
    df_copy = df_copy[mask]
    initial_rows = len(df)
    final_rows = len(df_copy)
    print(f"Filas eliminadas por outliers: {initial_rows - final_rows}")
    
    return df_copy

def convert_data_types(df: pd.DataFrame, columns_types: Dict[str, str]) -> pd.DataFrame:
    """
    Convierte el tipo de datos de múltiples columnas
    
    Args:
        df: DataFrame a procesar
        columns_types: Diccionario con {nombre_columna: tipo_objetivo}
                      tipos válidos: 'int', 'float', 'string', 'datetime', 'category'
    """
    df_copy = df.copy()
    
    # Verificar que todas las columnas existan
    missing_cols = [col for col in columns_types.keys() if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    for column, target_type in columns_types.items():
        try:
            if target_type == "int":
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce').astype('Int64')
            elif target_type == "float":
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            elif target_type == "string":
                df_copy[column] = df_copy[column].astype(str)
            elif target_type == "datetime":
                df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
            elif target_type == "category":
                df_copy[column] = df_copy[column].astype('category')
            else:
                raise ValueError(f"Tipo '{target_type}' no válido para columna '{column}'")
        except Exception as e:
            raise ValueError(f"Error convirtiendo columna '{column}' a {target_type}: {str(e)}")
    
    return df_copy

def remove_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Elimina una o múltiples columnas específicas del DataFrame
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas a eliminar
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    existing_columns = [col for col in columns if col in df_copy.columns]
    if existing_columns:
        df_copy = df_copy.drop(columns=existing_columns)
        print(f"Columnas eliminadas: {existing_columns}")
    else:
        print("No se encontraron columnas para eliminar")
    
    return df_copy

def clean_text_column(df: pd.DataFrame, columns: Union[str, List[str]], operations: List[str]) -> pd.DataFrame:
    """
    Limpia una o múltiples columnas de texto
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas de texto
        operations: Lista de operaciones ['strip', 'lower', 'upper', 'remove_special_chars']
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    # Verificar que todas las columnas existan
    missing_cols = [col for col in columns if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    for column in columns:
        for operation in operations:
            if operation == "strip":
                df_copy[column] = df_copy[column].astype(str).str.strip()
            elif operation == "lower":
                df_copy[column] = df_copy[column].astype(str).str.lower()
            elif operation == "upper":
                df_copy[column] = df_copy[column].astype(str).str.upper()
            elif operation == "remove_special_chars":
                df_copy[column] = df_copy[column].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    
    print(f"Operaciones aplicadas a columnas {columns}: {operations}")
    return df_copy

def standardize_numeric_columns(df: pd.DataFrame, columns: Union[str, List[str]], method: str = "zscore") -> pd.DataFrame:
    """
    Estandariza columnas numéricas usando Z-score o Min-Max scaling
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas numéricas
        method: 'zscore' para estandarización Z-score, 'minmax' para Min-Max scaling
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    # Verificar que todas las columnas existan y sean numéricas
    for column in columns:
        if column not in df_copy.columns:
            raise ValueError(f"Columna '{column}' no encontrada")
        if df_copy[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Columna '{column}' debe ser numérica")
    
    for column in columns:
        if method == "zscore":
            mean_val = df_copy[column].mean()
            std_val = df_copy[column].std()
            df_copy[column] = (df_copy[column] - mean_val) / std_val
        elif method == "minmax":
            min_val = df_copy[column].min()
            max_val = df_copy[column].max()
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Método '{method}' no válido. Use 'zscore' o 'minmax'")
    
    print(f"Estandarización {method} aplicada a columnas: {columns}")
    return df_copy

def encode_categorical_columns(df: pd.DataFrame, columns: Union[str, List[str]], method: str = "onehot") -> pd.DataFrame:
    """
    Codifica columnas categóricas
    
    Args:
        df: DataFrame a procesar
        columns: Nombre de la columna o lista de columnas categóricas
        method: 'onehot' para One-Hot Encoding, 'label' para Label Encoding
    """
    df_copy = df.copy()
    
    # Convertir a lista si es string
    if isinstance(columns, str):
        columns = [columns]
    
    # Verificar que todas las columnas existan
    missing_cols = [col for col in columns if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    for column in columns:
        if method == "onehot" and df_copy[column].nunique() < 50:
            # One-Hot Encoding
            dummies = pd.get_dummies(df_copy[column], prefix=column)
            df_copy = pd.concat([df_copy.drop(column, axis=1), dummies], axis=1)
        elif method == "label":
            # Label Encoding
            unique_values = df_copy[column].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            df_copy[column] = df_copy[column].map(label_map)
        else:
            raise ValueError(f"Método '{method}' no válido. Use 'onehot' o 'label'")
    
    print(f"Codificación {method} aplicada a columnas: {columns}")
    return df_copy

def generar_nueva_tool(nombre: str, descripcion: str):
    """
    Genera una nueva tool con nombre y descripción, y la agrega a AVAILABLE_TOOLS.
    
    Args:
        nombre: Nombre de la tool
        descripcion: Descripción de la tool
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_retries=3,
    )
    
    system_prompt = '''Eres un experto en limpieza de datos y generación de herramientas para procesamiento de DataFrames. 
Tu tarea es crear una nueva herramienta que realice una operación específica sobre un DataFrame. 
La herramienta debe ser capaz de recibir un DataFrame y devolver un DataFrame modificado según la operación definida.

Debes seguir las mejores prácticas de programación y asegurarte de que la herramienta sea eficiente y fácil de usar.
La herramienta debe ser capaz de manejar errores comunes y proporcionar mensajes claros en caso de fallos.
Tu respuesta debe ser un código Python válido que defina una función con el nombre y la descripción proporcionados.
La función debe incluir un docstring que explique su propósito, los parámetros de entrada y el valor de retorno.

Ejemplo de respuesta:
```python
def nombre_de_la_funcion(df: pd.DataFrame, parametro1: tipo, parametro2: tipo) -> pd.DataFrame:
    \"\"\"Descripción de la función.
    Args:   
        df: DataFrame a procesar
        parametro1: Descripción del parámetro 1
        parametro2: Descripción del parámetro 2
    Returns:
        DataFrame modificado
    \"\"\"
    # Lógica de la función
    return df_modificado
```'''

    prompt = f'''Crea una nueva función de Python llamada {nombre} que realice la siguiente operación sobre un DataFrame:
{descripcion}.

Asegúrate de que la función sea eficiente, maneje errores comunes y proporcione mensajes claros en caso de fallos.
Solo devolvé el código de la función.'''
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    if not response or not response.content:
        raise ValueError("No se pudo generar la herramienta. Respuesta vacía del LLM.")
    
    code = response.content.strip()

    # Extraer el bloque de código si viene dentro de markdown
    if code.startswith("```python"):
        code = code.split("```python")[1].split("```")[0].strip()
    elif code.startswith("```"):
        code = code.split("```")[1].split("```")[0].strip()

    # Ejecutar el código para registrar la función
    local_vars = {}
    try:
        exec(code, globals(), local_vars)
    except Exception as e:
        raise RuntimeError(f"Error ejecutando la función generada:\n{code}\n\nError: {e}")

    # Recuperar la función desde local_vars
    funcion_generada = local_vars.get(nombre)
    if funcion_generada is None or not callable(funcion_generada):
        raise ValueError(f"No se pudo encontrar la función '{nombre}' luego de ejecutarla.")

    # Agregarla a la lista de herramientas disponibles
    AVAILABLE_TOOLS.append(funcion_generada)

    print(f"✅ Función '{nombre}' generada y añadida a AVAILABLE_TOOLS.")
    return funcion_generada

# -------- LISTA Y DESCRIPCIONES -------- #

AVAILABLE_TOOLS = [
    remove_duplicates,
    handle_missing_values,
    remove_outliers,
    convert_data_types,
    remove_columns,
    clean_text_column,
    standardize_numeric_columns,
    encode_categorical_columns,
    generar_nueva_tool
]

textual_description_of_tools = ''
for tool in AVAILABLE_TOOLS:
    if hasattr(tool, '__doc__') and tool.__doc__:
        textual_description_of_tools += f"{tool.__name__}: {tool.__doc__}\n\n"
    else:
        textual_description_of_tools += f"{tool.__name__}: No hay descripción disponible.\n\n"

