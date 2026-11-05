"""
Entrenamiento y Serialización de Modelos - Clase 12: Deploy
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Flujo de trabajo real:
1. Lee el dataset existente: 'datos_clientes.csv'
2. Entrena los 3 modelos de Machine Learning:
   - Clasificación: Árbol de Decisión (Aprobación de crédito)
   - Regresión: Regresión Lineal (Estimación de monto de préstamo)
   - Clustering: K-Means (Segmentación de perfiles de clientes)
3. Serializa y GENERA los 3 archivos de modelos:
   - modelo_clasificacion_arbol.joblib
   - modelo_regresion_lineal.joblib
   - modelo_clustering_kmeans.joblib
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib

DIR_ACTUAL = Path(__file__).parent
ARCHIVO_CSV = DIR_ACTUAL / "datos_clientes.csv"

def cargar_datos() -> pd.DataFrame:
    """Lee el dataset 'datos_clientes.csv'."""
    if not ARCHIVO_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{ARCHIVO_CSV.name}'. "
            "Asegúrate de que 'datos_clientes.csv' esté en la misma carpeta."
        )
    df = pd.read_csv(ARCHIVO_CSV)
    return df

def main():
    print("=========================================================")
    print("  CLASE 12: ENTRENAMIENTO DE LOS 3 MODELOS PARA DEPLOY   ")
    print("=========================================================")
    
    # 1. Cargar el dataset existente
    print(f"\n1. Leyendo datos desde '{ARCHIVO_CSV.name}'...")
    df = cargar_datos()
    features = ['edad', 'ingresos_k', 'score_crediticio']
    X = df[features]
    print(f"   -> Dataset cargado con éxito: {len(df)} registros.")
    print(df[features + ['aprobado', 'monto_prestamo_k']].head(3))
    
    # -------------------------------------------------------------
    # MODELO 1: CLASIFICACIÓN (Árbol de Decisión)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("2. Entrenando y GENERANDO: modelo_clasificacion_arbol.joblib")
    y_clf = df['aprobado']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo_arbol.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, modelo_arbol.predict(X_test_c))
    print(f"   Exactitud (Accuracy): {acc * 100:.1f}%")
    
    archivo_clf = DIR_ACTUAL / "modelo_clasificacion_arbol.joblib"
    joblib.dump(modelo_arbol, archivo_clf)
    print(f"   ✅ Archivo generado: {archivo_clf.name}")
    
    # -------------------------------------------------------------
    # MODELO 2: REGRESIÓN (Regresión Lineal)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("3. Entrenando y GENERANDO: modelo_regresion_lineal.joblib")
    y_reg = df['monto_prestamo_k']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X_train_r, y_train_r)
    r2 = r2_score(y_test_r, modelo_regresion.predict(X_test_r))
    print(f"   Coeficiente R2: {r2:.3f}")
    
    archivo_reg = DIR_ACTUAL / "modelo_regresion_lineal.joblib"
    joblib.dump(modelo_regresion, archivo_reg)
    print(f"   ✅ Archivo generado: {archivo_reg.name}")
    
    # -------------------------------------------------------------
    # MODELO 3: CLUSTERING (K-Means k=3, No Supervisado)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("4. Entrenando y GENERANDO: modelo_clustering_kmeans.joblib")
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    modelo_kmeans.fit(X)
    
    archivo_clu = DIR_ACTUAL / "modelo_clustering_kmeans.joblib"
    joblib.dump(modelo_kmeans, archivo_clu)
    print(f"   ✅ Archivo generado: {archivo_clu.name}")
    
    print("\n=========================================================")
    print("¡Listo! Se generaron los 3 archivos .joblib correctamente.")
    print("Ahora ya puedes ejecutar cualquiera de las aplicaciones Streamlit.")
    print("=========================================================")

if __name__ == "__main__":
    main()
