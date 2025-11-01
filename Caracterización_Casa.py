import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from scipy import stats


# ====================================================================================
Perfil_de_carga = 'Y'  # Condición para generar perfil de carga (Y/N)
Curva_de_duracion = 'Y'  # Condición para generar curva de duración de la carga (Y/N)



# Cargar datos saltando las filas de metadatos
df = pd.read_excel('Registro Casa Cumbres de Curumo.xlsx', sheet_name='Datos', skiprows=8)

# Renombrar columnas si es necesario (basado en la estructura que viste)
df.columns = ['Fecha', 'Hora', 'VmdR', 'IrTot', 'VmdS', 'IsTot', 'VmdT', 'ItTot']

# Eliminar filas con valores NaN
df = df.dropna()

# Convertir fecha y hora a datetime
df['Datetime'] = pd.to_datetime(df['Fecha'].astype(str) + ' ' + df['Hora'].astype(str), 
                            format='%d/%m/%y %H:%M')

# Calcular kVA para cada fase
df['kVA_R'] = (df['VmdR'] * df['IrTot']) / 1000
df['kVA_S'] = (df['VmdS'] * df['IsTot']) / 1000
df['kVA_T'] = (df['VmdT'] * df['ItTot']) / 1000

# Asegurar orden por fecha
df = df.sort_values('Datetime')

# Definir pot_total siempre (fuera del if) para que la CDC lo pueda usar
pot_total = df['kVA_R'] + df['kVA_S'] + df['kVA_T']

# Agregar columna de potencia total al DataFrame.
df['pot_total'] = pot_total

# =================================================================
# PERFIL DE CARGA, por fase y total.
# =================================================================

def generar_perfil_de_carga(df):

    # Crear gráficos - gráfico combinado de las tres fases (se guarda como perfil_de_carga.png)
    import matplotlib.dates as mdates
    from matplotlib.dates import DayLocator, HourLocator, DateFormatter

    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(df['Datetime'], df['kVA_R'], label='Fase R', linewidth=1, alpha=0.8)
    ax1.plot(df['Datetime'], df['kVA_S'], label='Fase S', linewidth=1, alpha=0.8)
    ax1.plot(df['Datetime'], df['kVA_T'], label='Fase T', linewidth=1, alpha=0.8)
    ax1.set_title('Perfil de Carga - Potencia Aparente (kVA) por Fase', fontsize=14, fontweight='bold')
    ax1.set_ylabel('kVA')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Formato eje x: días como etiqueta principal y horas como marcas menores
    ax1.xaxis.set_major_locator(DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter('%d-%b'))
    ax1.xaxis.set_minor_locator(HourLocator(interval=3))
    ax1.xaxis.set_minor_formatter(DateFormatter('%H:%M'))
    ax1.tick_params(axis='x', which='major', labelrotation=0, labelsize=10)
    ax1.tick_params(axis='x', which='minor', labelrotation=45, labelsize=8, pad=10)

    plt.tight_layout()
    fig1.savefig('perfil_de_carga.png', dpi=300)

    # Gráfico 2: Potencia total en archivo separado (potencia_total.png)
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    pot_total = df['kVA_R'] + df['kVA_S'] + df['kVA_T']
    ax2.plot(df['Datetime'], pot_total, label='Potencia Total', linewidth=1.5, color='black')
    ax2.set_title('Potencia Total del Sistema', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha y Hora')
    ax2.set_ylabel('kVA Total')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mismo tratamiento de ejes: días principales, horas menores (se ven las horas)
    ax2.xaxis.set_major_locator(DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(DateFormatter('%d-%b'))
    ax2.xaxis.set_minor_locator(HourLocator(interval=3))
    ax2.xaxis.set_minor_formatter(DateFormatter('%H:%M'))
    ax2.tick_params(axis='x', which='major', labelrotation=0, labelsize=10)
    ax2.tick_params(axis='x', which='minor', labelrotation=45, labelsize=8, pad=10)
    plt.tight_layout()
    fig2.savefig('potencia_total.png', dpi=300)
    
    return

if Perfil_de_carga == 'Y':
    generar_perfil_de_carga(df)
    
# =================================================================
# CURVA DE DURACIÓN DE LA CARGA (CDC) y tabla energía vs % tiempo
# =================================================================
def generar_curva_duracion(df):

    def curva_duracion_potencia(df, columna_tiempo='Datetime', columna_potencia='pot_total'):
        """
        Crea curva de duración para datos de potencia cada 5 minutos
        
        Parameters:
        df: DataFrame con columnas de tiempo y potencia
        columna_tiempo: nombre de la columna de tiempo
        columna_potencia: nombre de la columna de potencia
        """
        
        # Ordenar potencia de mayor a menor
        df_sorted = df.sort_values(columna_potencia, ascending=False)
        
        # Calcular probabilidad de excedencia (porcentaje de tiempo)
        n = len(df_sorted)
        df_sorted['Orden'] = range(1, n + 1)
        df_sorted['Porcentaje_Tiempo'] = (df_sorted['Orden'] / n) * 100
        
        # Crear tabla resumen
        tabla_duracion = df_sorted[[columna_tiempo, columna_potencia, 'Porcentaje_Tiempo']].copy()
        
        return tabla_duracion

    tabla_duracion = curva_duracion_potencia(df)
    # Asegurar orden por fecha
    tabla_duracion = tabla_duracion.sort_values('pot_total', ascending=False)
    tabla_duracion = round(tabla_duracion, 2)

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    # datos para graficar
    x = tabla_duracion['Porcentaje_Tiempo'].values
    y = tabla_duracion['pot_total'].values

    # ancho dinámico (más pequeño = más espacio entre barras)
    x_range = x.max() - x.min() if x.max() != x.min() else 1.0
    width = (x_range / len(x)) * 0.6  # ajustar el factor 0.6 para más/menos separación
    width = max(width, 0.05)          # límite inferior para que no desaparezcan las barras

    ax3.bar(x, y, width=width, align='center', alpha=0.8, color='black', edgecolor='k', linewidth=0.2)

    # márgenes en eje x (p. ej. 1% extra a cada lado) y eje y (10% sobre el máximo)
    x_margin = x_range * 0.01
    ax3.set_xlim(x.min() - 0.3, x.max() + 0.3)

    ax3.set_ylim(0, y.max() * 1.05)

    # reducir número de ticks en x para mejorar lectura
    num_ticks = 25       # Número de divisiones en x deseadas.
    xticks = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], num_ticks)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([f"{t:.0f}" for t in xticks], rotation=45, ha='right')

    ax3.set_title('Curva de Duración de la Carga - Potencia Total (kVA)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('kVA')
    ax3.set_xlabel('Porcentaje de Tiempo (%)')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig('curva_duracion_carga.png', dpi=300)
    
    return tabla_duracion
    


if Curva_de_duracion == 'Y':
    Tabla_duracion = generar_curva_duracion(df)
    
# Obtener valor máximo de potencia total.
D_max = df['pot_total'].max() #Demanda máxima en kVA.

# ===================================================================================
#                            DEMANDA PROMEDIO.
# ===================================================================================
# 1. Calcular potencia total en kW para cada medición
df['Potencia_kW'] = pot_total

# 2. Calcular energía consumida en cada intervalo de 5 minutos
intervalo_horas = 5 / 60 # 5 minutos = 5/60 = 0.0833 horas
df['Energia_kWh'] = pot_total * intervalo_horas

# 3. Calcular energía total consumida
energia_total = df['Energia_kWh'].sum()

# 4. Calcular tiempo total
tiempo_total = len(df) * intervalo_horas  # 1077 × 0.0833 = 89.75 horas

# 5. Aplicar la fórmula de demanda promedio
demanda_promedio = energia_total / tiempo_total

# ===================================================================================
#                            DEMANDA DIVERSIFICADA.
# ===================================================================================
# Demanda diversificada (D_div) es igual a la demanda máxima (D_max) por tener una sola carga.
D_div = D_max

# ===================================================================================
#                           DEMANDA COINCIDENTE MÁXIMA.
# ===================================================================================
# En este caso, al tratarse de una sola carga, la Demanda Coincidente
# Máxima (DCM) es igual a la Demanda Máxima (D_max).
DCM = D_max

# ===================================================================================
#                           DEMANDA NO-COINCIDENTE MÁXIMA.
# ===================================================================================
# En este caso, al tratarse de una sola carga, la Demanda No-Coincidente
# Máxima (DnCM) es igual a la Demanda Máxima (D_max).
DnCM = D_max

# ===================================================================================
#                                   FACTOR DE DEMANDA.
# ===================================================================================
# Capacidad Intalada de la carga (kVA) (CTC).
CTC = 61.042 # kVA (Resultados en Excel de la casa.)
Fd = D_max / CTC

# ===================================================================================
#                                   FACTOR DE UTILIZACIÓN.
# ===================================================================================
Capacidad_Sistema = (800 * 120 * 3) / 1000  # 800A * 120V * 3F / 1000 = 288 kVA
Fu = D_max / Capacidad_Sistema  # Datos del Excel de la casa.

# ===================================================================================
#                                     FACTOR DE CARGA.
# ===================================================================================
Fc = demanda_promedio / D_max

# ===================================================================================
#                                   FACTOR DE DIVERSIDAD.
# ===================================================================================
Fdiv = D_max / D_max            # En este caso, Fdiv = 1, Porque hay una carga

# ===================================================================================
#                                FACTOR DE COINCIDENCIA.
# ===================================================================================
Fcoinc = 1 / Fdiv          # En este caso, Fcoinc = 1, Porque hay una carga

# ===================================================================================
#                 TIEMPO MÁXIMO DE OPERACIÓN A DEMANDA MÁXIMA EN HORAS/AÑO.       
# ===================================================================================
TiempoMax = (Fc)*8760  # Tiempo máximo de operación a demanda máxima (horas/año)


# ***********************************************************************************
#                    APROXIMAMOS A 4 DECIMALES LOS RESULTADOS
# ***********************************************************************************

D_max = round(D_max, 4)
demanda_promedio = round(demanda_promedio, 4)
D_div = round(D_div, 4)
DCM = round(DCM, 4)
DnCM = round(DnCM, 4)
Fd = round(Fd, 4)
Fu = round(Fu, 4)
Fc = round(Fc, 4)
Fdiv = round(Fdiv, 4)
Fcoinc = round(Fcoinc, 4)
TiempoMax = round(TiempoMax, 4)

# ====================================================================================
#                CREAR DATAFRAME DE RESULTADOS Y EXPORTAR A EXCEL
# ====================================================================================

# Convertir resultados a Series para crear DataFrame
D_max = pd.Series(D_max)
demanda_promedio = pd.Series(demanda_promedio)
D_div = pd.Series(D_div)
DCM = pd.Series(DCM)
DnCM = pd.Series(DnCM)
Fd = pd.Series(Fd)
Fu = pd.Series(Fu)
Fc = pd.Series(Fc)
Fdiv = pd.Series(Fdiv)
Fcoinc = pd.Series(Fcoinc)
TiempoMax = pd.Series(TiempoMax)

# Crear DataFrame vacío para resultados
df_resultados = pd.DataFrame()

# Transferir los resultados al DataFrame
df_resultados['Dem max (kVA)'] = D_max
df_resultados['Dem prom (kW)'] = demanda_promedio
df_resultados['Dem Diversificada (kWh)'] = D_div
df_resultados['Dem Coincidente Max (kVA)'] = DCM
df_resultados['Dem No-Coincidente Max (kVA)'] = DnCM
df_resultados['Factor de Demanda'] = Fd
df_resultados['Factor de Utilización'] = Fu
df_resultados['Factor de Carga'] = Fc
df_resultados['Factor de Diversidad'] = Fdiv
df_resultados['Factor de Coincidencia'] = Fcoinc
df_resultados['Tiempo Máx Operación a Dem Max (hrs/año)'] = TiempoMax

if 'Tabla_duracion' in globals():
    fig4, ax4 = plt.subplots(figsize=(20, 8))

    x = Tabla_duracion['Porcentaje_Tiempo'].values
    y = Tabla_duracion['pot_total'].values

    # ancho de barras similar al usado antes
    x_range = x.max() - x.min() if x.max() != x.min() else 1.0
    width = (x_range / len(x)) * 0.6
    width = max(width, 0.05)

    ax4.bar(x, y, width=width, align='center', alpha=0.8, color='black', edgecolor='k', linewidth=0.2)

    # obtener valor escalar de demanda_promedio (puede ser Series)
    try:
        demanda_promedio_val = float(demanda_promedio.iloc[0])
    except Exception:
        demanda_promedio_val = float(demanda_promedio)

    # línea horizontal de la demanda promedio (color visible y label)
    ax4.axhline(y=demanda_promedio_val, color='red', linestyle='--', linewidth=2,
                label=f'Demanda promedio = {demanda_promedio_val:.2f} kW')

    # ejes y formato
    ax4.set_xlim(0, 100)  # opcional: fija 0..100% en x
    y_max = int(np.ceil(y.max())) if len(y) > 0 else 1
    ax4.set_ylim(0, max(y_max, 1) * 1.05)
    ax4.set_xticks(np.arange(0, 101, 10))
    ax4.set_yticks(np.arange(0, y_max + 1, 1))

    ax4.set_title('Curva de Duración de la Carga con Demanda Promedio', fontsize=14, fontweight='bold')
    ax4.set_ylabel('kVA')
    ax4.set_xlabel('Porcentaje de Tiempo (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    fig4.savefig('curva_duracion_con_promedio.png', dpi=300)
    plt.show()

# Exportar el DataFrame de resultados a Excel
output_path = 'Resultados_casa.xlsx'
try:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_resultados.to_excel(writer, sheet_name='Detalle', index=False)
except Exception as e:
    print(f"Error al exportar df_resultados a Excel: {e}")







