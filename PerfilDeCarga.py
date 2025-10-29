import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from scipy import stats


# ====================================================================================
Perfil_de_carga = 'N'  # Condición para generar perfil de carga (Y/N)
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

# =================================================================
# PERFIL DE CARGA, por fase y total.
# =================================================================

if Perfil_de_carga == 'Y':

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

# =================================================================
# CURVA DE DURACIÓN DE LA CARGA (CDC) y tabla energía vs % tiempo
# =================================================================
if Curva_de_duracion == 'Y':
    
    df['pot_total'] = pot_total

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
    
    



