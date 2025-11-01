import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# --------------------------------------------------------------------------------------------------------
#                       Configuración: Cambia 'Y' para ejecutar las funciones correspondientes.
# --------------------------------------------------------------------------------------------------------
Perfil_de_carga = 'Y'  # Cambia a 'Y' para ejecutar la función.
Curva_Duracion = 'Y'  # Cambia a 'Y' para ejecutar la función.
DemandaMaxProm = 'Y'  # Cambia a 'Y' para ejecutar la función.
BusA = 'Y'  # Cambia a 'Y' para ejecutar la función.
BusB = 'Y'  # Cambia a 'Y' para ejecutar la función.
ParametrosSubestacion = 'Y'  # Cambia a 'Y' para ejecutar la función.


# =======================================================================================================   
#                               Extraemos los datos de cada circuito de distribución.
# =======================================================================================================
dfA01 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_A01.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True,  # Maneja cualquier espacio/tab
                 skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfA02 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_A02.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfA03 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_A03.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfA04 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_A04.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfA05 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_A05.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfB01 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_B01.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfB02 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_B02.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfB03 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_B03.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

dfB04 = pd.read_csv(r'C:\Users\Usuario\Desktop\Universidad\15 - Sep - Dic 2025\Distribución\Tarea3\PLACER Jul 2005\PLA_B04.AMP_C-HAVG-Jul.txt', 
                 delim_whitespace=True, skiprows=2, names=['Valor', 'Fecha', 'Hora'])

print ()
# =======================================================================================================
#                              Seleccionamos solo la columna de interés (Potencias en KVA).
# =======================================================================================================
KVA_A01 = dfA01[['Valor']].values
KVA_A02 = dfA02[['Valor']].values
KVA_A03 = dfA03[['Valor']].values
KVA_A04 = dfA04[['Valor']].values
KVA_A05 = dfA05[['Valor']].values

KVA_B01 = dfB01[['Valor']].values
KVA_B02 = dfB02[['Valor']].values
KVA_B03 = dfB03[['Valor']].values
KVA_B04 = dfB04[['Valor']].values

# =======================================================================================================
#                     Extraemos las horas de medición para su posterior uso.
# =======================================================================================================
Horas_A01 = dfA01[['Hora']].values
Horas_A02 = dfA02[['Hora']].values
Horas_A03 = dfA03[['Hora']].values
Horas_A04 = dfA04[['Hora']].values
Horas_A05 = dfA05[['Hora']].values

Horas_B01 = dfB01[['Hora']].values
Horas_B02 = dfB02[['Hora']].values
Horas_B03 = dfB03[['Hora']].values
Horas_B04 = dfB04[['Hora']].values

# =======================================================================================================
#                     Extraemos las fechas de medición para su posterior uso.
# =======================================================================================================
Fechas_A01 = dfA01[['Fecha']].values
Fechas_A02 = dfA02[['Fecha']].values
Fechas_A03 = dfA03[['Fecha']].values
Fechas_A04 = dfA04[['Fecha']].values
Fechas_A05 = dfA05[['Fecha']].values

Fechas_B01 = dfB01[['Fecha']].values
Fechas_B02 = dfB02[['Fecha']].values
Fechas_B03 = dfB03[['Fecha']].values
Fechas_B04 = dfB04[['Fecha']].values

# =======================================================================================================
#                                Creamos una función para obtener el perfil de carga.
# =======================================================================================================

def PerfilDeCarga(Nombre_, KVA_, Horas_, Fechas_):
    KVA = np.asarray(KVA_).astype(float).squeeze()       # potencia
    HOR = np.asarray(Horas_).astype(str).squeeze()       # horas como strings
    FCH_raw = np.asarray(Fechas_).astype(str).squeeze()  # fechas como strings (mantener original)

    # convertir fechas a datetime para comparar días (no sobreescribir FCH_raw)
    FCH_dt = pd.to_datetime(FCH_raw, dayfirst=True, errors='coerce')


    # -------------------------------------------------------------------------------------------------------
    #                        Sumar Valores Diarios y Eliminar Valores Correspondientes.
    # -------------------------------------------------------------------------------------------------------
    i = 0
    p = 0
    # iterar por posiciones; usar while porque la longitud cambia al borrar
    while i < len(KVA) - 1:
        # si cualquiera de las dos fechas es NaT -> considerar cambio de día (no sumar)
        if pd.isna(FCH_dt[i]) or pd.isna(FCH_dt[i+1]) or (FCH_dt[i].date() != FCH_dt[i+1].date()):
            i += 1
            continue

        # mismo día -> sumar siguiente al actual y eliminar la fila siguiente en todos los arrays
        KVA[i] = KVA[i] + KVA[i+1]
        KVA = np.delete(KVA, i+1, axis=0)
        HOR = np.delete(HOR, i+1, axis=0)
        FCH_raw = np.delete(FCH_raw, i+1, axis=0)
        FCH_dt = np.delete(FCH_dt, i+1, axis=0)
        p += 1
        # no incrementar i: evaluar la nueva fila i+1 en la siguiente iteración

    # Reconstruir arrays con forma (n,1) — Fechas_A01 con los strings originales (o convertir a datetime si prefieres)
    KVA_ = KVA.reshape(-1, 1)
    Horas_ = HOR.reshape(-1, 1)
    Fechas_ = FCH_raw.reshape(-1, 1)            # mantiene el formato original "DD/MM/YYYY"
    
    # Preparar arrays (asegurar forma 1D)
    kva = np.asarray(KVA_).squeeze().astype(float)
    fch = np.asarray(Fechas_).squeeze().astype(str)
    horas = np.asarray(Horas_).squeeze().astype(str)   # <-- obtener horas también


    # crear fechas (solo la parte de fecha, sin horas) y datetime completo
    fechas_dt = pd.to_datetime(fch, dayfirst=True, errors='coerce').normalize()
    dt_full = pd.to_datetime(fch + ' ' + horas, dayfirst=True, errors='coerce')  # datetime con hora

    # filtrar valores válidos (fecha válida y potencia no NaN)
    mask = ~pd.isna(fechas_dt) & ~pd.isna(kva) & ~pd.isna(dt_full)
    fechas_dt = fechas_dt[mask]
    kva = kva[mask]
    dt_full = dt_full[mask]
    
    # gráfica: Potencia vs Fecha (sin horas en eje principal)
    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(fechas_dt, kva, color='black', linewidth=0.8, marker='o', markersize=3)
    ax.set_ylabel('kVA')
    ax.set_title('Potencia vs Fecha',fontsize=14, fontweight='bold')

    # formatear eje X para mostrar solo fechas (día) — ticks diarios
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # solo el dígito del día
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Para cada día obtener la última medición (max dt_full) y su hora
    df_aux = pd.DataFrame({'fecha': fechas_dt, 'dt': dt_full})
    agrup = df_aux.groupby(df_aux['fecha']).agg(ultimo_dt=('dt', 'max')).reset_index()

    # etiquetas hora (HH:MM) y fecha (formato dd/mm/YYYY)
    # posiciones las usamos sólo para ticks; no colocar texto con ax.text
    posiciones = agrup['ultimo_dt'].tolist()
    labels = [f"{t.strftime('%H:%M')}\n{t.strftime('%d/%m/%Y')}" for t in agrup['ultimo_dt']]

    # Asignar ticks y etiquetas: cada tick muestra hora en la primera línea y fecha en la segunda (rotado 90°)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)

    ax.grid(True, alpha=0.3)

    # Nombre variable (cámbialo antes de ejecutar)
    Nombre = Nombre_
    _invalid = '<>:"/\\|?*'
    safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(Nombre))
    filename = f"PerfilDeCarga_{safe_name}.jpg"
    output_path = os.path.join(os.getcwd(), filename)

    # guardar figura
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    
    return KVA_, Horas_, Fechas_

# =======================================================================================================
#                                Llamamos a la función para cada circuito.
# =======================================================================================================

if Perfil_de_carga == 'Y':
    KVA_A01_pc, Horas_A01_pc, Fechas_A01_pc = PerfilDeCarga('Circuito A01', KVA_A01, Horas_A01, Fechas_A01)
    KVA_A02_pc, Horas_A02_pc, Fechas_A02_pc = PerfilDeCarga('Circuito A02', KVA_A02, Horas_A02, Fechas_A02)
    KVA_A03_pc, Horas_A03_pc, Fechas_A03_pc = PerfilDeCarga('Circuito A03', KVA_A03, Horas_A03, Fechas_A03)
    KVA_A04_pc, Horas_A04_pc, Fechas_A04_pc = PerfilDeCarga('Circuito A04', KVA_A04, Horas_A04, Fechas_A04)
    KVA_A05_pc, Horas_A05_pc, Fechas_A05_pc = PerfilDeCarga('Circuito A05', KVA_A05, Horas_A05, Fechas_A05)


    KVA_B01_pc, Horas_B01_pc, Fechas_B01_pc = PerfilDeCarga('Circuito B01', KVA_B01, Horas_B01, Fechas_B01)
    KVA_B02_pc, Horas_B02_pc, Fechas_B02_pc = PerfilDeCarga('Circuito B02', KVA_B02, Horas_B02, Fechas_B02)
    KVA_B03_pc, Horas_B03_pc, Fechas_B03_pc = PerfilDeCarga('Circuito B03', KVA_B03, Horas_B03, Fechas_B03)
    KVA_B04_pc, Horas_B04_pc, Fechas_B04_pc = PerfilDeCarga('Circuito B04', KVA_B04, Horas_B04, Fechas_B04)

# =======================================================================================================
#                     Creamos una función para obtener la curva de duración.
# =======================================================================================================

def CurvaDeDuracion(Nombre_, KVA_, Horas_, Fechas_):
    
    kva_arr = np.asarray(KVA_).ravel().astype(float)
    fechas_arr = np.asarray(Fechas_).ravel().astype(str)
    horas_arr  = np.asarray(Horas_).ravel().astype(str)

    DT_Orden = pd.DataFrame()  # DataFrame vacío para almacenar resultados si es necesario
    DT_Orden = pd.DataFrame({'KVA': kva_arr, 'Fecha': fechas_arr,'Hora': horas_arr})
    DT_Orden = DT_Orden.sort_values(by='KVA', ascending=False).reset_index(drop=True)

    # Calcular probabilidad de excedencia (porcentaje de tiempo)
    n = len(DT_Orden)
    DT_Orden['Orden'] = range(1, n + 1)
    DT_Orden['Porcentaje_Tiempo'] = (DT_Orden['Orden'] / n) * 100

    # Crear tabla resumen
    tabla_duracion = DT_Orden.copy()

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    # datos para graficar
    x = tabla_duracion['Porcentaje_Tiempo'].values
    y = tabla_duracion['KVA'].values

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

    # Nombre variable (cámbialo antes de ejecutar)
    Nombre = Nombre_
    _invalid = '<>:"/\\|?*'
    safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(Nombre))
    filename = f"CurvaDeDuracion_{safe_name}.jpg"
    output_path = os.path.join(os.getcwd(), filename)

    # guardar figura
    fig3.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()


if Curva_Duracion == 'Y':
    CurvaDeDuracion('Circuito A01', KVA_A01_pc, Horas_A01_pc, Fechas_A01_pc)
    CurvaDeDuracion('Circuito A02', KVA_A02_pc, Horas_A02_pc, Fechas_A02_pc)
    CurvaDeDuracion('Circuito A03', KVA_A03_pc, Horas_A03_pc, Fechas_A03_pc)
    CurvaDeDuracion('Circuito A04', KVA_A04_pc, Horas_A04_pc, Fechas_A04_pc)
    CurvaDeDuracion('Circuito A05', KVA_A05_pc, Horas_A05_pc, Fechas_A05_pc)

    CurvaDeDuracion('Circuito B01', KVA_B01_pc, Horas_B01_pc, Fechas_B01_pc)
    CurvaDeDuracion('Circuito B02', KVA_B02_pc, Horas_B02_pc, Fechas_B02_pc)
    CurvaDeDuracion('Circuito B03', KVA_B03_pc, Horas_B03_pc, Fechas_B03_pc)
    CurvaDeDuracion('Circuito B04', KVA_B04_pc, Horas_B04_pc, Fechas_B04_pc)

# =======================================================================================================
#                                Calculamos la demanda máxima y promedio con una función.
# =======================================================================================================
def DemandaMaxPromedio(KVA_pc, KVA_, Horas_, Fechas_):

    Pot_MAX = int(max(KVA_pc))             # Potencia Máx.

    # -------------------------------------------------------------------------------------------------------
    #                      Cálculo de Demanda Promedio.
    # -------------------------------------------------------------------------------------------------------

    # Convertimos las variables para trabajar mejor.
    KVA = np.asarray(KVA_).astype(float).squeeze()       # potencia
    HOR = np.asarray(Horas_).astype(str).squeeze()       # horas como strings
    FCH_raw = np.asarray(Fechas_).astype(str).squeeze()  # fechas como strings (mantener original)

    # Convertir fechas a datetime para comparar días (no sobreescribir FCH_raw)
    FCH_dt = pd.to_datetime(FCH_raw, dayfirst=True, errors='coerce')

    # Creamos un dataFrame para organizar mejor los valores.
    df = pd.DataFrame()
    Horas_medidas = []
    i = 0
    p = 0
    # iterar por posiciones; usar while porque la longitud cambia al borrar
    while i < len(KVA_) - 1:
        # si cualquiera de las dos fechas es NaT -> considerar cambio de día (no sumar)
        if pd.isna(FCH_dt[i]) or pd.isna(FCH_dt[i+1]) or (FCH_dt[i].date() != FCH_dt[i+1].date()):
            i += 1   
            
            if p == 0:
                p = 1 
                Horas_medidas.append(p) 
                
            else:
                Horas_medidas.append(p)        
            p = 0
            continue

        # mismo día -> sumar siguiente al actual y eliminar la fila siguiente en todos los arrays
        KVA_[i] = KVA_[i] + KVA_[i+1]
        KVA_ = np.delete(KVA_, i+1, axis=0)
        HOR = np.delete(HOR, i+1, axis=0)
        FCH_raw = np.delete(FCH_raw, i+1, axis=0)
        FCH_dt = np.delete(FCH_dt, i+1, axis=0)
        p += 1
        # no incrementar i: evaluar la nueva fila i+1 en la siguiente iteración
        
    Horas_medidas.append(p) # Guardamos el último números de horas.

    # Reconstruir arrays con forma (n,1) — Fechas_A01 con los strings originales (o convertir a datetime si prefieres)
    KVA_ = KVA_.reshape(-1, 1)
    Horas_ = HOR.reshape(-1, 1)
    Fechas_ = FCH_raw.reshape(-1, 1)            # mantiene el formato original "DD/MM/YYYY"

    # Preparar arrays (asegurar forma 1D)
    kva = np.asarray(KVA_).squeeze().astype(float)
    fch = np.asarray(Fechas_).squeeze().astype(str)
    horas = np.asarray(Horas_).squeeze().astype(str)   # <-- obtener horas también


    # Construimos DataFrame con una fila por día y la cantidad registrada
    df = pd.DataFrame({'Potencia': kva, 'Hora Medidas': Horas_medidas})

    # 2. Calculamos el tiempo de mediciones.
    intervalo_horas = df['Hora Medidas'].sum()

    # 3. Calculamos la energía consumida en kWh
    df['Energia_kWh'] = df['Potencia'] * intervalo_horas

    # 4. Calcular energía total consumida
    energia_total = df['Energia_kWh'].sum()

    # 5. Calcular tiempo total
    tiempo_total = len(df)*intervalo_horas  # N* de mediciones x 621 horas

    # 6. Aplicar la fórmula de demanda promedio
    demanda_promedio = energia_total / tiempo_total

    return Pot_MAX, demanda_promedio, Horas_medidas

if DemandaMaxProm == 'Y':

    Pot_MAX_A01, Dem_Prom_A01, Horas_medidas_A01 = DemandaMaxPromedio(KVA_A01_pc, KVA_A01, Horas_A01, Fechas_A01)
    Pot_MAX_A02, Dem_Prom_A02, Horas_medidas_A02 = DemandaMaxPromedio(KVA_A02_pc, KVA_A02, Horas_A02, Fechas_A02)
    Pot_MAX_A03, Dem_Prom_A03, Horas_medidas_A03 = DemandaMaxPromedio(KVA_A03_pc, KVA_A03, Horas_A03, Fechas_A03)
    Pot_MAX_A04, Dem_Prom_A04, Horas_medidas_A04 = DemandaMaxPromedio(KVA_A04_pc, KVA_A04, Horas_A04, Fechas_A04)
    Pot_MAX_A05, Dem_Prom_A05, Horas_medidas_A05 = DemandaMaxPromedio(KVA_A05_pc, KVA_A05, Horas_A05, Fechas_A05)

    Pot_MAX_B01, Dem_Prom_B01, Horas_medidas_B01 = DemandaMaxPromedio(KVA_B01_pc, KVA_B01, Horas_B01, Fechas_B01)
    Pot_MAX_B02, Dem_Prom_B02, Horas_medidas_B02 = DemandaMaxPromedio(KVA_B02_pc, KVA_B02, Horas_B02, Fechas_B02)
    Pot_MAX_B03, Dem_Prom_B03, Horas_medidas_B03 = DemandaMaxPromedio(KVA_B03_pc, KVA_B03, Horas_B03, Fechas_B03)
    Pot_MAX_B04, Dem_Prom_B04, Horas_medidas_B04 = DemandaMaxPromedio(KVA_B04_pc, KVA_B04, Horas_B04, Fechas_B04)


# =======================================================================================================
#                        Calculo de Caracteristicas de los Buses.
# =======================================================================================================

def CaracteristicasBusA(Pot_MAX_A01, Pot_MAX_A02, Pot_MAX_A03, Pot_MAX_A04, Pot_MAX_A05):
    # =======================================================================================================
    #                        Cálculo de Demanda Máxima Diversificada (BUS A).
    # =======================================================================================================
    # Lista de demandas máximas de cada circuito.
    DemandasMax = [Pot_MAX_A01, Pot_MAX_A02, Pot_MAX_A03, Pot_MAX_A04, Pot_MAX_A05] 

    # Demanda máxima del sistema (mayor entre todas las demandas máximas de los circuitos).
    DemMaxs = max(DemandasMax)

    i = 0
    circuito_max = 0

    # iterar por posiciones; usar while porque la longitud cambia al borrar.
    while i < len(DemandasMax) - 1:
            # si cualquiera de las dos fechas es NaT -> considerar cambio de día (no sumar)
            if DemandasMax[i] == DemMaxs:
                i += 1
                continue
            
            else:
                circuito_max += DemandasMax[i]
                i += 1

    # DEMANDA DIVERSIFICADA.
    D_Div = (circuito_max - DemMaxs)

    # =======================================================================================================
    #                                    Demanda Coincidente máxima (BUS A).
    # =======================================================================================================
    Dem_Coinc = DemMaxs

    # =======================================================================================================
    #                                    Demanda Coincidente máxima (BUS A).
    # =======================================================================================================
    Dem_Coinc = DemMaxs

    # =======================================================================================================
    #                                    Demanda Coincidente máxima (BUS A).
    # =======================================================================================================
    Dem_No_Coinc = circuito_max + DemMaxs

    # =======================================================================================================
    #                                    Factor de Diversidad (BUS A).
    # =======================================================================================================
    F_Div = circuito_max / (DemMaxs)

    # =======================================================================================================
    #                                    Factor de Coincidencia (BUS A).
    # =======================================================================================================
    F_Coinc = 1/F_Div
    
    return DemMaxs, D_Div, Dem_Coinc, Dem_No_Coinc, F_Div, F_Coinc

def CaracteristicasBusB(Pot_MAX_B01, Pot_MAX_B02, Pot_MAX_B03, Pot_MAX_B04):
    # =======================================================================================================
    #                        Cálculo de Demanda Máxima Diversificada (BUS B).
    # =======================================================================================================
    # Lista de demandas máximas de cada circuito.
    DemandasMax = [Pot_MAX_B01, Pot_MAX_B02, Pot_MAX_B03, Pot_MAX_B04] 

    # Demanda máxima del sistema (mayor entre todas las demandas máximas de los circuitos).
    DemMaxs = max(DemandasMax)

    i = 0
    circuito_max = 0

    # iterar por posiciones; usar while porque la longitud cambia al borrar.
    while i < len(DemandasMax) - 1:
            # si cualquiera de las dos fechas es NaT -> considerar cambio de día (no sumar)
            if DemandasMax[i] == DemMaxs:
                i += 1
                continue
            
            else:
                circuito_max += DemandasMax[i]
                i += 1

    # DEMANDA DIVERSIFICADA.
    D_Div = (circuito_max - DemMaxs)

    # =======================================================================================================
    #                                    Demanda Coincidente máxima (BUS B).
    # =======================================================================================================
    Dem_Coinc = DemMaxs

    # =======================================================================================================
    #                                    Demanda Coincidente máxima (BUS B).
    # =======================================================================================================
    Dem_Coinc = DemMaxs

    # =======================================================================================================
    #                                    Demanda No Coincidente máxima (BUS B).
    # =======================================================================================================
    Dem_No_Coinc = circuito_max + DemMaxs

    # =======================================================================================================
    #                                    Factor de Diversidad (BUS B).
    # =======================================================================================================
    F_Div = circuito_max / (DemMaxs)

    # =======================================================================================================
    #                                    Factor de Coincidencia (BUS B).
    # =======================================================================================================
    F_Coinc = 1/F_Div
    
    return DemMaxs, D_Div, Dem_Coinc, Dem_No_Coinc, F_Div, F_Coinc

if BusA == 'Y':
    # BUS A: Circuitos A01, A02, A03, A04, A05.
    DemMaxs_BUS_A, D_Div_BUS_A, Dem_Coinc_BUS_A, Dem_No_Coinc_BUS_A, F_Div_BUS_A, F_Coinc_BUS_A = CaracteristicasBusA(Pot_MAX_A01, Pot_MAX_A02, Pot_MAX_A03, Pot_MAX_A04, Pot_MAX_A05)

if BusB == 'Y':  
    # BUS B: Circuitos B01, B02, B03, B04.
    DemMaxs_BUS_B, D_Div_BUS_B, Dem_Coinc_BUS_B, Dem_No_Coinc_BUS_B, F_Div_BUS_B, F_Coinc_BUS_B = CaracteristicasBusB(Pot_MAX_B01, Pot_MAX_B02, Pot_MAX_B03, Pot_MAX_B04)    

# =======================================================================================================
#                                CAPACIDAD INSTALADA DE CADA CIRCUITO.
# =======================================================================================================
CTC_A01 = 7130        # Capacidad Instalada del circuito A01 en kVA
CTC_A02 = 7270        # Capacidad Instalada del circuito A02 en kVA
CTC_A03 = 1915        # Capacidad Instalada del circuito A03 en kVA
CTC_A04 = 7297.5      # Capacidad Instalada del circuito A04 en kVA
CTC_A05 = 13450.5     # Capacidad Instalada del circuito A05 en kVA

CTC_B01 = 2770        # Capacidad Instalada del circuito B01 en kVA
CTC_B02 = 3682.2      # Capacidad Instalada del circuito B02 en kVA
CTC_B03 = 10715.5     # Capacidad Instalada del circuito B03 en kVA
CTC_B04 = 12345.0     # Capacidad Instalada del circuito B04 en kVA


# =======================================================================================================
#                                     CALCULAMOS LOS PARAMETROS CARACTERISTICOS.
# =======================================================================================================

def CaracteristicasSubestacion(Pot_MAX_, Dem_Prom_, CTC_):
    
    # =======================================================================================================
    #                                     FACTOR DE DEMANDA.
    # =======================================================================================================

    Fd = 0 # Factor de Demanda (No tratamos demanda en este caso).

    # =======================================================================================================
    #                                     FACTOR DE UTILIZACIÓN.
    # =======================================================================================================
    Fu = Pot_MAX_/CTC_

    # =======================================================================================================
    #                                     FACTOR DE CARGA.
    # =======================================================================================================
    Fc = Dem_Prom_ / Pot_MAX_

    # =======================================================================================================
    #                                    TIEMPO MAXIMO DE USO EN UN AñO.
    # =======================================================================================================
    T_max = 8760 * Fc
    
    return Fd, Fu, Fc, T_max

# =======================================================================================================
#                                Llamamos a la función para cada circuito.
# =======================================================================================================

if ParametrosSubestacion == 'Y':
    Fd_A01, Fu_A01, Fc_A01, T_max_A01 = CaracteristicasSubestacion(Pot_MAX_A01, Dem_Prom_A01, CTC_A01)
    Fd_A02, Fu_A02, Fc_A02, T_max_A02 = CaracteristicasSubestacion(Pot_MAX_A02, Dem_Prom_A02, CTC_A02)
    Fd_A03, Fu_A03, Fc_A03, T_max_A03 = CaracteristicasSubestacion(Pot_MAX_A03, Dem_Prom_A03, CTC_A03)
    Fd_A04, Fu_A04, Fc_A04, T_max_A04 = CaracteristicasSubestacion(Pot_MAX_A04, Dem_Prom_A04, CTC_A04)
    Fd_A05, Fu_A05, Fc_A05, T_max_A05 = CaracteristicasSubestacion(Pot_MAX_A05, Dem_Prom_A05, CTC_A05)

    Fd_B01, Fu_B01, Fc_B01, T_max_B01 = CaracteristicasSubestacion(Pot_MAX_B01, Dem_Prom_B01, CTC_B01)
    Fd_B02, Fu_B02, Fc_B02, T_max_B02 = CaracteristicasSubestacion(Pot_MAX_B02, Dem_Prom_B02, CTC_B02)
    Fd_B03, Fu_B03, Fc_B03, T_max_B03 = CaracteristicasSubestacion(Pot_MAX_B03, Dem_Prom_B03, CTC_B03)
    Fd_B04, Fu_B04, Fc_B04, T_max_B04 = CaracteristicasSubestacion(Pot_MAX_B04, Dem_Prom_B04, CTC_B04)

if Perfil_de_carga == 'Y':
    # =======================================================================================================
    #                                     GRAFICAMOS LAS BARRAS.
    # =======================================================================================================

    def Perfil_De_Carga_Bus_A(nombre_salida='Bus A', base_fila='A01'):
        """
        Usa como referencia las fechas/horas de la subruta base_fila (p. ej. 'A01') para los ticks X
        y dibuja KVA_*_pc de cada circuito (si existen) con colores distintos.
        """
        # lista de circuitos esperados (ajustar si necesitas otros nombres)
        circuitos = [
            ('A01', KVA_A01_pc, Horas_A01_pc, Fechas_A01_pc),
            ('A02', KVA_A02_pc, Horas_A02_pc, Fechas_A02_pc),
            ('A03', KVA_A03_pc, Horas_A03_pc, Fechas_A03_pc),
            ('A04', KVA_A04_pc, Horas_A04_pc, Fechas_A04_pc),
            ('A05', KVA_A05_pc, Horas_A05_pc, Fechas_A05_pc),
        ]

        # colores a rotar
        colores = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']

        # construir posiciones/etiquetas X a partir del circuito base (por defecto A01)
        base = next((c for c in circuitos if c[0] == base_fila), None)
        if base is None:
            raise RuntimeError(f"No se encontró circuito base {base_fila} para ticks X.")
        _, _, base_horas, base_fechas = base
        base_f = np.asarray(base_fechas).ravel().astype(str)
        base_h = np.asarray(base_horas).ravel().astype(str)
        base_dt = pd.to_datetime(base_f + ' ' + base_h, dayfirst=True, errors='coerce')
        df_base = pd.DataFrame({'fecha': base_dt.normalize(), 'dt': base_dt}).dropna()
        agrup = df_base.groupby('fecha').agg(ultimo_dt=('dt','max')).reset_index()
        posiciones = agrup['ultimo_dt'].tolist()
        labels = [f"{t.strftime('%H:%M')}\n{t.strftime('%d/%m/%Y')}" for t in agrup['ultimo_dt']]

        # figura
        fig, ax = plt.subplots(figsize=(18,6))

        leyendas = []
        idx_color = 0
        for tag, KVA_pc, Hor_pc, Fech_pc in circuitos:
            # comprobar existencia (variables pueden no estar definidas si no se ejecutó Perfil_de_carga)
            try:
                kva = np.asarray(KVA_pc).ravel().astype(float)
                fch = np.asarray(Fech_pc).ravel().astype(str)
            except Exception:
                continue
            # fechas por fila (normalizadas)
            fechas_dt = pd.to_datetime(fch, dayfirst=True, errors='coerce').normalize()
            # filtrar según fechas válidas
            mask = ~pd.isna(fechas_dt) & ~np.isnan(kva)
            if mask.sum() == 0:
                continue
            fechas_dt = fechas_dt[mask]
            kva = kva[mask]

            color = colores[idx_color % len(colores)]
            ax.plot(fechas_dt, kva, marker='o', linewidth=1.0, markersize=3, color=color)
            leyendas.append(tag)
            idx_color += 1

        # ticks X con hora+fecha en dos líneas, rotadas 90°
        ax.set_xticks(posiciones)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)

        ax.set_ylabel('kVA')
        ax.set_title('Perfiles de Carga por circuito Bus A.')
        ax.grid(True, alpha=0.3)
        ax.legend(leyendas, loc='upper right')

        # guardar (normalizar nombre)
        _invalid = '<>:"/\\|?*'
        safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(nombre_salida))
        filename = f"{safe_name}.jpg"
        output_path = os.path.join(os.getcwd(), filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()

    def Perfil_De_Carga_Bus_B(nombre_salida='Bus B', base_fila='B01'):
        """
        Usa como referencia las fechas/horas de la subruta base_fila (p. ej. 'A01') para los ticks X
        y dibuja KVA_*_pc de cada circuito (si existen) con colores distintos.
        """
        # lista de circuitos esperados (ajustar si necesitas otros nombres)
        circuitos = [
            ('B01', KVA_B01_pc, Horas_B01_pc, Fechas_B01_pc),
            ('B02', KVA_B02_pc, Horas_B02_pc, Fechas_B02_pc),
            ('B03', KVA_B03_pc, Horas_B03_pc, Fechas_B03_pc),
            ('B04', KVA_B04_pc, Horas_B04_pc, Fechas_B04_pc),
        ]

        # colores a rotar
        colores = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']

        # construir posiciones/etiquetas X a partir del circuito base (por defecto A01)
        base = next((c for c in circuitos if c[0] == base_fila), None)
        if base is None:
            raise RuntimeError(f"No se encontró circuito base {base_fila} para ticks X.")
        _, _, base_horas, base_fechas = base
        base_f = np.asarray(base_fechas).ravel().astype(str)
        base_h = np.asarray(base_horas).ravel().astype(str)
        base_dt = pd.to_datetime(base_f + ' ' + base_h, dayfirst=True, errors='coerce')
        df_base = pd.DataFrame({'fecha': base_dt.normalize(), 'dt': base_dt}).dropna()
        agrup = df_base.groupby('fecha').agg(ultimo_dt=('dt','max')).reset_index()
        posiciones = agrup['ultimo_dt'].tolist()
        labels = [f"{t.strftime('%H:%M')}\n{t.strftime('%d/%m/%Y')}" for t in agrup['ultimo_dt']]

        # figura
        fig, ax = plt.subplots(figsize=(18,6))

        leyendas = []
        idx_color = 0
        for tag, KVA_pc, Hor_pc, Fech_pc in circuitos:
            # comprobar existencia (variables pueden no estar definidas si no se ejecutó Perfil_de_carga)
            try:
                kva = np.asarray(KVA_pc).ravel().astype(float)
                fch = np.asarray(Fech_pc).ravel().astype(str)
            except Exception:
                continue
            # fechas por fila (normalizadas)
            fechas_dt = pd.to_datetime(fch, dayfirst=True, errors='coerce').normalize()
            # filtrar según fechas válidas
            mask = ~pd.isna(fechas_dt) & ~np.isnan(kva)
            if mask.sum() == 0:
                continue
            fechas_dt = fechas_dt[mask]
            kva = kva[mask]

            color = colores[idx_color % len(colores)]
            ax.plot(fechas_dt, kva, marker='o', linewidth=1.0, markersize=3, color=color)
            leyendas.append(tag)
            idx_color += 1

        # ticks X con hora+fecha en dos líneas, rotadas 90°
        ax.set_xticks(posiciones)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)

        ax.set_ylabel('kVA')
        ax.set_title('Perfiles de Carga por circuito Bus B.')
        ax.grid(True, alpha=0.3)
        ax.legend(leyendas, loc='upper right')

        # guardar (normalizar nombre)
        _invalid = '<>:"/\\|?*'
        safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(nombre_salida))
        filename = f"{safe_name}.jpg"
        output_path = os.path.join(os.getcwd(), filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        
    Perfil_De_Carga_Bus_A('Perfiles_Carga_Bus_A', base_fila='A01')
    Perfil_De_Carga_Bus_B('Perfiles_Carga_Bus_B', base_fila='B01')

    # =======================================================================================================
    #                     Creamos una función para obtener el perfil de carga.
    # =======================================================================================================


    # Juntamos todas las potencias en una sola.
    KVA_BUS_A = KVA_A01_pc + KVA_A02_pc + KVA_A03_pc + KVA_A04_pc + KVA_A05_pc
    KVA_BUS_B = KVA_B01_pc + KVA_B02_pc + KVA_B03_pc + KVA_B04_pc

    def Plot_KVA_BUS_A(Nombre='KVA_BUS_A'):
        """
        Grafica KVA_BUS_A (suma de A01..A05 procesadas) vs fecha.
        Usa Fechas_A01_pc y Horas_A01_pc para posicionar ticks con 'HH:MM' y 'DD/MM/YYYY'.
        """
        # construir KVA_BUS_A a partir de las series procesadas (ajustar si faltan circuitos)
        arrs = []
        for v in (KVA_A01_pc, KVA_A02_pc, KVA_A03_pc, KVA_A04_pc, KVA_A05_pc):
            arrs.append(np.asarray(v).ravel().astype(float))
        # truncar al mínimo común
        minlen = min(len(a) for a in arrs)
        arrs = [a[:minlen] for a in arrs]
        KVA_BUS_A = sum(arrs)

        # usar Fechas/Horas de A01 procesadas para los ticks (truncadas al mismo largo)
        fch = np.asarray(Fechas_A01_pc).ravel().astype(str)[:minlen]
        hor = np.asarray(Horas_A01_pc).ravel().astype(str)[:minlen]

        # datetimes
        dt_full = pd.to_datetime(fch + ' ' + hor, dayfirst=True, errors='coerce')
        fechas_dt = pd.to_datetime(fch, dayfirst=True, errors='coerce').normalize()

        # filtrar válidos
        mask = ~pd.isna(fechas_dt) & ~pd.isna(KVA_BUS_A) & ~pd.isna(dt_full)
        fechas_dt = fechas_dt[mask]
        kva = KVA_BUS_A[mask]
        dt_full = dt_full[mask]

        # posiciones para ticks: última hora de cada día
        df_aux = pd.DataFrame({'fecha': fechas_dt, 'dt': dt_full})
        agrup = df_aux.groupby('fecha').agg(ultimo_dt=('dt', 'max')).reset_index()
        posiciones = agrup['ultimo_dt'].tolist()
        labels = [f"{t.strftime('%H:%M')}\n{t.strftime('%d/%m/%Y')}" for t in agrup['ultimo_dt']]

        # plot
        fig, ax = plt.subplots(figsize=(18,6))
        ax.plot(fechas_dt, kva, color='black', linewidth=1.2, marker='o', markersize=3)
        ax.set_ylabel('kVA')
        ax.set_title('BUS_A: Potencia vs Fecha', fontsize=14, fontweight='bold')

        # asignar ticks con hora (línea 1) y fecha (línea 2), rotados 90°
        ax.set_xticks(posiciones)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)

        ax.grid(True, alpha=0.3)
        # guardar
        _invalid = '<>:"/\\|?*'
        safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(Nombre))
        fname = f"{safe_name}_BUS_A.jpg"
        fig.savefig(os.path.join(os.getcwd(), fname), dpi=300, bbox_inches='tight')
        plt.tight_layout()
        return KVA_BUS_A

    def Plot_KVA_BUS_B(Nombre='KVA_BUS_B'):
        """
        Grafica KVA_BUS_B (suma de B01..B04 procesadas) vs fecha.
        Usa Fechas_B01_pc y Horas_B01_pc para posicionar ticks con 'HH:MM' y 'DD/MM/YYYY'.
        """
        # construir KVA_BUS_B a partir de las series procesadas (ajustar si faltan circuitos)
        arrs = []
        for v in (KVA_B01_pc, KVA_B02_pc, KVA_B03_pc, KVA_B04_pc):
            arrs.append(np.asarray(v).ravel().astype(float))
        # truncar al mínimo común
        minlen = min(len(a) for a in arrs)
        arrs = [a[:minlen] for a in arrs]
        KVA_BUS_B = sum(arrs)
        KVA_BUS_B = np.asarray(KVA_BUS_B).ravel()   # asegurar 1D

        # usar Fechas/Horas de B01 procesadas para los ticks (truncadas al mismo largo)
        fch = np.asarray(Fechas_B01_pc).ravel().astype(str)[:minlen]
        hor = np.asarray(Horas_B01_pc).ravel().astype(str)[:minlen]

        # datetimes
        dt_full = pd.to_datetime(fch + ' ' + hor, dayfirst=True, errors='coerce')
        fechas_dt = pd.to_datetime(fch, dayfirst=True, errors='coerce').normalize()

        # filtrar válidos (asegurar máscara 1D)
        mask = (~pd.isna(fechas_dt)) & (~np.isnan(KVA_BUS_B)) & (~pd.isna(dt_full))
        mask = np.asarray(mask).ravel()
        fechas_dt = fechas_dt[mask]
        kva = KVA_BUS_B[mask]
        dt_full = dt_full[mask]

        # posiciones para ticks: última hora de cada día
        df_aux = pd.DataFrame({'fecha': fechas_dt, 'dt': dt_full})
        agrup = df_aux.groupby('fecha').agg(ultimo_dt=('dt', 'max')).reset_index()
        posiciones = agrup['ultimo_dt'].tolist()
        labels = [f"{t.strftime('%H:%M')}\n{t.strftime('%d/%m/%Y')}" for t in agrup['ultimo_dt']]

        # plot
        fig, ax = plt.subplots(figsize=(18,6))
        ax.plot(fechas_dt, kva, color='black', linewidth=1.2, marker='o', markersize=3)
        ax.set_ylabel('kVA')
        ax.set_title('BUS_B: Potencia vs Fecha', fontsize=14, fontweight='bold')

        # asignar ticks con hora (línea 1) y fecha (línea 2), rotados 90°
        ax.set_xticks(posiciones)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)

        ax.grid(True, alpha=0.3)
        # guardar
        _invalid = '<>:"/\\|?*'
        safe_name = ''.join(ch if ch not in _invalid else '_' for ch in str(Nombre))
        fname = f"{safe_name}_BUS_B.jpg"
        fig.savefig(os.path.join(os.getcwd(), fname), dpi=300, bbox_inches='tight')
        plt.tight_layout()
        return KVA_BUS_B

    # Llamar función (quita/comment si no quieres ejecución inmediata)
    Plot_KVA_BUS_A('Perfil_De_Carga')
    Plot_KVA_BUS_B('Perfil_De_Carga')

    # =======================================================================================================
    #                                     CURVA DE DURACION
    # =======================================================================================================
    CurvaDeDuracion('BUS_A', KVA_BUS_A, Horas_A01_pc, Fechas_A01_pc)
    CurvaDeDuracion('BUS_B', KVA_BUS_B, Horas_B01_pc, Fechas_B01_pc)

    # =======================================================================
    #                   POTENCIA MAX Y DEM PROMEDIO BUS A Y BUS B
    # =======================================================================

    def DemandaPromedioBus(KVA_BUS, Horas_pc):
        # asegurar vectores 1D y numéricos
        kva = np.asarray(KVA_BUS).ravel().astype(float)
        horas = np.asarray(Horas_pc).ravel().astype(int)
        # Potencia máxima
        max_bus = int(np.max(KVA_BUS))
        # Demanda promedio
        # Construimos DataFrame con una fila por día y la cantidad registrada
        df = pd.DataFrame({'Potencia': kva, 'Hora Medidas': horas})

        # 2. Calculamos el tiempo de mediciones.
        intervalo_horas = df['Hora Medidas'].sum()

        # 3. Calculamos la energía consumida en kWh
        df['Energia_kWh'] = df['Potencia'] * intervalo_horas

        # 4. Calcular energía total consumida
        energia_total = df['Energia_kWh'].sum()

        # 5. Calcular tiempo total
        tiempo_total = len(df)*intervalo_horas  # N* de mediciones x 621 horas

        # 6. Aplicar la fórmula de demanda promedio
        demanda_promedio = energia_total / tiempo_total
        
        return max_bus, demanda_promedio

    Max_Bus_A, DemandaPromedio_A = DemandaPromedioBus(KVA_BUS_A, Horas_medidas_A01)
    Max_Bus_B, DemandaPromedio_B = DemandaPromedioBus(KVA_BUS_B, Horas_medidas_B01)

    CTC_BUS_A = 3000 #30MVA llevados a KVA
    CTC_BUS_B = 3000 #30MVA llevados a KVA

    def CaracteristicasSubestacion_Barras(Pot_MAX_, Dem_Prom_, CTC_):
        
        # =======================================================================================================
        #                                     FACTOR DE DEMANDA.
        # =======================================================================================================

        Fd = 0 # Factor de Demanda (No tratamos demanda en este caso).

        # =======================================================================================================
        #                                     FACTOR DE UTILIZACIÓN.
        # =======================================================================================================
        Fu = Pot_MAX_/CTC_

        # =======================================================================================================
        #                                     FACTOR DE CARGA.
        # =======================================================================================================
        Fc = Dem_Prom_ / Pot_MAX_

        # =======================================================================================================
        #                                    TIEMPO MAXIMO DE USO EN UN AñO.
        # =======================================================================================================
        T_max = 8760 * Fc
        
        return Fd, Fu, Fc, T_max

    Fd_BusA, Fu_BusA, Fc_BusA, T_max_BusA = CaracteristicasSubestacion_Barras(Max_Bus_A, DemandaPromedio_A, CTC_BUS_A)
    Fd_BusB, Fu_BusB, Fc_BusB, T_max_BusB = CaracteristicasSubestacion_Barras(Max_Bus_B, DemandaPromedio_B, CTC_BUS_B)
    
    
def Exportar_Resultados_Excel(output_name='Resultados_Subes.xlsx'):
    # helper para obtener variable si existe
    G = globals()
    def get(name, default=None):
        return G.get(name, default)

    # etiquetas que queremos exportar por circuito
    etiquetas = [
        ('Demanda Máx.', 'Pot_Max_kVA'),
        ('Demanda Prom.', 'Demanda_Promedio'),
        ('Factor de Demanda', 'Fd'),
        ('Factor de Utilización', 'Fu'),
        ('Factor de Carga', 'Fc'),
        ('T_max (h)', 'T_max'),
        ('Cap Instalada', 'CTC')
    ]

    # construir sheet Circuitos A
    cols_A = []
    for suf in ['A01','A02','A03','A04','A05']:
        datos = {}
        # pot max
        pot = get(f'Pot_MAX_{suf}', get(f'Pot_MAX_{suf.replace("A","A")}', None))
        if pot is None:
            pot = get(f'Pot_MAX_{suf}', None)
        datos['Pot_Max_kVA'] = pot
        # demanda promedio
        datos['Demanda_Promedio'] = get(f'Dem_Prom_{suf}', None) or get(f'Dem_Prom_{suf}', None)
        # factores y T_max
        datos['Fd'] = get(f'Fd_{suf}', None)
        datos['Fu'] = get(f'Fu_{suf}', None)
        datos['Fc'] = get(f'Fc_{suf}', None)
        datos['T max'] = get(f'T_max_{suf}', None)
        # CTC
        datos['CTC'] = get(f'CTC_{suf}', None)
        # series vertical: index = etiquetas, values = datos
        s = pd.Series(datos, name=f'Circuito {suf}')
        cols_A.append(s)

    if cols_A:
        df_A = pd.concat(cols_A, axis=1)
    else:
        df_A = pd.DataFrame()

    # construir sheet Circuitos B
    cols_B = []
    for suf in ['B01','B02','B03','B04']:
        datos = {}
        datos['Pot_Max_kVA'] = get(f'Pot_MAX_{suf}', None)
        datos['Demanda_Promedio'] = get(f'Dem_Prom_{suf}', None)
        datos['Fd'] = get(f'Fd_{suf}', None)
        datos['Fu'] = get(f'Fu_{suf}', None)
        datos['Fc'] = get(f'Fc_{suf}', None)
        datos['T max'] = get(f'T_max_{suf}', None)
        datos['CTC'] = get(f'CTC_{suf}', None)
        s = pd.Series(datos, name=f'Circuito {suf}')
        cols_B.append(s)

    if cols_B:
        df_B = pd.concat(cols_B, axis=1)
    else:
        df_B = pd.DataFrame()

    # hoja Buses: datos resumidos de BUS A y BUS B
    datos_busA = {
        'Dem. Máx.': get('Max_Bus_A', get('DemMaxs_BUS_A', None)),
        'Dem. Prom': get('DemandaPromedio_A', None),
        'F. demanda': get('Fd_BusA', None),
        'F. utilización': get('Fu_BusA', None),
        'F. carga': get('Fc_BusA', None),
        'T max (h)': get('T_max_BusA', None),
        'Capacidad Instalada': get('CTC_BUS_A', None),
        'Dem. Máx.': get('DemMaxs_BUS_A', None),
        'Dem. Div': get('D_Div_BUS_A', None),
        'Dem. Coinc': get('Dem_Coinc_BUS_A', None),
        'Dem. No Coinc': get('Dem_No_Coinc_BUS_A', None),
        'F. Div': get('F_Div_BUS_A', None),
        'F. Coinc': get('F_Coinc_BUS_A', None)
    }
    datos_busB = {
        'Dem. Máx.': get('Max_Bus_B', get('DemMaxs_BUS_B', None)),
        'Dem. Prom': get('DemandaPromedio_B', None),
        'F. demanda': get('Fd_BusB', None),
        'F. utilización': get('Fu_BusB', None),
        'F. carga': get('Fc_BusB', None),
        'T max (h)': get('T_max_BusB', None),
        'Capacidad Instalada': get('CTC_BUS_B', None),
        'Dem. Máx.': get('DemMaxs_BUS_B', None),
        'Dem. Div': get('D_Div_BUS_B', None),
        'Dem. Coinc': get('Dem_Coinc_BUS_B', None),
        'Dem. No Coinc': get('Dem_No_Coinc_BUS_B', None),
        'F. Div': get('F_Div_BUS_B', None),
        'F. Coinc': get('F_Coinc_BUS_B', None)
    }
    df_buses = pd.concat([pd.Series(datos_busA, name='BUS_A'), pd.Series(datos_busB, name='BUS_B')], axis=1)

    # Escribir Excel
    out_path = os.path.join(os.getcwd(), output_name)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        if not df_A.empty:
            df_A.to_excel(writer, sheet_name='Circuitos_A')
        if not df_B.empty:
            df_B.to_excel(writer, sheet_name='Circuitos_B')
        df_buses.to_excel(writer, sheet_name='Buses')

    print ()
    print(f"Resultados exportados a: {out_path}")
    return out_path

# ejecutar exportación
Exportar_Resultados_Excel('Resultados_Subes.xlsx')

