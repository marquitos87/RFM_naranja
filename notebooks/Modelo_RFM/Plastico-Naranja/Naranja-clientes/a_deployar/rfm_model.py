import pandas as pd
import numpy as np
import calendar
import datetime



#==================================================================================================================

def add_months(sourcedate, months):
    
    """Función que permite sumar o restar 'months' meses a una fecha 'sourcedate' determinada.
    El formato de 'sourcedate' es de la forma datetime.date(año, mes, dia)."""
    
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)

#==================================================================================================================

def datetime_to_integer(dt_time):
     
    """Función que permite cambiar el formato de una fecha 'dt_time' a un número entero. 
    El formato de 'dt_time' es datetime.date(año, mes, dia)"""
    
    integer = 10000*dt_time.year + 100*dt_time.month + dt_time.day
    return integer


#====================================================================================================================

def preprocesamiento(rfm):
    
    df = rfm[rfm.ANTIGUEDAD >= 6]
    
    return df

    
#==========================================================================================================================

def RScore(x,p,d):   
    
    """Funcion para obtener el Recency score. x es cada registro de la serie rfm['RECENCIA'] y d[p] es la serie quantile['RECENCIA'] """
    
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.4]:
        return 4
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]:
        return 2
    else:
        return 1

def FMScore(x,p,d):  
    
    """Funcion para obtener el score para la frecuencia y para el monto"""
    
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.4]:
        return 2
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]:
        return 4
    else:
        return 5
      
#=========================================================================================================================

def rfm_scoring(rfm):
    
    """Genera la división de las variables recencia, frecuencia y monto en quintiles. Además calcula un RFMscore con
    formato string y un Total_score cuyo valor es la suma de los scores individuales. Finalmente, se guarda un .csv
    listo para analizar. Como argumentos la función necesita 'rfm' que es el dataframe generado por la consulta SQL 
    y 'fecha_in' y 'fecha_out', strings de las fechas 'fi_int' y 'ff_int' para poder etiquetar el nombre del 
    dataframe guardado."""
    
    
    quantile = rfm[['RECENCIA', 'MONTO', 'FRECUENCIA']].quantile(q=[0.2,0.4,0.6,0.8])
    
    
    rfm['R_Quintil'] = rfm['RECENCIA'].apply(RScore,args=('RECENCIA',quantile))
    rfm['F_Quintil'] = rfm['FRECUENCIA'].apply(FMScore, args=('FRECUENCIA',quantile))
    rfm['M_Quintil'] = rfm['MONTO'].apply(FMScore, args=('MONTO',quantile))
    
    
    rfm['RFMScore'] = rfm.R_Quintil.map(str) \
                            + rfm.F_Quintil.map(str) \
                            + rfm.M_Quintil.map(str)
    
    rfm['Total_score'] = rfm['R_Quintil'] + rfm['F_Quintil'] + rfm['M_Quintil']
    
    return rfm

#=========================================================================================================


def label_segmentos(rfm):
    
    """Función que segmenta los clientes según su Frecuencia y Recencia en 10 grupos. Además se crea un diccionario
    con los dni's en cada categoría en el intervalo de tiempo analizado. Finalmente genera un barplot con la fracción
    de clientes en cada categoría. Requiere como argumentos 'rfm' que es el dataframe final obtenido por la función 
    rfm_scoring y 'fecha_in' y 'fecha_out', strings de las fechas 'fi_int' y 'ff_int' para poder etiquetar el nombre 
    del dataframe guardado. """
    
    segt_map = {
    r'[1-2][1-2]': 'Hibernando',
    r'[1-2][3-4]': 'En Riesgo',
    r'[1-2]5': 'No se pueden perder',
    r'3[1-2]': 'Cercanos a Hibernar',
    r'33': 'Necesitan Atencion',
    r'[3-4][4-5]': 'Leales',
    r'41': 'Prometedores',
    r'51': 'Reciente operativo',
    r'[4-5][2-3]': 'Potencialmente Leales',
    r'5[4-5]': 'Campeones'
    }
    rfm['Segmento'] = rfm['R_Quintil'].map(str) + rfm['F_Quintil'].map(str)
    rfm['Segmento'] = rfm['Segmento'].replace(segt_map, regex=True)
    rfm = rfm[['DNI', 'RFMScore', 'Segmento']]

    rfm.to_csv(f'RFM-segmentos.csv', index = False)


