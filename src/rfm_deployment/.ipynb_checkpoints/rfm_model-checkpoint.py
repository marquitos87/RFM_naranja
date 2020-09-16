import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime
import cairo
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from itertools import product


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
    #df = df.dropna(how='any',axis=0) 
    
    return df

    
#=====================================================================================================================

def distribucion_aperturas(df):
    a4_dims = (15, 8)
    df['APERTURA'] = df['APERTURA'].apply(lambda x: x[0:4])
    df = df[['APERTURA', 'DNI']].groupby(['APERTURA']).count()
    a4_dims = (15, 8)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax.set_title('Distribución de aperturas de cuenta por año')
    m = sns.barplot(ax = ax, y=df['DNI'], x=df.index)
    m.set_xticklabels(rotation=45, labels=df.index)
    ax.set(xlabel='Año de apertura', ylabel='Cantidad')
    plt.show()

#====================================================================================================================

def histogramas(R_serie, F_serie, M_serie):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8.27))
    plt.subplots_adjust(wspace = 0.4)
    fig.suptitle('Distribución de clientes según Recencia, Frecuencia y Monto')
    ax1.hist(R_serie, bins = 50, range = [0,100] , facecolor = 'green', alpha = 0.75, 
             edgecolor = 'black', linewidth = 0.5 )
    ax1.set(xlabel='Recencia (días)', ylabel = 'Cantidad')
    ax1.tick_params(axis='both', labelrotation = 45)
    ax2.hist(F_serie, bins = 50, range = [0,90] , facecolor = 'blue', alpha = 0.75, 
             edgecolor = 'black', linewidth = 0.5 )
    ax2.set(xlabel='Frecuencia')
    ax2.tick_params(axis='both', labelrotation = 45)
    ax3.hist(M_serie, bins = 50, range = [0,150000] , facecolor = 'red', alpha = 0.75, 
             edgecolor = 'black', linewidth = 0.5 )
    ax3.set(xlabel='Monto (Pesos)')
    ax3.tick_params(axis='both', labelrotation = 45)
    
    plt.show()
    

    
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
    
def quintil_rent(x,p,d):  
    
    """Funcion para obtener la division por quintiles de la rentabilidad"""
    
    if x <= d[p][0.20]:
        return 'Q1'
    elif x <= d[p][0.4]:
        return 'Q2'
    elif x <= d[p][0.6]: 
        return 'Q3'
    elif x <= d[p][0.8]:
        return 'Q4'
    else:
        return 'Q5'    
#=========================================================================================================================

def rfm_scoring(rfm, fecha_in, fecha_out):
    
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
    rfm.to_csv(f'rfm-final-{fecha_in}--{fecha_out}.csv', index = False)
    
    return rfm
#=======================================================================================================

def rentabilidad(fecha_in_1, fecha_out_1):
    
    rfm = pd.read_csv(f'rfm-segmentos-{fecha_in_1}--{fecha_out_1}.csv')
    
    q6 = float(rfm[['RENTABILIDAD']].mean() + 3 * rfm[['RENTABILIDAD']].std())
    q0 = float(rfm[['RENTABILIDAD']].mean() - 3 * rfm[['RENTABILIDAD']].std())
    
    #Los siguientes son dataframes que contienen q0 y q6 (peores y mejores clientes segun rentabilidad, respectivamente)
    #los descartamos para el análisis siguiente.
    #df_q6 = rfm[rfm['RENTABILIDAD'] > q6 ]
    #df_q0 = rfm[rfm['RENTABILIDAD'] < q0 ]
    
    df_quintil = rfm[rfm['RENTABILIDAD'] > q0][rfm['RENTABILIDAD'] < q6]
    
    quintiles = df_quintil[['RENTABILIDAD']].quantile(q=[0.2,0.4,0.6,0.8])
    rfm['RENT_QUINTIL'] = rfm['RENTABILIDAD'].apply(quintil_rent, args=('RENTABILIDAD',quintiles))
    df_quintil_segment = rfm[['RENT_QUINTIL', 'Segment']]
    
    summary = pd.pivot_table(data=df_quintil_segment,
              index='RENT_QUINTIL',columns='Segment',
              aggfunc='size').apply(lambda x: (x/sum(x))*100,axis=0)
    
    summary.to_csv(f'rentabilidad--{fecha_in_1}-{fecha_out_1}.csv')
    
        
    fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5, figsize=(15, 9))
    plt.subplots_adjust(wspace = 0.5)
    fig.suptitle('Distribución de rentabilidad para cada segmento RFM', fontsize=25)

    sns.barplot(ax=ax0, y=summary['Campeones'], x=summary.index)
    ax0.set(ylabel = 'Campeones (%)')
    sns.barplot(ax=ax1, y=summary['Leales'], x=summary.index)
    ax1.set(ylabel = 'Leales (%)')
    sns.barplot(ax=ax2, y=summary['Potencialmente Leales'], x=summary.index)
    ax2.set(ylabel = 'Potencialmente Leales (%)')
    sns.barplot(ax=ax3, y=summary['Prometedores'], x=summary.index)
    ax3.set(ylabel = 'Prometedores (%)')
    sns.barplot(ax=ax4, y=summary['Reciente operativo'], x=summary.index)
    ax4.set(ylabel = 'Reciente operativo (%)')
    sns.barplot(ax=ax5, y=summary['No se pueden perder'], x=summary.index)
    ax5.set(ylabel = 'No se pueden perder (%)')
    sns.barplot(ax=ax6, y=summary['Necesitan Atencion'], x=summary.index)
    ax6.set(ylabel = 'Necesitan Atencion (%)')
    sns.barplot(ax=ax7, y=summary['En Riesgo'], x=summary.index)
    ax7.set(ylabel = 'En Riesgo (%)')
    sns.barplot(ax=ax8, y=summary['Cercanos a Hibernar'], x=summary.index)
    ax8.set(ylabel = 'Cercanos a Hibernar (%)')
    sns.barplot(ax=ax9, y=summary['Hibernando'], x=summary.index)
    ax9.set(ylabel = 'Hibernando (%)')
    
    return summary

#=======================================================================================================

def rentabilidad_acum(fecha_in_1, fecha_out_1, c):
    
    rfm = pd.read_csv(f'rfm-segmentos-{fecha_in_1}--{fecha_out_1}.csv')
    q6 = float(rfm[['RENTABILIDAD']].mean() + 3 * rfm[['RENTABILIDAD']].std())
    q0 = float(rfm[['RENTABILIDAD']].mean() - 3 * rfm[['RENTABILIDAD']].std())
    df_percentil = rfm[rfm['RENTABILIDAD'] > q0][rfm['RENTABILIDAD'] < q6]
    percentil = df_percentil[['RENTABILIDAD']].quantile(q=[round(c/100, 2) for c in range(1, 100 + 1)])
    
    lista = []

    for i in range(1, 100 + 1):
    
        if i == 1:
        
            lista.append(df_percentil[df_percentil['RENTABILIDAD']<=percentil['RENTABILIDAD'][round(i/100, 2)]]['RENTABILIDAD'].sum()) 
        
        elif i != 101:
        
            acum_1 = df_percentil[df_percentil['RENTABILIDAD']<=percentil['RENTABILIDAD'][round((i/100)-0.01, 2)]]['RENTABILIDAD'].sum()
            acum_2 = df_percentil[df_percentil['RENTABILIDAD']<=percentil['RENTABILIDAD'][round(i/100, 2)]]['RENTABILIDAD'].sum()
            lista.append(round(acum_2-acum_1, 2))
 
        else:
        
            pass
    
    dic = {'Percentil': [round(x/100,2) for x in range(1, 100 + 1)], 'Rent_Acum': lista}
    df_acum = pd.DataFrame(dic)
    
    a = df_acum[['Percentil', 'Rent_Acum']].iloc[99-c:]
    b = round(((a.Rent_Acum.sum())/(df_acum.Rent_Acum.sum()))*100, 2)
    print('-------------------------------------------------------------------------------------------------')
    print(f'El {c}% de los clientes más rentables explican el {b}% de la rentabilidad total')
    print('-------------------------------------------------------------------------------------------------')
    
    a4_dims = (15, 8)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig.suptitle('Rentabilidad acumulada por cada percentil', fontsize=25)
    sns.lineplot(x=df_acum.Percentil, y=df_acum.Rent_Acum, color='red', linewidth=2.5)
    ax.set(ylabel = 'Rentabilidad Acumulada ($)', xlabel = 'Percentil')
    plt.gca().invert_xaxis()
    plt.show()
    
    return df_acum
    
#=======================================================================================================

def rent_div_percentiles(fecha_in_1, fecha_out_1):
    
    df_percentil = pd.read_csv(f'rfm-segmentos-{fecha_in_1}--{fecha_out_1}.csv')
    
    percentiles = df_percentil[['RENTABILIDAD']].quantile(q = [round(c/100, 2) for c in range(1, 100 + 1)])
    
    for i in range(1,100 + 1):
        
        for j in df_percentil['RENTABILIDAD']:
        
            if j <= percentiles['RENTABILIDAD'][i/100]:
            
                df_percentil['P_RENT'] = f'Q{i}'
        
    else:
            
        pass 
        
    df_percentil.to_csv('percentil.csv', index = False)
    
    return df_percentil
    

        
#=======================================================================================================

def histogramas2(rfm):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace = 0.5)
    fig.suptitle('Media de las variables RFM para cada valor de Score Total')
    
    rfm.groupby('Total_score')['RECENCIA'].mean().plot(ax=ax1, kind ='bar', color = 'green')
    ax1.set(xlabel = 'Score total' , ylabel = 'Recencia Promedio')
    ax1.tick_params(axis='both', labelrotation = 45)
    rfm.groupby('Total_score')['FRECUENCIA'].mean().plot(ax=ax2, kind ='bar', color = 'blue')
    ax2.set(xlabel = 'Score total' ,ylabel ='Frecuencia Promedio')
    ax2.tick_params(axis='both', labelrotation = 45)
    rfm.groupby('Total_score')['MONTO'].mean().plot(ax=ax3, kind ='bar', color = 'red')
    ax3.set(xlabel = 'Score total' ,ylabel = 'Monto Promedio')
    ax3.tick_params(axis='both', labelrotation = 45)
    
    plt.show()
    

#=========================================================================================================


def segmentar(rfm, fecha_in, fecha_out):

	"""Cuenta cantidad de clientes segun el RFMscore y genera un barplot. Requiere como argumentos 'rfm' que es 
	el dataframe final obtenido por la función rfm_scoring y 'fecha_in' y 'fecha_out', strings de las fechas
	'fi_int' y 'ff_int' para poder etiquetar el nombre del dataframe guardado"""


	lista = [str(i) for i in range(1,6)]
	perm = [p for p in product(lista, repeat=3)]
	tuple_to_string = [''.join(i) for i in perm]

	dict_aux = {}
	for k in tuple_to_string:
		dict_aux[k] =[]

	for i in dict_aux.keys():
		dict_aux[i] = rfm[rfm['RFMScore']==i]['DNI']

	cant = {} #para generar un dataframe con RFScore, cantidad de comercios y porcentaje
	for cat in dict_aux.keys():
		cant[cat] = len(dict_aux[cat])

	data =  pd.DataFrame(list(cant.items()))
	data = data.rename(columns={0: "RFMScore", 1: "cantidad"})
	data['%'] = round(data['cantidad']/data['cantidad'].sum() * 100, 2)

	#graficamos barplots con la distribucion de RFMScores
	a4_dims = (15, 8.27)
	fig, ax = plt.subplots(figsize=a4_dims)
	sns.barplot(ax=ax, y=data['%'], x=data['RFMScore'])
	ax.set_xticklabels(rotation=90, labels=data['RFMScore'])
	plt.show()

	return data


#=============================================================================================================

def label_segmentos(rfm, fecha_in, fecha_out):
    
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
    rfm['Segment'] = rfm['R_Quintil'].map(str) + rfm['F_Quintil'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)

    rfm.to_csv(f'rfm-segmentos-{fecha_in}--{fecha_out}.csv', index = False)

    dic_dni = {'Campeones': [], 'En Riesgo': [], 'Reciente operativo': [],
                'Leales': [], 'Potencialmente Leales': [], 'Hibernando': [], 'Necesitan Atencion': [],
                'Cercanos a Hibernar': [], 'No se pueden perder': [], 'Prometedores': []}
    
    for i in dic_dni.keys():
        dic_dni[i] = rfm[rfm['Segment']==i]['DNI']
          
    cant = {} #para generar un dataframe con segmento, cantidad de cuits y porcentaje
    
    for cat in dic_dni.keys():
        
        cant[cat] = len(dic_dni[cat])
        
        print(f'La cantidad de comercios en el segmento {cat} es de: {len(dic_dni[cat])}')
        
    rfm_segmentado =  pd.DataFrame(list(cant.items()))
    rfm_segmentado = rfm_segmentado.rename(columns={0: "Segment", 1: "cantidad"})
    rfm_segmentado['%'] = round(rfm_segmentado['cantidad']/rfm_segmentado['cantidad'].sum() * 100, 2)
    rfm_segmentado = rfm_segmentado.set_index('Segment')
    rfm_segmentado.to_csv(f'line_plot-{fecha_in}--{fecha_out}.csv')

      
    #print('-------------------------------------------------------------------------------------------------')
    #print('Distribución porcentual de los clientes según los segmentos')
    #a4_dims = (15, 8)
    #fig, ax = plt.subplots(figsize=a4_dims)
    #m=sns.barplot(ax=ax, y=rfm_segmentado['%'], x=rfm_segmentado.index)
    #m.set_xticklabels(rotation=45, labels=rfm_segmentado.index)
    #fig.savefig(f"barplot-seg-{fecha_in}--{fecha_out}.pdf")
    #plt.show()
    
    return rfm_segmentado, dic_dni

#================================================================================================================

def evolucion_trimestral_segmentos(df_lineas):
    
    plt.figure(figsize = (15,10))
    plt.plot('Trimestre', 'Campeones', data = df_lineas, marker = 'o', markerfacecolor = 'blue', markersize = 6, color = 'blue', linewidth = 1 )
    plt.plot('Trimestre', 'Leales', data = df_lineas, marker = 'o', markerfacecolor = 'red', markersize = 6, color = 'red', linewidth = 1 )
    plt.plot('Trimestre', 'Potencialmente Leales', data = df_lineas, marker = 'o', markerfacecolor = 'green', markersize = 6, color = 'green', linewidth = 1 )
    plt.plot('Trimestre', 'Reciente operativo', data = df_lineas, marker = 'o', markerfacecolor = 'black', markersize = 6, color = 'black', linewidth = 1 )
    plt.plot('Trimestre', 'Prometedores', data = df_lineas, marker = 'o', markerfacecolor = 'olive', markersize = 6, color = 'olive', linewidth = 1 )
    plt.plot('Trimestre', 'No se pueden perder', data = df_lineas, marker = 'o', markerfacecolor = 'yellow', markersize = 6, color = 'yellow', linewidth = 1 )
    plt.plot('Trimestre', 'En Riesgo', data = df_lineas, marker = 'o', markerfacecolor = 'blue', markersize = 6, color = 'skyblue', linewidth = 1 )
    plt.plot('Trimestre', 'Necesitan Atencion', data = df_lineas, marker = 'o', markerfacecolor = 'brown', markersize = 6, color = 'brown', linewidth = 1 )
    plt.plot('Trimestre', 'Cercanos a Hibernar', data = df_lineas, marker = 'o', markerfacecolor = 'magenta', markersize = 6, color = 'magenta', linewidth = 1 )
    plt.plot('Trimestre', 'Hibernando', data = df_lineas, marker = 'o', markerfacecolor = 'pink', markersize = 6, color = 'pink', linewidth = 1 )
    plt.legend(loc = 'upper right', prop ={'size': 12})
    plt.xticks(ha='right',rotation=45)
    plt.xlabel('Trimestre')
    plt.ylabel('% poblacion')
    plt.savefig('lineas_tiempo.pdf')
    plt.show()

#===============================================================================================================

def migraciones(fecha_in_1, fecha_in_2, fecha_out_1, fecha_out_2):
    
    df1 = pd.read_csv(f'rfm-segmentos-{fecha_in_1}--{fecha_out_1}.csv')
    df2 = pd.read_csv(f'rfm-segmentos-{fecha_in_2}--{fecha_out_2}.csv')
    fecha_in_1 = str(fecha_in_1)
    fecha_in_2 = str(fecha_in_2)

    df1_a = df1[['DNI','Segment']]
    df1_a = df1_a.rename(columns={'Segment':f'Segment_{int(fecha_in_1[0:4])}'})

    df2_a = df2[['DNI','Segment']]
    df2_a = df2_a.rename(columns={'Segment':f'Segment_{int(fecha_in_2[0:4])}'})

    mergeddf = df1_a.merge(df2_a, on = 'DNI')
    
    comparedf=pd.concat([mergeddf.Segment_2019.value_counts(normalize=True),
                         mergeddf.Segment_2018.value_counts(normalize=True)],axis=1, sort=False)
    comparedf = comparedf.reset_index()
    a = comparedf[['index', f'Segment_{int(fecha_in_1[0:4])}']]
    a['year'] = int(fecha_in_1[0:4]) 
    b = comparedf[['index', f'Segment_{int(fecha_in_2[0:4])}']]
    b['year'] = int(fecha_in_2[0:4])
    a = a.rename(columns = {'index':'index', f'Segment_{int(fecha_in_1[0:4])}':'propor'})
    b = b.rename(columns={'index':'index', f'Segment_{int(fecha_in_2[0:4])}':'propor'})
    c = pd.concat([a,b])
    
    fig, ax = plt.subplots(figsize=(13,8))
    fig.suptitle('Distribución de clientes segun segmentos y trimestre', fontsize=25)
    sns.barplot(x = 'index', y = 'propor', hue = 'year',  data=c)
    ax.set_xlabel('')
    ax.set_ylabel('Clientes (%)')
    plt.xticks(rotation=30, ha='center')
    plt.show()
    
    print('--------------------------------------------------------------------------------------------------------')
    print('Porcentaje de migración de clientes entre segmentos comparando el mismo trimestre en años consecutivos..')
    sumdf=pd.pivot_table(data=mergeddf.set_index('DNI'),
                          index=f'Segment_{int(fecha_in_2[0:4])}',columns=f'Segment_{int(fecha_in_1[0:4])}',
                          aggfunc='size').apply(lambda x: (x/sum(x))*100,axis=1)

    sumdf['Total'] = sumdf.apply(lambda x: sum(x),axis=1)
    sumdf.to_csv(f'migracion-{fecha_in_1}-{fecha_out_1}--{fecha_in_2}-{fecha_out_2}.csv', index = False)
    
    return sumdf

#===============================================================================================================

def riesgo(fecha_in_1, fecha_out_1):
    
    df1 = pd.read_csv(f'rfm-segmentos-{fecha_in_1}--{fecha_out_1}.csv')
    df1 = df1.dropna()
    
    categories = {}
    for cat in df1['NIVEL_RIESGO_TRIM'].unique():
        categories[cat] = cat.split()[2]
        
    df1['NIVEL_RIESGO_TRIM'] = df1['NIVEL_RIESGO_TRIM'].map(categories)
    df1 = df1[~df1['NIVEL_RIESGO_TRIM'].isin(['NUEVOS/INACTIVOS', 'NA'])]
    
    df_riesgo = df1[['NIVEL_RIESGO_TRIM', 'Segment']]
    
    summary_riesgo = pd.pivot_table(data=df_riesgo,
              index='NIVEL_RIESGO_TRIM',columns='Segment',
              aggfunc='size').apply(lambda x: (x/sum(x))*100,axis=0)
    
    summary_riesgo = summary_riesgo.reindex(['RB', 'RMB', 'RM', 'RMA', 'RA'])
    #summary_riesgo.loc["Total"] = summary_riesgo.sum()
    summary_riesgo.to_csv(f'riesgo--{fecha_in_1}-{fecha_out_1}.csv', index = False)
    
    fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5, figsize=(15, 9))
    plt.subplots_adjust(wspace = 0.5)
    fig.suptitle('Distribución de riesgo para cada segmento RFM', fontsize=25)

    sns.barplot(ax=ax0, y=summary_riesgo['Campeones'], x=summary_riesgo.index)
    ax0.set(ylabel = 'Campeones (%)')
    sns.barplot(ax=ax1, y=summary_riesgo['Leales'], x=summary_riesgo.index)
    ax1.set(ylabel = 'Leales (%)')
    sns.barplot(ax=ax2, y=summary_riesgo['Potencialmente Leales'], x=summary_riesgo.index)
    ax2.set(ylabel = 'Potencialmente Leales (%)')
    sns.barplot(ax=ax3, y=summary_riesgo['Prometedores'], x=summary_riesgo.index)
    ax3.set(ylabel = 'Prometedores (%)')
    sns.barplot(ax=ax4, y=summary_riesgo['Reciente operativo'], x=summary_riesgo.index)
    ax4.set(ylabel = 'Reciente operativo (%)')
    sns.barplot(ax=ax5, y=summary_riesgo['No se pueden perder'], x=summary_riesgo.index)
    ax5.set(ylabel = 'No se pueden perder (%)')
    sns.barplot(ax=ax6, y=summary_riesgo['Necesitan Atencion'], x=summary_riesgo.index)
    ax6.set(ylabel = 'Necesitan Atencion (%)')
    sns.barplot(ax=ax7, y=summary_riesgo['En Riesgo'], x=summary_riesgo.index)
    ax7.set(ylabel = 'En Riesgo (%)')
    sns.barplot(ax=ax8, y=summary_riesgo['Cercanos a Hibernar'], x=summary_riesgo.index)
    ax8.set(ylabel = 'Cercanos a Hibernar (%)')
    sns.barplot(ax=ax9, y=summary_riesgo['Hibernando'], x=summary_riesgo.index)
    ax9.set(ylabel = 'Hibernando (%)')

    return summary_riesgo
    
    
#===============================================================================================================

class RFM_graph(Gtk.Window):

    def __init__(self):
        self.porcentajes = {
                           'Campeones': 0.00,
                           'Leales': 0.00,
                           'Potencialmente Leales': 0.00,
                           'Reciente operativo': 0.00,
                           'Prometedores': 0.00,
                           'Necesitan Atencion': 0.00,
                           'Cercanos a Hibernar': 0.00,
                           'No se pueden perder': 0.00,
                           'En Riesgo': 0.00,
                           'Hibernando': 0.00
                           }
        
        self.cantidad = {
                           'Campeones': 0.00,
                           'Leales': 0.00,
                           'Potencialmente Leales': 0.00,
                           'Reciente operativo': 0.00,
                           'Prometedores': 0.00,
                           'Necesitan Atencion': 0.00,
                           'Cercanos a Hibernar': 0.00,
                           'No se pueden perder': 0.00,
                           'En Riesgo': 0.00,
                           'Hibernando': 0.00
                           }
        
        self.R_MEDIAN = {
                           'Campeones': 0.00,
                           'Leales': 0.00,
                           'Potencialmente Leales': 0.00,
                           'Reciente operativo': 0.00,
                           'Prometedores': 0.00,
                           'Necesitan Atencion': 0.00,
                           'Cercanos a Hibernar': 0.00,
                           'No se pueden perder': 0.00,
                           'En Riesgo': 0.00,
                           'Hibernando': 0.00
                           }
            
        self.F_MEDIAN = {
                           'Campeones': 0.00,
                           'Leales': 0.00,
                           'Potencialmente Leales': 0.00,
                           'Reciente operativo': 0.00,
                           'Prometedores': 0.00,
                           'Necesitan Atencion': 0.00,
                           'Cercanos a Hibernar': 0.00,
                           'No se pueden perder': 0.00,
                           'En Riesgo': 0.00,
                           'Hibernando': 0.00
                           }
                
        self.M_MEDIAN = {
                           'Campeones': 0.00,
                           'Leales': 0.00,
                           'Potencialmente Leales': 0.00,
                           'Reciente operativo': 0.00,
                           'Prometedores': 0.00,
                           'Necesitan Atencion': 0.00,
                           'Cercanos a Hibernar': 0.00,
                           'No se pueden perder': 0.00,
                           'En Riesgo': 0.00,
                           'Hibernando': 0.00
                           }
       
       
        super(RFM_graph, self).__init__()
        self.init_ui()
        self.load_image()
       
    def init_ui(self):    
        darea = Gtk.DrawingArea()
        darea.connect("draw", self.on_draw)
        self.add(darea)

        self.set_title("Summary")
        self.resize(1040, 1700)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()
       
    def load_image(self):
        self.ims = cairo.ImageSurface.create_from_png("background_rfm_gimp.png")
       
    def on_draw(self, wid, cr):
        
        scale = 1.
        cr = cairo.Context(self.ims)
        #cr.scale(scale, scale)    # scale the context by (x, y)
        #cr.set_source_surface(self.ims, 10, 10)
        #cr.paint()
        #cr.restore()
       
        # Acá va el texto y su configuración
       
        cr.set_source_rgb(0, 0, 0)

        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(11)
        f_corr = scale / 0.5
      
        
        cr.move_to(115 * f_corr, 35 * f_corr)
        cr.show_text(f"pop: {self.cantidad['No se pueden perder']} clientes ({self.porcentajes['No se pueden perder']})")
        cr.move_to(115 * f_corr, 140 * f_corr)
        cr.show_text(f"pop: {self.cantidad['En Riesgo']} clientes ({self.porcentajes['En Riesgo']})")
        cr.move_to(115 * f_corr, 263 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Hibernando']} clientes ({self.porcentajes['Hibernando']})")
        cr.move_to(380 * f_corr, 80 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Leales']} clientes ({self.porcentajes['Leales']})")
        cr.move_to(570 * f_corr, 80 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Campeones']} clientes ({self.porcentajes['Campeones']})")
        cr.move_to(570 * f_corr, 280 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Reciente operativo']} clientes ({self.porcentajes['Reciente operativo']})")
        cr.move_to(445 * f_corr, 280 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Prometedores']} clientes ({self.porcentajes['Prometedores']})")
        cr.move_to(310 * f_corr, 270 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Cercanos a Hibernar']} clientes ({self.porcentajes['Cercanos a Hibernar']})")
        cr.move_to(310 * f_corr, 160 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Necesitan Atencion']} clientes ({self.porcentajes['Necesitan Atencion']})")
        cr.move_to(515 * f_corr, 200 * f_corr)
        cr.show_text(f"pop: {self.cantidad['Potencialmente Leales']} clientes ({self.porcentajes['Potencialmente Leales']})")

        
        cr.move_to(115 * f_corr, 45 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['No se pueden perder']} (días)")
        cr.move_to(115 * f_corr, 150 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['En Riesgo']} (días)")
        cr.move_to(115 * f_corr, 273 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Hibernando']} (días)")
        cr.move_to(380 * f_corr, 90 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Leales']} (días)")
        cr.move_to(570 * f_corr, 90 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Campeones']} (días)")
        cr.move_to(570 * f_corr, 290 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Reciente operativo']} (días)")
        cr.move_to(445 * f_corr, 290 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Prometedores']} (días)")
        cr.move_to(310 * f_corr, 280 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Cercanos a Hibernar']} (días)")
        cr.move_to(310 * f_corr, 170 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Necesitan Atencion']} (días)")
        cr.move_to(515 * f_corr, 210 * f_corr)
        cr.show_text(f"R_MEDIAN: {self.R_MEDIAN['Potencialmente Leales']} (días)")



        
        cr.move_to(115 * f_corr, 55 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['No se pueden perder']}")
        cr.move_to(115 * f_corr, 160 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['En Riesgo']}")
        cr.move_to(115 * f_corr, 283 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Hibernando']}")
        cr.move_to(380 * f_corr, 100 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Leales']}")
        cr.move_to(570 * f_corr, 100 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Campeones']}")
        cr.move_to(570 * f_corr, 300 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Reciente operativo']}")
        cr.move_to(445 * f_corr, 300 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Prometedores']}")
        cr.move_to(310 * f_corr, 290 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Cercanos a Hibernar']}")
        cr.move_to(310 * f_corr, 180 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Necesitan Atencion']}")
        cr.move_to(515 * f_corr, 220 * f_corr)
        cr.show_text(f"F_MEDIAN: {self.F_MEDIAN['Potencialmente Leales']}")



        
        cr.move_to(115 * f_corr, 65 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['No se pueden perder']} (pesos)")
        cr.move_to(115 * f_corr, 170 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['En Riesgo']} (pesos)")
        cr.move_to(115 * f_corr, 293 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Hibernando']} (pesos)")
        cr.move_to(380 * f_corr, 110 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Leales']} (pesos)")
        cr.move_to(570 * f_corr, 110 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Campeones']} (pesos)")
        cr.move_to(570 * f_corr, 310 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Reciente operativo']} (pesos)")
        cr.move_to(445 * f_corr, 310 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Prometedores']} (pesos)")
        cr.move_to(310 * f_corr, 300 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Cercanos a Hibernar']} (pesos)")
        cr.move_to(310 * f_corr, 190 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Necesitan Atencion']} (pesos)")
        cr.move_to(515 * f_corr, 230 * f_corr)
        cr.show_text(f"M_MEDIAN: {self.M_MEDIAN['Potencialmente Leales']} (pesos)")

        
        self.ims.write_to_png ("rfm-trimestral.png")
        
#==================================================================================================================

def rfm_describe():

    #---------------------------------------------
    #Diagrama RFM descriptivo para cada segmento.
    #---------------------------------------------

    surface = cairo.ImageSurface.create_from_png("background_rfm_gimp.png")
    ctx = cairo.Context(surface)

    # Acá va el texto y su configuración
    scale = 0.8       
    ctx.set_source_rgb(0, 0, 0)
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(12)
    f_corr = scale / 0.5

    #NO SE PUEDEN PERDER
    ctx.move_to(125 * f_corr, 50 * f_corr)
    ctx.show_text("Han comprado grandes cantidades")
    ctx.move_to(125 * f_corr, 60 * f_corr)
    ctx.show_text("pero hace mucho tiempo")
    #EN RIESGO
    ctx.move_to(125 * f_corr, 140 * f_corr)
    ctx.show_text("Fueron buenos compradores")
    ctx.move_to(125 * f_corr, 150 * f_corr)
    ctx.show_text("pero hace mucho tiempo")
    #HIBERNANDO
    ctx.move_to(170 * f_corr, 320 * f_corr)
    ctx.show_text("'Inactivos'")
    #LEALES
    ctx.move_to(430 * f_corr, 80 * f_corr)
    ctx.show_text("Compran regularmente. Responden ofertas")
    #CAMPEONES
    ctx.move_to(700 * f_corr, 80 * f_corr)
    ctx.show_text("Compran frecuentemente")
    ctx.move_to(700 * f_corr, 90 * f_corr)
    ctx.show_text("con montos altos")
    #Reciente operativo
    ctx.move_to(710 * f_corr, 350 * f_corr)
    ctx.show_text("Compraron actualmente")
    ctx.move_to(710 * f_corr, 360 * f_corr)
    ctx.show_text("en pequeñas cantidades")
    #PROMETEDORES
    ctx.move_to(550 * f_corr, 350 * f_corr)
    ctx.show_text("Compraron recientemente")
    ctx.move_to(550 * f_corr, 360 * f_corr)
    ctx.show_text("en pequeñas cantidades")
    #CERCANOS A HIBERNAR
    ctx.move_to(400 * f_corr, 300 * f_corr)
    ctx.show_text("Tienden a volverse inactivos")
    ctx.move_to(400 * f_corr, 310 * f_corr)
    ctx.show_text("si no se estimulan")
    #NECESITAN ATENCION
    ctx.move_to(400 * f_corr, 210 * f_corr)
    ctx.show_text("Clientes promedio")
    #POTENCIALMENTE LEALES
    ctx.move_to(630 * f_corr, 230 * f_corr)
    ctx.show_text("Compraron recientemente")
    ctx.move_to(630 * f_corr, 240 * f_corr)
    ctx.show_text("con frecuencia promedio")

    #BIBLIO:
    ctx.move_to(500 * f_corr, 430 * f_corr)
    ctx.show_text("Fuente: https://clevertap.com/blog/automate-user-segmentation-with-rfm-analysis/")

    ctx.fill()


    surface.write_to_png('rfm-describe.png')







    


    
