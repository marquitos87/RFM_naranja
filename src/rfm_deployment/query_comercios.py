import cx_Oracle
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


#==================================================================================================================

def f_query(fi_int, ff_int): #, fecha_init, fecha_init_month, fecha_end):
    
    """Consulta Oracle PL-SQL a la base de datos de Naranja. Los argumentos 'fi_int' y 'ff_int' indican los extremos
    del intervalo de tiempo para llevar a cabo el modelo RFM. Ambas fechas ingresan a la consulta como números enteros.
    'fecha_init' representa lo mismo que 'fi_init' pero con un formato diferente (%d/%m/%Y)"""
    
    query = ("""

select a.dim_tiempos, c.cuit, a.dim_rubros, b.rubro_descripcion, 
c.comercio_descripcion, d.segmento, a.met_importe, a.dim_tiempos_presentacion

from dw.fac_facturacion_ca a

inner join

dw.dim_rubros b

on a.dim_rubros = b.dimension_key

inner join

dw.dim_geo_comercios c

on c.dimension_key = b.dimension_key

left join inteligenciacomercial.ca_segmentos d

on d.cuit = c.cuit """

f'where a.dim_tiempos between {ff_int} and {fi_int}'

"""and c.fecha_baja = to_date('00010101000000', 'YYYYMMDDHH24MISS')

and (c.cuit <> 0 and c.cuit <> 1)


""")

    return query
#=========================================================================================================================

def consulta_DW(query, cur, fecha_in, fecha_out):
    
    """Ejecuta la consulta al data Warehouse, genera un dataframe y lo guarda localmente. Necesita 4 argumentos:
    'query', la consulta SQL, 'cur', un cursor, 'fecha_in' y 'fecha_out', strings de las fechas 'fi_int' y 'ff_int' para
    poder etiquetar el nombre del dataframe guardado."""
    
    cur.execute(query)
    res = cur.fetchall()
    columns = [c[0] for c in cur.description]
    data = pd.DataFrame( data = res , columns=columns)
    data.to_csv(f'ca_{fecha_in}--{fecha_out}.csv',  index = False)
    return data


#=========================================================================================================================

