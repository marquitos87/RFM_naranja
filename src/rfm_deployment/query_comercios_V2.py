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

def f_query(fi_int, ff_int, fecha_init, fecha_end, label_seg):
    
    """Consulta Oracle PL-SQL a la base de datos de Naranja. Los argumentos 'fi_int' y 'ff_int' indican los extremos
    del intervalo de tiempo para llevar a cabo el modelo RFM. Ambas fechas ingresan a la consulta como números enteros.
    'fecha_init' representa lo mismo que 'fi_init' pero con un formato diferente (%d/%m/%Y)"""
    
    query = ("""

select

r.cuit,  """

f"(to_date('{fecha_init}', 'DD/MM/YYYY') - to_date(substr(max(r.fecha), 1,10))) as recencia,  "

"""count(r.cuit) as frecuencia,

sum(r.importe) as monto

from

(select p.*, s.segmento segmento

from

(select

b.cuit cuit,

a.importe importe,

a.fecha fecha

from

(select

f.dim_geo_comercios dim_geo_comercios,

r.rubro_descripcion rubro_descripcion,

f.met_importe importe,

f.atr_moneda moneda,

f.atr_fecha_presentacion fecha

from dw.dim_rubros r

inner join dw.fac_consumos_comercios f

on f.dim_rubros = r.dimension_key """

f'where f.DIM_TIEMPOS between {ff_int} and {fi_int}'

"""and f.DIM_RUBROS NOT IN (69, 27, 115, -1)

and f.ATR_DEBITO NOT IN ('S')

and f.atr_fecha_presentacion <> to_date('00010101000000', 'YYYYMMDDHH24MISS') """

f"and f.atr_fecha_presentacion between to_date('{fecha_end}', 'DD/MM/YYYY') and to_date('{fecha_init}', 'DD/MM/YYYY')) a " 

"""inner join

(select

dimension_key,

comercio_id,

comercio_codigo,

comercio_descripcion,

origen, 

cuit,

rubro,

fecha_baja

from dw.dim_geo_comercios ) b

on b.dimension_key = a.dim_geo_comercios

where b.origen = 'C'

and b.comercio_descripcion NOT IN ('MERCADO PAGO')

and b.fecha_baja = to_date('00010101000000', 'YYYYMMDDHH24MISS')) p

left join

(select distinct atr_cuit, atr_segmento segmento from dw.fac_rentabilidad_ca_resumen """

f'where dim_tiempos between {ff_int} and {fi_int} ) s '

"""on p.cuit = s.atr_cuit """

f"where upper(s.segmento) = '{label_seg}' ) r " 

"""where r.cuit NOT IN (0, 1)

group by r.cuit


""")

    return query
#=========================================================================================================================

def consulta_DW(query, cur, fecha_in, fecha_out, label_seg):
    
    """Ejecuta la consulta al data Warehouse, genera un dataframe y lo guarda localmente. Necesita 4 argumentos:
    'query', la consulta SQL, 'cur', un cursor, 'fecha_in' y 'fecha_out', strings de las fechas 'fi_int' y 'ff_int' para
    poder etiquetar el nombre del dataframe guardado."""
    
    cur.execute(query)
    res = cur.fetchall()
    columns = [c[0] for c in cur.description]
    data = pd.DataFrame( data = res , columns=columns)
    data.to_parquet(f'{label_seg}-{fecha_in}--{fecha_out}.parquet',  index = False)
    return data


#=========================================================================================================================

