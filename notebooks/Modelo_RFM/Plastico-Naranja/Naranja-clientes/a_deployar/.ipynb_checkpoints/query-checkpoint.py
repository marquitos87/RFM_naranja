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

def f_query(fi_int, ff_int, fecha_init, fecha_init_month, fecha_end):
    
    """Consulta Oracle PL-SQL a la base de datos de Naranja. Los argumentos 'fi_int' y 'ff_int' indican los extremos
    del intervalo de tiempo para llevar a cabo el modelo RFM. Ambas fechas ingresan a la consulta como números enteros.
    'fecha_init' representa lo mismo que 'fi_init' pero con un formato diferente (%d/%m/%Y)"""
    
    query = ("""    
    
select o.dni, o.antiguedad, o.recencia, o.frecuencia, o.monto

from

(select f.dni dni, g.antiguedad antiguedad, f.frecuencia frecuencia, """

f"(to_date('{fecha_init}', 'DD/MM/YYYY') - to_date(substr(f.fecha,1,10))) RECENCIA, " 

"""f.monto monto

from              

(select d.dim_cuentas dim_cuentas, e.tipo_cuenta tipo_cuenta, e.dni dni,

        d.frecuencia frecuencia, d.fecha fecha, d.monto monto

from dw.dim_cuentas e

inner join

            (select c.dim_cuentas, count(c.dim_cuentas) as frecuencia, max(c.atr_fecha_presentacion) as fecha,

             sum(c.monto_parcial) as monto

             from (

                   select a.dim_cuentas, a.atr_fecha_presentacion, a.met_importe,

                   case WHEN a.ATR_MONEDA = 'Dolares' THEN a.met_importe * b.valor

                   else a.met_importe

                   end monto_parcial           

                   from dw.fac_consumos_comercios a

                   left join(

                             select atr_periodo, avg(valor) valor

                             from riesgocrediticio.cartera_precio_dolar

                             group by atr_periodo) b

                   on a.dim_tiempos = b.atr_periodo """

                   f'where a.DIM_TIEMPOS BETWEEN {ff_int} AND {fi_int} '
             
     f"and a.atr_fecha_presentacion between to_date('{fecha_end}', 'DD/MM/YYYY') and to_date('{fecha_init}', 'DD/MM/YYYY') "

                   """and (a.ATR_ESTADO_CUENTA_LISTIN = 'RE Recuperable (Amarilla)' or a.ATR_ESTADO_CUENTA_LISTIN IS NULL)

                   and a.DIM_RUBROS NOT IN (69, 27, 115, -1)

                   and a.ATR_DEBITO <> 'S') c

             group by c.dim_cuentas) d

on d.dim_cuentas = e.dimension_key ) f

inner join

        (SELECT a.dim_cuentas dim_cuentas, a.met_limite_tc limite_credito, """

f"b.fecha_ingreso apertura, round(months_between(to_date('{fecha_init_month}', 'MM/YYYY'), to_date(to_char(b.fecha_ingreso, 'DD/MM/YYYY'), 'DD/MM/YYYY')), 2) antiguedad "

         """FROM DW.FAC_CUENTAS_SALDOS a

         INNER JOIN DW.dim_CLIENTES b ON b.DIMENSION_KEY = a.DIM_CLIENTES

         WHERE b.FECHA_BAJA = TO_DATE('00010101000000','YYYYMMDDHH24MISS')

         AND a.ATR_ESTADO_APERTURA = '096 - APROBACION' """

         f'AND a.dim_tiempos = {fi_int}) g on  f.dim_cuentas = g.dim_cuentas ) o '            

"""

""")

    return query
#=========================================================================================================================

def consulta_DW(query, cur):
    
    """Ejecuta la consulta al data Warehouse, genera un dataframe y lo guarda localmente. Necesita 4 argumentos:
    'query', la consulta SQL, 'cur', un cursor, 'fecha_in' y 'fecha_out', strings de las fechas 'fi_int' y 'ff_int' para
    poder etiquetar el nombre del dataframe guardado."""
    
    cur.execute(query)
    res = cur.fetchall()
    columns = [c[0] for c in cur.description]
    data = pd.DataFrame( data = res , columns=columns)
    data.to_csv(f'RFM.csv',  index = False)
    return data


#=========================================================================================================================

con = cx_Oracle.connect('MMONTERO_DIS/password@//cluster-dwhAIX-scan:1521/dwh_app_service', encoding = 'utf8')
cur = con.cursor() #creo un cursor
cur.execute(query)
res = cur.fetchall()
columns = [c[0] for c in cur.description]
data = pd.DataFrame( data = res , columns=columns)