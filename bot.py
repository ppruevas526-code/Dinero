import time
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================
# VERIFICAR DEPENDENCIAS
# =============================================
try:
    import pandas as pd
    import numpy as np
    print("✅ Pandas y Numpy instalados correctamente")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

try:
    from iqoptionapi.stable_api import IQ_Option
    print("✅ IQ Option API instalada correctamente")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# =============================================
# CONFIGURACIÓN
# =============================================
EMAIL = "ppruevas526@gmail.com"
PASSWORD = "Crepes.2026*"
ACTIVO = "EURUSD"
TIEMPO_VELA = 300  # 5 minutos
CANTIDAD_VELAS = 100
MONTO = 1.0
DURACION = 5  # 5 minutos de duración
MODO_REAL = False
MAX_OPERACIONES_POR_DIA = 30
MAX_PERCARDIDA_DIARIA = 10.0

# =============================================
# CONTADOR DE SEÑALES
# =============================================
class ContadorSenales:
    def __init__(self):
        self.total_senales = 0
        self.senales_call = 0
        self.senales_put = 0
        self.senales_esperar = 0
        self.operaciones_ejecutadas = 0
        self.operaciones_ganadas = 0
        self.operaciones_perdidas = 0
        self.ganancia_total = 0.0
        self.perdida_total = 0.0
        self.balance_inicial = 0.0
        self.balance_actual = 0.0
        self.historial_senales = []
        self.historial_operaciones = []
    
    def registrar_senal(self, tipo, precio, hora):
        self.total_senales += 1
        if tipo == "CALL":
            self.senales_call += 1
        elif tipo == "PUT":
            self.senales_put += 1
        else:
            self.senales_esperar += 1
        
        self.historial_senales.append({
            'fecha': hora,
            'tipo': tipo,
            'precio': precio
        })
        
        if len(self.historial_senales) > 100:
            self.historial_senales.pop(0)
    
    def registrar_operacion(self, direccion, precio, orden_id, resultado=None, ganancia=0):
        self.operaciones_ejecutadas += 1
        if resultado == "ganada":
            self.operaciones_ganadas += 1
            self.ganancia_total += ganancia
        elif resultado == "perdida":
            self.operaciones_perdidas += 1
            self.perdida_total += abs(ganancia)
        
        self.historial_operaciones.append({
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'direccion': direccion,
            'precio': precio,
            'orden_id': orden_id,
            'resultado': resultado,
            'ganancia': ganancia
        })
        
        if len(self.historial_operaciones) > 100:
            self.historial_operaciones.pop(0)
    
    def actualizar_balance(self, balance):
        if self.balance_inicial == 0:
            self.balance_inicial = balance
        self.balance_actual = balance
    
    def mostrar_estadisticas(self):
        stats = self.obtener_estadisticas()
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS DEL BOT")
        print("="*50)
        print(f"🔹 SEÑALES TOTALES: {stats['total_senales']}")
        print(f"   • CALL: {stats['senales_call']}")
        print(f"   • PUT: {stats['senales_put']}")
        print(f"   • ESPERAR: {stats['senales_esperar']}")
        print("-"*50)
        print(f"🔹 OPERACIONES: {stats['operaciones_ejecutadas']}")
        print(f"   • Ganadas: {stats['operaciones_ganadas']}")
        print(f"   • Perdidas: {stats['operaciones_perdidas']}")
        print(f"   • Tasa de aciertos: {stats['tasa_aciertos']}%")
        print("-"*50)
        print(f"🔹 FINANZAS:")
        print(f"   • Ganancia total: +${stats['ganancia_total']}")
        print(f"   • Pérdida total: -${stats['perdida_total']}")
        print(f"   • Beneficio neto: ${stats['beneficio_neto']}")
        print(f"   • Beneficio %: {stats['beneficio_porcentaje']}%")
        print("-"*50)
        print(f"🔹 BALANCE:")
        print(f"   • Inicial: ${stats['balance_inicial']}")
        print(f"   • Actual: ${stats['balance_actual']}")
        print("="*50 + "\n")
    
    def obtener_estadisticas(self):
        tasa_aciertos = 0
        if self.operaciones_ejecutadas > 0:
            tasa_aciertos = (self.operaciones_ganadas / self.operaciones_ejecutadas) * 100
        
        beneficio_neto = self.ganancia_total - self.perdida_total
        beneficio_porcentaje = 0
        if self.balance_inicial > 0:
            beneficio_porcentaje = (beneficio_neto / self.balance_inicial) * 100
        
        return {
            'total_senales': self.total_senales,
            'senales_call': self.senales_call,
            'senales_put': self.senales_put,
            'senales_esperar': self.senales_esperar,
            'operaciones_ejecutadas': self.operaciones_ejecutadas,
            'operaciones_ganadas': self.operaciones_ganadas,
            'operaciones_perdidas': self.operaciones_perdidas,
            'tasa_aciertos': round(tasa_aciertos, 2),
            'ganancia_total': round(self.ganancia_total, 2),
            'perdida_total': round(self.perdida_total, 2),
            'beneficio_neto': round(beneficio_neto, 2),
            'beneficio_porcentaje': round(beneficio_porcentaje, 2),
            'balance_inicial': round(self.balance_inicial, 2),
            'balance_actual': round(self.balance_actual, 2)
        }

# =============================================
# CONEXIÓN A IQ OPTION
# =============================================
print("\n🔌 Conectando a IQ Option...")

API = IQ_Option(EMAIL, PASSWORD)
API.connect()

if not API.check_connect():
    print("❌ Error de conexión.")
    sys.exit(1)

balance = API.get_balance()
if balance is None:
    print("❌ No se pudo obtener el saldo.")
    sys.exit(1)

print(f"✅ Conectado | Saldo: {balance:.2f} USD")

contador = ContadorSenales()
contador.actualizar_balance(balance)

# =============================================
# ESTRATEGIA: Cruce de Medias (MA5/MA10)
# =============================================
def analizar_velas(velas):
    df = pd.DataFrame(velas, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
    df = df.sort_values('timestamp')
    
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    
    ultima = df.iloc[-1]
    
    if ultima['ma5'] > ultima['ma10']:
        return "CALL"
    elif ultima['ma5'] < ultima['ma10']:
        return "PUT"
    return "ESPERAR"

def hay_operacion_abierta():
    try:
        posiciones = API.get_positions()
        if posiciones:
            for p in posiciones:
                if p.get('asset') == ACTIVO and p.get('status') == 'open':
                    return True
        return False
    except:
        return False

# =============================================
# 🔥 FUNCIÓN PARA VERSIÓN 7.1.1
# =============================================
def ejecutar_operacion(direccion, precio):
    """Ejecuta una operación binaria (versión 7.1.1)"""
    print(f"🚀 EJECUTANDO {direccion.upper()} a {precio:.5f}")
    
    # 🔥 Usar buy_digital_spot_v2 para versión 7.1.1
    status, orden_id = API.buy_digital_spot_v2(ACTIVO, MONTO, direccion, DURACION)
    
    if status:
        print(f"✅ Orden enviada | ID: {orden_id}")
        
        print(f"⏳ Esperando {DURACION} minuto(s)...")
        time.sleep(DURACION * 60 + 5)
        
        try:
            # Verificar resultado
            check_close, win_money = API.check_win_digital_v2(orden_id)
            if check_close:
                if float(win_money) > 0:
                    print(f"💰 ¡GANANCIA! +{win_money:.2f} USD")
                    contador.registrar_operacion(direccion, precio, orden_id, "ganada", float(win_money))
                else:
                    perdida = MONTO
                    print(f"📉 Pérdida: {perdida:.2f} USD")
                    contador.registrar_operacion(direccion, precio, orden_id, "perdida", 0)
                
                nuevo_balance = API.get_balance()
                if nuevo_balance:
                    contador.actualizar_balance(nuevo_balance)
                    print(f"💰 Balance actual: {nuevo_balance:.2f} USD")
        except Exception as e:
            print(f"⚠️ Error al verificar resultado: {e}")
        
        return True
    else:
        print(f"❌ Falló la orden: {orden_id}")
        return False

# =============================================
# BUCLE PRINCIPAL
# =============================================
print(f"\n🚀 BOT INICIADO | Activo: {ACTIVO} | Vela: {TIEMPO_VELA}s")
print("📊 Estrategia: Cruce de Medias (MA5/MA10)")
print("⏳ Comenzando en 5 segundos... (Ctrl+C para detener)")
time.sleep(5)

while True:
    try:
        velas = API.get_candles(ACTIVO, TIEMPO_VELA, CANTIDAD_VELAS, time.time())
        if not velas:
            print("⚠️ No hay velas. Reintentando...")
            time.sleep(2)
            continue
        
        señal = analizar_velas(velas)
        precio_actual = velas[-1]['close']
        hora = time.strftime('%H:%M:%S')
        
        contador.registrar_senal(señal, precio_actual, hora)
        print(f"📊 #{contador.total_senales} | {hora} | Precio: {precio_actual:.5f} | Señal: {señal}")
        
        if contador.total_senales % 5 == 0 and contador.total_senales > 0:
            contador.mostrar_estadisticas()
        
        if señal != "ESPERAR":
            if hay_operacion_abierta():
                print("⚠️ Ya hay operación abierta. Esperando...")
            else:
                if contador.operaciones_ejecutadas >= MAX_OPERACIONES_POR_DIA:
                    print(f"⚠️ Límite de operaciones alcanzado.")
                    break
                
                if contador.perdida_total >= MAX_PERCARDIDA_DIARIA:
                    print(f"⚠️ Límite de pérdida alcanzado.")
                    break
                
                direccion = "call" if señal == "CALL" else "put"
                ejecutar_operacion(direccion, precio_actual)
        
        ahora = time.time()
        siguiente_cierre = ((ahora // TIEMPO_VELA) + 1) * TIEMPO_VELA
        espera = siguiente_cierre - ahora
        
        if espera > 0:
            print(f"⏳ Esperando {espera:.1f} segundos...")
            time.sleep(espera + 0.5)
    
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido")
        contador.mostrar_estadisticas()
        break
    except Exception as e:
        print(f"❌ ERROR: {e}")
        time.sleep(5)