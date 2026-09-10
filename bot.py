import time
import sys
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import json
import os

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

# IA - opcionales (si no están, funcionan con lógica interna)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN_OK = True
    print("✅ Scikit-learn disponible (IA activada)")
except ImportError:
    SKLEARN_OK = False
    print("⚠️ Scikit-learn no instalado. Usando IA interna simplificada.")
    print("   Instala con: pip install scikit-learn")

# =============================================
# CONFIGURACIÓN
# =============================================
EMAIL = "ppruevas526@gmail.com"
PASSWORD = "Crepes.2026*"

# 🔥 MÚLTIPLES PARES SIMULTÁNEOS
ACTIVOS = ["EURUSD", "GBPUSD", "USDJPY"]  # Añade los que quieras
TIEMPO_VELA = 300       # 5 minutos
CANTIDAD_VELAS = 150    # Más velas para mejor análisis IA
MONTO = 1.0
DURACION = 5            # 5 minutos de duración
MODO_REAL = False
MAX_OPERACIONES_POR_DIA = 30
MAX_PERDIDA_DIARIA = 10.0
MIN_CONFIANZA_IA = 0.65  # Umbral mínimo de confianza para operar

# =============================================
# 🧠 MÓDULO 1: ANÁLISIS DE VELAS BINARIAS
# =============================================
class AnalizadorVelas:
    """Analiza patrones, anatomía y estructura de velas"""

    @staticmethod
    def anatomia_vela(df):
        """Analiza la forma de la última vela"""
        v = df.iloc[-1]
        cuerpo = abs(v['close'] - v['open'])
        rango = v['high'] - v['low']
        mecha_sup = v['high'] - max(v['close'], v['open'])
        mecha_inf = min(v['close'], v['open']) - v['low']

        return {
            'cuerpo_pct': (cuerpo / rango * 100) if rango > 0 else 0,
            'mecha_sup_pct': (mecha_sup / rango * 100) if rango > 0 else 0,
            'mecha_inf_pct': (mecha_inf / rango * 100) if rango > 0 else 0,
            'alcista': v['close'] > v['open'],
            'cuerpo_grande': cuerpo > (rango * 0.6),
            'cuerpo_pequeno': cuerpo < (rango * 0.2),
            'doji': cuerpo < (rango * 0.1) if rango > 0 else False,
            'martillo': (mecha_inf > cuerpo * 2) and (mecha_sup < cuerpo * 0.5),
            'estrella_fugaz': (mecha_sup > cuerpo * 2) and (mecha_inf < cuerpo * 0.5),
        }

    @staticmethod
    def patron_doble_vela(df):
        """Detecta patrones de 2 velas (envolvente, harami, etc.)"""
        if len(df) < 3:
            return None

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_cuerpo = abs(prev['close'] - prev['open'])
        curr_cuerpo = abs(curr['close'] - curr['open'])

        # Envolvente alcista
        if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['close'] > prev['open'] and curr['open'] < prev['close']):
            return "ENVOLVENTE_ALCISTA"

        # Envolvente bajista
        if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
            curr['close'] < prev['open'] and curr['open'] > prev['close']):
            return "ENVOLVENTE_BAJISTA"

        # Harami (indecisión)
        if curr['high'] < prev['high'] and curr['low'] > prev['low']:
            if curr_cuerpo < prev_cuerpo * 0.5:
                return "HARAMI"

        return None

    @staticmethod
    def patron_triple_vela(df):
        """Detecta patrones de 3 velas (estrella mañana/tarde, 3 soldados)"""
        if len(df) < 4:
            return None

        v1, v2, v3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        # Estrella de la mañana (alcista)
        if (v1['close'] < v1['open'] and
            abs(v2['close'] - v2['open']) < abs(v1['close'] - v1['open']) * 0.5 and
            v3['close'] > v3['open'] and v3['close'] > (v1['open'] + v1['close']) / 2):
            return "ESTRELLA_MANANA"

        # Estrella del atardecer (bajista)
        if (v1['close'] > v1['open'] and
            abs(v2['close'] - v2['open']) < abs(v1['close'] - v1['open']) * 0.5 and
            v3['close'] < v3['open'] and v3['close'] < (v1['open'] + v1['close']) / 2):
            return "ESTRELLA_ATARDECER"

        # 3 soldados blancos
        if all(v['close'] > v['open'] for v in [v1, v2, v3]):
            if v2['close'] > v1['close'] and v3['close'] > v2['close']:
                return "TRES_SOLDADOS"

        # 3 cuervos negros
        if all(v['close'] < v['open'] for v in [v1, v2, v3]):
            if v2['close'] < v1['close'] and v3['close'] < v2['close']:
                return "TRES_CUERVOS"

        return None

    @staticmethod
    def estructura_mercado(df, lookback=20):
        """Detecta si hay tendencia o rango"""
        highs = df['high'].iloc[-lookback:].values
        lows = df['low'].iloc[-lookback:].values

        swing_highs, swing_lows = [], []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl:
                return "TENDENCIA_ALCISTA"
            elif lh and ll:
                return "TENDENCIA_BAJISTA"

        return "RANGO"


# =============================================
# 🧠 MÓDULO 2: INDICADORES TÉCNICOS
# =============================================
class Indicadores:
    """Calcula indicadores técnicos sin dependencias externas"""

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def adx(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        tr = Indicadores.atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def bollinger(series, period=20, std_dev=2):
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return ma + std * std_dev, ma, ma - std * std_dev

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = Indicadores.ema(series, fast)
        ema_slow = Indicadores.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicadores.ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def stochastic(df, k_period=14, d_period=3):
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, 1e-10)
        d = k.rolling(d_period).mean()
        return k, d


# =============================================
# 🧠 MÓDULO 3: ESTRATEGIAS
# =============================================
class Estrategias:
    """Conjunto de estrategias que devuelven (señal, confianza)"""

    @staticmethod
    def tendencia(df):
        """Seguimiento de tendencia con EMAs"""
        ema_fast = Indicadores.ema(df['close'], 10)
        ema_slow = Indicadores.ema(df['close'], 30)

        if len(ema_fast) < 3:
            return "ESPERAR", 0

        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
            return "CALL", 0.80
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
            return "PUT", 0.80
        return "ESPERAR", 0

    @staticmethod
    def reversion_media(df):
        """Reversión a la media con RSI + Bollinger"""
        rsi = Indicadores.rsi(df['close'])
        bb_up, bb_mid, bb_low = Indicadores.bollinger(df['close'])

        if len(rsi) < 2:
            return "ESPERAR", 0

        precio = df['close'].iloc[-1]
        rsi_v = rsi.iloc[-1]

        if rsi_v < 30 and precio < bb_low.iloc[-1]:
            return "CALL", 0.85
        elif rsi_v > 70 and precio > bb_up.iloc[-1]:
            return "PUT", 0.85
        return "ESPERAR", 0

    @staticmethod
    def ruptura(df, lookback=20):
        """Ruptura de máximos/mínimos"""
        if len(df) < lookback + 2:
            return "ESPERAR", 0

        high_n = df['high'].iloc[-lookback:-1].max()
        low_n = df['low'].iloc[-lookback:-1].min()
        precio = df['close'].iloc[-1]

        if precio > high_n:
            return "CALL", 0.75
        elif precio < low_n:
            return "PUT", 0.75
        return "ESPERAR", 0

    @staticmethod
    def momentum(df):
        """Momentum con MACD + Stochastic + RSI"""
        macd_line, signal_line, hist = Indicadores.macd(df['close'])
        k, d = Indicadores.stochastic(df)
        rsi = Indicadores.rsi(df['close'])
        adx, plus_di, minus_di = Indicadores.adx(df)

        if len(macd_line) < 2:
            return "ESPERAR", 0

        macd_v = macd_line.iloc[-1]
        sig_v = signal_line.iloc[-1]
        k_v = k.iloc[-1]
        d_v = d.iloc[-1]
        rsi_v = rsi.iloc[-1]
        adx_v = adx.iloc[-1]

        if pd.isna(adx_v) or pd.isna(rsi_v):
            return "ESPERAR", 0

        if (macd_v > sig_v and k_v > d_v and rsi_v > 50 and adx_v > 20):
            return "CALL", 0.85
        elif (macd_v < sig_v and k_v < d_v and rsi_v < 50 and adx_v > 20):
            return "PUT", 0.85
        return "ESPERAR", 0

    @staticmethod
    def patrones(df):
        """Estrategia basada en patrones de velas"""
        patron_triple = AnalizadorVelas.patron_triple_vela(df)
        patron_doble = AnalizadorVelas.patron_doble_vela(df)
        anatomia = AnalizadorVelas.anatomia_vela(df)

        # Triple vela tiene prioridad
        if patron_triple in ["ESTRELLA_MANANA", "TRES_SOLDADOS"]:
            return "CALL", 0.85
        elif patron_triple in ["ESTRELLA_ATARDECER", "TRES_CUERVOS"]:
            return "PUT", 0.85

        if patron_doble == "ENVOLVENTE_ALCISTA":
            return "CALL", 0.80
        elif patron_doble == "ENVOLVENTE_BAJISTA":
            return "PUT", 0.80

        # Velas individuales
        if anatomia['martillo']:
            return "CALL", 0.70
        elif anatomia['estrella_fugaz']:
            return "PUT", 0.70

        return "ESPERAR", 0

    @staticmethod
    def scalping(df):
        """Scalping con momentum corto"""
        roc = df['close'].pct_change(3) * 100
        rsi = Indicadores.rsi(df['close'], 7)

        if len(roc) < 1 or len(rsi) < 1:
            return "ESPERAR", 0

        roc_v = roc.iloc[-1]
        rsi_v = rsi.iloc[-1]

        if roc_v > 0.05 and 50 < rsi_v < 65:
            return "CALL", 0.70
        elif roc_v < -0.05 and 35 < rsi_v < 50:
            return "PUT", 0.70
        return "ESPERAR", 0


# =============================================
# 🧠 MÓDULO 4: DETECTOR DE RÉGIMEN DE MERCADO
# =============================================
class DetectorRegimen:
    """Detecta el régimen actual del mercado"""

    @staticmethod
    def detectar(df):
        adx, plus_di, minus_di = Indicadores.adx(df)
        atr = Indicadores.atr(df)
        bb_up, bb_mid, bb_low = Indicadores.bollinger(df['close'])

        if len(adx) < 1 or pd.isna(adx.iloc[-1]):
            return "DESCONOCIDO"

        adx_v = adx.iloc[-1]
        precio = df['close'].iloc[-1]
        atr_pct = (atr.iloc[-1] / precio * 100) if precio > 0 else 0
        bb_width = ((bb_up.iloc[-1] - bb_low.iloc[-1]) / bb_mid.iloc[-1]) if bb_mid.iloc[-1] > 0 else 0

        # Tendencia fuerte
        if adx_v > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                return "TENDENCIA_ALCISTA"
            else:
                return "TENDENCIA_BAJISTA"

        # Rango (baja volatilidad + ADX bajo)
        if adx_v < 20 and bb_width < 0.02:
            return "RANGO"

        # Volátil
        if atr_pct > 0.15:
            return "VOLATIL"

        return "NORMAL"


# =============================================
# 🧠 MÓDULO 5: SELECTOR DE ESTRATEGIA CON IA
# =============================================
class SelectorEstrategiaIA:
    """
    Selector inteligente que aprende qué estrategia funciona
    mejor según el régimen de mercado.

    Nota: _guardar_memoria() se llama desde hilos de operación en
    paralelo con el hilo principal, por lo que usa un lock propio
    para evitar escrituras concurrentes corruptas en el JSON.
    """

    def __init__(self, archivo_memoria="ia_memoria.json"):
        self.archivo = archivo_memoria
        self.memoria = self._cargar_memoria()
        self.rendimiento = self.memoria.get('rendimiento', {})
        self._lock = threading.Lock()

        # Estrategias disponibles
        self.estrategias = {
            'tendencia': Estrategias.tendencia,
            'reversion': Estrategias.reversion_media,
            'ruptura': Estrategias.ruptura,
            'momentum': Estrategias.momentum,
            'patrones': Estrategias.patrones,
            'scalping': Estrategias.scalping,
        }

    def _cargar_memoria(self):
        if os.path.exists(self.archivo):
            try:
                with open(self.archivo, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ No se pudo leer memoria IA existente: {e}")
        return {'rendimiento': {}, 'operaciones': []}

    def _guardar_memoria(self):
        """Escritura atómica: escribe a un temporal y luego renombra,
        así una interrupción a mitad de escritura no corrompe el archivo."""
        tmp = f"{self.archivo}.tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(self.memoria, f, indent=2)
            os.replace(tmp, self.archivo)
        except Exception as e:
            print(f"⚠️ No se pudo guardar memoria IA: {e}")

    def seleccionar(self, regimen):
        """
        Selecciona la mejor estrategia para el régimen actual
        basado en el historial de éxito
        """
        recomendadas = {
            "TENDENCIA_ALCISTA": ['tendencia', 'momentum', 'ruptura'],
            "TENDENCIA_BAJISTA": ['tendencia', 'momentum', 'ruptura'],
            "RANGO": ['reversion', 'patrones', 'scalping'],
            "VOLATIL": ['ruptura', 'momentum', 'patrones'],
            "NORMAL": ['tendencia', 'patrones', 'momentum'],
            "DESCONOCIDO": ['tendencia', 'patrones', 'momentum'],
        }

        candidatas = recomendadas.get(regimen, list(self.estrategias.keys()))

        mejor_estrategia = candidatas[0]
        mejor_score = -1

        with self._lock:
            for nombre in candidatas:
                clave = f"{regimen}::{nombre}"
                stats = self.rendimiento.get(clave, {'wins': 0, 'total': 0})

                if stats['total'] >= 3:
                    win_rate = stats['wins'] / stats['total']
                    score = win_rate
                else:
                    score = 0.5 - (candidatas.index(nombre) * 0.05)

                if score > mejor_score:
                    mejor_score = score
                    mejor_estrategia = nombre

        return mejor_estrategia, mejor_score

    def analizar_con_ia(self, df, regimen):
        """
        Ejecuta la mejor estrategia + todas las demás y combina señales
        usando pesos basados en el historial
        """
        mejor_nombre, _ = self.seleccionar(regimen)

        señales = {}
        for nombre, func in self.estrategias.items():
            try:
                señal, conf = func(df)
                señales[nombre] = {'señal': señal, 'conf': conf}
            except Exception as e:
                señales[nombre] = {'señal': 'ESPERAR', 'conf': 0}

        votos_call = 0
        votos_put = 0
        total_peso = 0

        with self._lock:
            for nombre, data in señales.items():
                if data['señal'] == 'ESPERAR':
                    continue

                clave = f"{regimen}::{nombre}"
                stats = self.rendimiento.get(clave, {'wins': 0, 'total': 0})

                if stats['total'] >= 3:
                    peso = stats['wins'] / stats['total']
                else:
                    peso = 0.5

                if nombre == mejor_nombre:
                    peso *= 1.3

                peso *= data['conf']
                total_peso += peso

                if data['señal'] == 'CALL':
                    votos_call += peso
                else:
                    votos_put += peso

        if total_peso == 0:
            return "ESPERAR", 0, mejor_nombre, señales

        if votos_call > votos_put:
            confianza = votos_call / total_peso
            return "CALL", confianza, mejor_nombre, señales
        elif votos_put > votos_call:
            confianza = votos_put / total_peso
            return "PUT", confianza, mejor_nombre, señales

        return "ESPERAR", 0, mejor_nombre, señales

    def registrar_resultado(self, regimen, estrategia, gano):
        """Registra el resultado para aprendizaje continuo.
        Se llama desde hilos de operación -> protegido por lock."""
        clave = f"{regimen}::{estrategia}"
        with self._lock:
            if clave not in self.rendimiento:
                self.rendimiento[clave] = {'wins': 0, 'total': 0}

            self.rendimiento[clave]['total'] += 1
            if gano:
                self.rendimiento[clave]['wins'] += 1

            self.memoria['rendimiento'] = self.rendimiento
            wins = self.rendimiento[clave]['wins']
            total = self.rendimiento[clave]['total']
            self._guardar_memoria()

        win_rate = wins / total
        print(f"🧠 IA aprendió: {clave} → {win_rate*100:.0f}% éxito ({wins}/{total})")


# =============================================
# CONTADOR DE SEÑALES (thread-safe para multi-par + operaciones async)
# =============================================
class ContadorSenales:
    def __init__(self):
        self._lock = threading.Lock()
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

    def registrar_senal(self, activo, tipo, precio, hora, regimen, estrategia, confianza):
        with self._lock:
            self.total_senales += 1
            if tipo == "CALL":
                self.senales_call += 1
            elif tipo == "PUT":
                self.senales_put += 1
            else:
                self.senales_esperar += 1

            self.historial_senales.append({
                'fecha': hora,
                'activo': activo,
                'tipo': tipo,
                'precio': precio,
                'regimen': regimen,
                'estrategia': estrategia,
                'confianza': confianza
            })

            if len(self.historial_senales) > 200:
                self.historial_senales.pop(0)

            return self.total_senales

    def registrar_operacion_iniciada(self):
        """Reserva un slot de operación en el momento de lanzarla
        (no cuando termina), para que el límite diario se respete
        aunque haya varias operaciones en vuelo a la vez."""
        with self._lock:
            self.operaciones_ejecutadas += 1
            return self.operaciones_ejecutadas

    def registrar_resultado_operacion(self, activo, direccion, precio, orden_id, resultado, ganancia):
        with self._lock:
            if resultado == "ganada":
                self.operaciones_ganadas += 1
                self.ganancia_total += ganancia
            elif resultado == "perdida":
                self.operaciones_perdidas += 1
                self.perdida_total += abs(ganancia)

            self.historial_operaciones.append({
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'activo': activo,
                'direccion': direccion,
                'precio': precio,
                'orden_id': orden_id,
                'resultado': resultado,
                'ganancia': ganancia
            })

            if len(self.historial_operaciones) > 200:
                self.historial_operaciones.pop(0)

            return self.perdida_total

    def actualizar_balance(self, balance):
        with self._lock:
            if self.balance_inicial == 0:
                self.balance_inicial = balance
            self.balance_actual = balance

    def limite_operaciones_alcanzado(self):
        with self._lock:
            return self.operaciones_ejecutadas >= MAX_OPERACIONES_POR_DIA

    def limite_perdida_alcanzado(self):
        with self._lock:
            return self.perdida_total >= MAX_PERDIDA_DIARIA

    def mostrar_estadisticas(self):
        stats = self.obtener_estadisticas()
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS DEL BOT")
        print("="*50)
        print(f"🔹 SEÑALES TOTALES: {stats['total_senales']}")
        print(f"   • CALL: {stats['senales_call']} | PUT: {stats['senales_put']} | ESPERAR: {stats['senales_esperar']}")
        print("-"*50)
        print(f"🔹 OPERACIONES: {stats['operaciones_ejecutadas']}")
        print(f"   • Ganadas: {stats['operaciones_ganadas']} | Perdidas: {stats['operaciones_perdidas']}")
        print(f"   • Tasa de aciertos: {stats['tasa_aciertos']}%")
        print("-"*50)
        print(f"🔹 FINANZAS:")
        print(f"   • Ganancia total: +${stats['ganancia_total']}")
        print(f"   • Pérdida total: -${stats['perdida_total']}")
        print(f"   • Beneficio neto: ${stats['beneficio_neto']}")
        print(f"   • Beneficio %: {stats['beneficio_porcentaje']}%")
        print("-"*50)
        print(f"🔹 BALANCE:")
        print(f"   • Inicial: ${stats['balance_inicial']} | Actual: ${stats['balance_actual']}")
        print("="*50 + "\n")

    def obtener_estadisticas(self):
        with self._lock:
            tasa = 0
            if self.operaciones_ejecutadas > 0:
                tasa = (self.operaciones_ganadas / self.operaciones_ejecutadas) * 100

            neto = self.ganancia_total - self.perdida_total
            pct = 0
            if self.balance_inicial > 0:
                pct = (neto / self.balance_inicial) * 100

            return {
                'total_senales': self.total_senales,
                'senales_call': self.senales_call,
                'senales_put': self.senales_put,
                'senales_esperar': self.senales_esperar,
                'operaciones_ejecutadas': self.operaciones_ejecutadas,
                'operaciones_ganadas': self.operaciones_ganadas,
                'operaciones_perdidas': self.operaciones_perdidas,
                'tasa_aciertos': round(tasa, 2),
                'ganancia_total': round(self.ganancia_total, 2),
                'perdida_total': round(self.perdida_total, 2),
                'beneficio_neto': round(neto, 2),
                'beneficio_porcentaje': round(pct, 2),
                'balance_inicial': round(self.balance_inicial, 2),
                'balance_actual': round(self.balance_actual, 2)
            }


# =============================================
# 🗣️ MENSAJES DE ERROR LEGIBLES
# =============================================
# Traduce las respuestas crípticas de la API (dicts con 'code'/'message')
# a un texto entendible, y da el motivo probable + qué hacer.
_TRADUCCIONES_ERROR = {
    'invalid instrument': (
        'el instrumento no está disponible para operar en este momento '
        '(mercado cerrado o sin opciones digitales habilitadas para esta cuenta)'
    ),
    'invalid amount': 'el monto configurado en MONTO no es válido para este instrumento',
    'not enough money': 'no hay saldo suficiente en la cuenta para esta operación',
    'active is suspended': 'este instrumento está suspendido temporalmente por el broker',
}


def formatear_error_orden(respuesta):
    """Convierte la respuesta de error de buy_digital_spot_v2 en un
    mensaje legible. Acepta tanto dicts {'code':..,'message':..} como
    strings u otros tipos, por si la librería cambia el formato."""
    if isinstance(respuesta, dict):
        codigo = respuesta.get('code', 'desconocido')
        mensaje_original = str(respuesta.get('message', '')).strip()
        explicacion = _TRADUCCIONES_ERROR.get(mensaje_original.lower(), mensaje_original or 'motivo no especificado')
        return f"{explicacion} (código interno: {codigo})"
    return str(respuesta)


# =============================================
# GESTOR DE OPERACIONES ABIERTAS (clave del cambio async)
# =============================================
class GestorOperaciones:
    """
    Lleva el registro de qué activos tienen una operación en curso
    y lanza/gestiona los hilos que esperan el resultado, sin bloquear
    el bucle principal de análisis.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._activos_ocupados = set()
        self._hilos = []

    def esta_ocupado(self, activo):
        with self._lock:
            return activo in self._activos_ocupados

    def _marcar_ocupado(self, activo):
        with self._lock:
            self._activos_ocupados.add(activo)

    def _liberar(self, activo):
        with self._lock:
            self._activos_ocupados.discard(activo)

    def lanzar_operacion(self, activo, direccion, precio, regimen, estrategia):
        """Lanza la operación en un hilo aparte y retorna inmediatamente."""
        self._marcar_ocupado(activo)
        hilo = threading.Thread(
            target=self._ejecutar_y_esperar,
            args=(activo, direccion, precio, regimen, estrategia),
            daemon=True,
        )
        with self._lock:
            self._hilos.append(hilo)
        hilo.start()

    def _ejecutar_y_esperar(self, activo, direccion, precio, regimen, estrategia):
        """Corre en un hilo de fondo: abre la orden, espera su cierre
        y registra el resultado, sin bloquear el análisis de otros pares."""
        try:
            print(f"🚀 EJECUTANDO {direccion.upper()} en {activo} @ {precio:.5f}")
            print(f"   🧠 Régimen: {regimen} | Estrategia: {estrategia}")

            try:
                status, orden_id = API.buy_digital_spot_v2(activo, MONTO, direccion, DURACION)
            except Exception as e:
                print(f"❌ Error al ejecutar en {activo}: {e}")
                return

            if not status:
                mensaje_legible = formatear_error_orden(orden_id)
                print(f"❌ No se pudo abrir la orden en {activo}: {mensaje_legible}")

                # Si el motivo es que el instrumento no está disponible,
                # forzamos un refresco de disponibilidad para que el
                # siguiente ciclo ya no lo vuelva a intentar de inmediato.
                if isinstance(orden_id, dict) and \
                   str(orden_id.get('message', '')).strip().lower() == 'invalid instrument':
                    try:
                        obtener_activos_disponibles(forzar=True)
                    except Exception:
                        pass
                return

            print(f"✅ Orden enviada en {activo} | ID: {orden_id}")
            time.sleep(DURACION * 60 + 5)

            check_close, win_money = API.check_win_digital_v2(orden_id)
            if not check_close:
                print(f"⚠️ {activo}: no se pudo confirmar el cierre de la orden {orden_id}")
                return

            gano = float(win_money) > 0
            if gano:
                print(f"💰 ¡GANANCIA en {activo}! +{win_money:.2f} USD")
                contador.registrar_resultado_operacion(activo, direccion, precio, orden_id, "ganada", float(win_money))
            else:
                print(f"📉 Pérdida en {activo}: {MONTO:.2f} USD")
                contador.registrar_resultado_operacion(activo, direccion, precio, orden_id, "perdida", MONTO)

            selector_ia.registrar_resultado(regimen, estrategia, gano)

            nuevo_balance = API.get_balance()
            if nuevo_balance:
                contador.actualizar_balance(nuevo_balance)
                print(f"💰 Balance actual: {nuevo_balance:.2f} USD")

        except Exception as e:
            print(f"⚠️ Error en hilo de operación de {activo}: {e}")
        finally:
            self._liberar(activo)

    def esperar_pendientes(self, timeout=None):
        """Usado al apagar el bot: espera a que los hilos en vuelo terminen."""
        with self._lock:
            hilos = list(self._hilos)
        for h in hilos:
            h.join(timeout=timeout)


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

if not MODO_REAL:
    API.change_balance('PRACTICE')
    print("💰 Modo DEMO activado")
else:
    API.change_balance('REAL')
    print("⚠️ Modo REAL activado")

balance = API.get_balance()
print(f"✅ Conectado | Saldo: {balance:.2f} USD")


# =============================================
# 🩹 PARCHE DEFENSIVO: bug conocido de iqoptionapi
# =============================================
# Un hilo interno de la librería (usado al pedir horarios de digitales)
# a veces recibe None del servidor y hace data["underlying"] sin
# validar, reventando con:
#   TypeError: 'NoneType' object is not subscriptable
# Envolvemos el método en la instancia para que nunca devuelva None
# y reintente un par de veces si la primera consulta falla.
_get_digital_underlying_original = API.get_digital_underlying_list_data


def _get_digital_underlying_seguro(intentos=3, espera=1.0):
    for intento in range(1, intentos + 1):
        try:
            data = _get_digital_underlying_original()
        except Exception as e:
            print(f"⚠️ Intento {intento}/{intentos}: error consultando subyacentes digitales: {e}")
            data = None

        if data and isinstance(data, dict) and data.get('underlying') is not None:
            return data

        if intento < intentos:
            time.sleep(espera)

    print("⚠️ No se pudo obtener la lista de subyacentes digitales tras varios intentos; "
          "se devuelve lista vacía para evitar que el bot se caiga.")
    return {"underlying": []}


API.get_digital_underlying_list_data = _get_digital_underlying_seguro

# Además, si algún otro hilo interno de la librería revienta igual,
# que se vea como un aviso claro y no como una traza intimidante.
_excepthook_original = threading.excepthook


def _excepthook_amigable(args):
    if args.exc_type is TypeError and 'NoneType' in str(args.exc_value):
        print(f"⚠️ Hipo de conexión con IQ Option (hilo interno de la librería). "
              f"El bot sigue corriendo; si se repite mucho, revisa tu conexión a internet.")
        return
    _excepthook_original(args)


threading.excepthook = _excepthook_amigable


contador = ContadorSenales()
contador.actualizar_balance(balance)

# Inicializar IA
selector_ia = SelectorEstrategiaIA()
print(f"🧠 IA cargada con {len(selector_ia.rendimiento)} patrones aprendidos")

# Gestor de operaciones asíncronas
gestor_operaciones = GestorOperaciones()


# =============================================
# 🧭 VERIFICADOR DE DISPONIBILIDAD DE INSTRUMENTOS
# =============================================
# Antes de intentar cualquier orden, consultamos si el mercado digital
# está abierto para cada activo configurado. Evita el error
# "invalid instrument" y avisa dónde SÍ se puede entrar ahora mismo.
CACHE_DISPONIBILIDAD_SEGUNDOS = 300  # refrescar cada 5 minutos

_disponibilidad_cache = {'set': set(), 'actualizado': 0.0}
_disponibilidad_lock = threading.Lock()


def obtener_activos_disponibles(forzar=False):
    """Devuelve el subconjunto de ACTIVOS que tiene digitales abiertas
    ahora mismo, y avisa por consola el motivo de los que no califican.
    También sugiere alternativas operables que no están en ACTIVOS."""
    ahora = time.time()
    with _disponibilidad_lock:
        if not forzar and (ahora - _disponibilidad_cache['actualizado']) < CACHE_DISPONIBILIDAD_SEGUNDOS:
            return set(_disponibilidad_cache['set'])

    disponibles = set()
    try:
        open_time = API.get_all_open_time()
        digital_open = open_time.get('digital', {})

        subyacentes = API.get_digital_underlying_list_data().get('underlying', [])
        nombres_digital = {s['name'] for s in subyacentes}

        for activo in ACTIVOS:
            abierto = digital_open.get(activo, {}).get('open', False)
            en_lista = activo in nombres_digital

            if abierto and en_lista:
                disponibles.add(activo)
            else:
                motivos = []
                if not abierto:
                    motivos.append("mercado cerrado")
                if not en_lista:
                    motivos.append("sin digitales habilitadas para esta cuenta")
                print(f"🚫 {activo}: NO disponible para operar ahora ({', '.join(motivos)})")

        # Sugerir alternativas operables ahora mismo, aunque no estén en ACTIVOS
        try:
            abiertos_ahora = {k for k, v in digital_open.items() if v.get('open')}
            operables_ahora = sorted(abiertos_ahora & nombres_digital)
            sugerencias = [a for a in operables_ahora if a not in ACTIVOS]
            if not disponibles and sugerencias:
                print(f"💡 Ninguno de tus ACTIVOS configurados está operable ahora. "
                      f"Sí están disponibles (agrégalos a ACTIVOS si querés usarlos): "
                      f"{sugerencias[:15]}")
        except Exception:
            pass

    except Exception as e:
        print(f"⚠️ No se pudo verificar disponibilidad de instrumentos ({e}). "
              f"Se asumen todos disponibles para no frenar el bot.")
        disponibles = set(ACTIVOS)

    with _disponibilidad_lock:
        _disponibilidad_cache['set'] = disponibles
        _disponibilidad_cache['actualizado'] = ahora

    return set(disponibles)


# =============================================
# ANÁLISIS INTEGRADO DE UN PAR
# =============================================
def analizar_activo(activo, velas):
    """Analiza un activo con IA y devuelve señal + contexto"""
    df = pd.DataFrame(velas, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) < 30:
        return {
            'activo': activo,
            'señal': 'ESPERAR',
            'confianza': 0,
            'regimen': 'DESCONOCIDO',
            'estrategia': None,
            'precio': df['close'].iloc[-1] if len(df) else 0,
            'detalles': {}
        }

    # 1. Detectar régimen
    regimen = DetectorRegimen.detectar(df)

    # 2. IA selecciona y combina estrategias
    señal, confianza, estrategia, detalles = selector_ia.analizar_con_ia(df, regimen)

    # 3. Contexto extra
    anatomia = AnalizadorVelas.anatomia_vela(df)
    estructura = AnalizadorVelas.estructura_mercado(df)

    return {
        'activo': activo,
        'señal': señal,
        'confianza': confianza,
        'regimen': regimen,
        'estrategia': estrategia,
        'precio': df['close'].iloc[-1],
        'estructura': estructura,
        'anatomia': anatomia,
        'detalles': detalles
    }


# =============================================
# BUCLE PRINCIPAL MULTI-PAR (no bloqueante)
# =============================================
print(f"\n🚀 BOT IA INICIADO")
print(f"📊 Activos: {', '.join(ACTIVOS)}")
print(f"⏱️ Vela: {TIEMPO_VELA}s | Duración op: {DURACION} min")
print(f"🧠 Estrategias IA: {list(selector_ia.estrategias.keys())}")
print(f"🎯 Confianza mínima: {MIN_CONFIANZA_IA*100:.0f}%")
print("⏳ Comenzando en 5 segundos... (Ctrl+C para detener)")
time.sleep(5)

detenido_por_limite = False

while True:
    try:
        # 🔌 Verificar conexión antes de cada ciclo; los hipos de
        # conexión son la causa más común de errores raros de la API.
        if not API.check_connect():
            print("🔌 Conexión perdida, reconectando...")
            API.connect()
            time.sleep(2)
            if not API.check_connect():
                print("⚠️ No se pudo reconectar, se reintenta en el próximo ciclo.")
                time.sleep(5)
                continue
            print("✅ Reconectado")

        # 🧭 Verificar qué activos están realmente operables ahora mismo
        # (mercado abierto + digitales habilitadas). Se cachea unos minutos
        # para no saturar la API con esta consulta en cada ciclo.
        activos_operables = obtener_activos_disponibles()
        if activos_operables:
            print(f"✅ Operables ahora: {sorted(activos_operables)}")
        else:
            print("🚫 Ningún activo configurado está operable en este momento.")

        # Obtener velas de todos los activos (las operaciones en curso
        # de otros pares NO bloquean este análisis)
        analisis_activos = []

        for activo in ACTIVOS:
            try:
                velas = API.get_candles(activo, TIEMPO_VELA, CANTIDAD_VELAS, time.time())
                if not velas or len(velas) < 30:
                    print(f"⚠️ {activo}: pocas velas, saltando")
                    continue

                resultado = analizar_activo(activo, velas)
                analisis_activos.append(resultado)

                hora = time.strftime('%H:%M:%S')
                ocupado = " (operación en curso)" if gestor_operaciones.esta_ocupado(activo) else ""
                print(f"📊 {hora} | {activo} | {resultado['precio']:.5f} | "
                      f"{resultado['señal']} ({resultado['confianza']*100:.0f}%) | "
                      f"Régimen: {resultado['regimen']} | Estrategia: {resultado['estrategia']}{ocupado}")

                total = contador.registrar_senal(
                    activo, resultado['señal'], resultado['precio'],
                    hora, resultado['regimen'], resultado['estrategia'],
                    resultado['confianza']
                )
            except Exception as e:
                print(f"❌ Error analizando {activo}: {e}")

        # Estadísticas cada 5 señales
        if contador.obtener_estadisticas()['total_senales'] % 5 == 0 and contador.obtener_estadisticas()['total_senales'] > 0:
            contador.mostrar_estadisticas()

        # Buscar oportunidades operables: solo en activos que no tengan
        # ya una operación en curso Y que estén confirmados como operables
        # (evita el error "invalid instrument" al intentar la orden)
        oportunidades = [
            a for a in analisis_activos
            if a['señal'] != 'ESPERAR'
            and a['confianza'] >= MIN_CONFIANZA_IA
            and not gestor_operaciones.esta_ocupado(a['activo'])
            and a['activo'] in activos_operables
        ]

        # Señales que sí calificaban por estrategia pero el mercado
        # está cerrado para ese instrumento: avisar por qué no se ejecutan
        bloqueadas_por_mercado = [
            a for a in analisis_activos
            if a['señal'] != 'ESPERAR'
            and a['confianza'] >= MIN_CONFIANZA_IA
            and a['activo'] not in activos_operables
        ]
        for b in bloqueadas_por_mercado:
            print(f"⏭️ {b['activo']}: señal {b['señal']} ({b['confianza']*100:.0f}%) "
                  f"válida pero el mercado está cerrado para este instrumento, se omite.")

        if oportunidades:
            oportunidades.sort(key=lambda x: x['confianza'], reverse=True)

            if contador.limite_perdida_alcanzado():
                print("⚠️ Límite de pérdida diaria alcanzado. No se abren nuevas operaciones.")
                detenido_por_limite = True
            elif contador.limite_operaciones_alcanzado():
                print("⚠️ Límite de operaciones diarias alcanzado. No se abren nuevas operaciones.")
                detenido_por_limite = True
            else:
                # Puede lanzar más de una oportunidad por ciclo (una por
                # activo libre), ya que cada una corre en su propio hilo
                for op in oportunidades:
                    if contador.limite_operaciones_alcanzado() or contador.limite_perdida_alcanzado():
                        break
                    print(f"\n🎯 OPORTUNIDAD: {op['activo']} → {op['señal']} "
                          f"(confianza {op['confianza']*100:.0f}%)")
                    contador.registrar_operacion_iniciada()
                    direccion = "call" if op['señal'] == "CALL" else "put"
                    gestor_operaciones.lanzar_operacion(
                        op['activo'], direccion, op['precio'],
                        op['regimen'], op['estrategia']
                    )
        else:
            print(f"⏸️ Sin oportunidades nuevas (confianza mínima {MIN_CONFIANZA_IA*100:.0f}%)")

        if detenido_por_limite:
            print("🧭 Esperando a que cierren las operaciones abiertas antes de detener el bot...")
            gestor_operaciones.esperar_pendientes()
            break

        # Esperar cierre de vela (el bucle sigue analizando mientras
        # las operaciones lanzadas corren en paralelo)
        ahora = time.time()
        siguiente_cierre = ((ahora // TIEMPO_VELA) + 1) * TIEMPO_VELA
        espera = siguiente_cierre - ahora
        if espera > 0:
            print(f"⏳ Esperando {espera:.1f} segundos...\n")
            time.sleep(espera + 0.5)

    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
        contador.mostrar_estadisticas()
        print("🧭 Esperando a que cierren las operaciones en curso (Ctrl+C de nuevo para forzar salida)...")
        try:
            gestor_operaciones.esperar_pendientes()
        except KeyboardInterrupt:
            print("⚠️ Saliendo sin esperar el cierre de operaciones abiertas.")
        print(f"\n🧠 IA aprendió {len(selector_ia.rendimiento)} combinaciones régimen/estrategia")
        break
    except Exception as e:
        print(f"❌ ERROR: {e}")
        time.sleep(5)