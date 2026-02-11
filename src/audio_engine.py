"""
Audio Engine - Backend de Procesamiento de Voz
Maneja toda la lógica de grabación, DSP, síntesis y transcripción
8kHz / 8-bit con Pre-énfasis, Dithering y Optimización Warm-up & Pre-roll
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import librosa
import threading
from scipy.signal import butter, filtfilt
from datetime import datetime


class AudioEngine:
    """
    Motor de audio para procesamiento de voz telefónica 8kHz/8-bit
    """
    
    def __init__(self, sample_rate=8000, output_folder="./output"):
        """
        Inicializar el motor de audio
        
        Args:
            sample_rate: Frecuencia de muestreo (default: 8000 Hz)
            output_folder: Carpeta para guardar archivos
        """
        self.fs = sample_rate
        self.output_folder = output_folder
        self.is_recording = False
        self.current_audio = None
        self.current_audio_processed = None
        
        # Configuración DSP
        self.pre_emphasis_coef = 0.97
        self.preroll_duration = 0.5  # 500ms
        self.warmup_duration = 0.3   # 300ms
        self.trim_threshold_db = 15
        
        # Callbacks para comunicación con UI
        self.log_callback = None
        self.status_callback = None
    
    def set_log_callback(self, callback):
        """Configurar callback para enviar logs a la UI"""
        self.log_callback = callback
    
    def set_status_callback(self, callback):
        """Configurar callback para actualizar estado de la UI"""
        self.status_callback = callback
    
    def _log(self, mensaje):
        """Enviar log a la UI si hay callback configurado"""
        if self.log_callback:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_callback(f"• [{timestamp}] {mensaje}")
    
    def _update_status(self, status, color=None, text=None):
        """Actualizar estado en la UI"""
        if self.status_callback:
            self.status_callback(status, color, text)
    
    def generar_beep(self, frecuencia=800, duracion=0.15):
        """
        Genera un beep corto
        
        Args:
            frecuencia: Frecuencia del tono en Hz
            duracion: Duración en segundos
            
        Returns:
            array numpy con el beep generado
        """
        t = np.linspace(0, duracion, int(self.fs * duracion))
        beep = 0.3 * np.sin(2 * np.pi * frecuencia * t)
        
        # Aplicar fade in/out para evitar clicks
        fade = int(self.fs * 0.01)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        
        return beep
    
    def warmup_audio_buffer(self):
        """
        Warm-up del buffer de audio para estabilizar el driver
        Abre un flujo silencioso durante 300ms antes de la grabación real
        """
        self._log("🔧 Inicializando buffer de audio (warm-up)...")
        
        # Configurar latencia baja ANTES del warm-up
        sd.default.latency = 'low'
        
        # Callback vacío para el warm-up
        def warmup_callback(indata, frames, time, status):
            pass
        
        # Abrir stream de warm-up
        with sd.InputStream(
            samplerate=self.fs, 
            channels=1, 
            callback=warmup_callback,
            blocksize=1024
        ):
            threading.Event().wait(self.warmup_duration)
        
        self._log("✓ Buffer estabilizado y listo")
    
    def cuenta_regresiva(self):
        """
        Cuenta regresiva con indicadores visuales y sonoros
        SECUENCIA: Visual 3-2-1 → Beep auditivo → GRABACIÓN
        """
        # FASE 1: Conteo VISUAL
        for i in range(3, 0, -1):
            self._update_status("preparing", "#FF9500", f"⏱️ PREPARANDO... {i}")
            self._log(f"Preparando... {i}")
            threading.Event().wait(0.8)
        
        # FASE 2: Beeps auditivos
        for i in range(3, 0, -1):
            self._log(f"Grabación en {i}...")
            beep = self.generar_beep(frecuencia=600)
            sd.play(beep, self.fs)
            sd.wait()
            threading.Event().wait(0.6)
        
        # FASE 3: Beep de inicio
        self._update_status("recording", "#FF3B30", "🔴 ¡HABLA AHORA!")
        self._log("¡HABLA AHORA!")
        beep_inicio = self.generar_beep(frecuencia=1000, duracion=0.2)
        sd.play(beep_inicio, self.fs)
        sd.wait()
    
    def grabar_audio(self, duracion):
        """
        Grabar audio con pre-roll para eliminar recorte inicial
        
        Args:
            duracion: Duración solicitada en segundos
            
        Returns:
            numpy array con el audio grabado
        """
        # Configurar latencia baja EXPLÍCITAMENTE
        sd.default.latency = 'low'
        
        # Agregar pre-roll
        total_duration = duracion + self.preroll_duration
        total_samples = int(total_duration * self.fs)
        
        # Log sincronizado con inicio de buffer
        self._log(">>> INICIO DE LA GRABACIÓN (buffer activo)")
        self.is_recording = True
        
        # Grabar
        audio_raw = sd.rec(total_samples, samplerate=self.fs, channels=1, dtype='float32')
        sd.wait()
        
        self.is_recording = False
        self._log("<<< FINAL DE LA GRABACIÓN")
        
        return audio_raw.flatten()
    
    def procesar_audio(self, audio_raw):
        """
        Procesar audio: filtrado, trim, pre-énfasis
        
        Args:
            audio_raw: Audio crudo de la grabación
            
        Returns:
            tuple: (audio_procesado, duracion_segundos, inicio_ms)
        """
        self._update_status("processing", "#34C759", "✓ PROCESANDO...")
        
        # 1. Filtro paso-alto para eliminar ruido de baja frecuencia
        b, a = butter(3, 80 / (self.fs / 2), btype='high')
        y = filtfilt(b, a, audio_raw)
        
        # 2. Trim dinámico con umbral de 15dB
        y, index = librosa.effects.trim(y, top_db=self.trim_threshold_db)
        inicio_ms = (index[0] / self.fs) * 1000
        self._log(f"✓ Inicio de voz detectado a {inicio_ms:.0f}ms (trim: {self.trim_threshold_db}dB)")
        
        # 3. Calcular duración real
        duracion_real = len(y) / self.fs
        self._log(f"📊 Duración del audio: {duracion_real:.2f} segundos ({len(y)} muestras)")
        
        # 4. Pre-énfasis para realzar altas frecuencias
        y_preemph = np.append(y[0], y[1:] - self.pre_emphasis_coef * y[:-1])
        self._log("✓ Pre-énfasis aplicado (realce de consonantes)")
        
        # 5. Normalización
        if np.max(np.abs(y_preemph)) > 0:
            y_preemph = y_preemph / np.max(np.abs(y_preemph)) * 0.95
        
        return y_preemph, duracion_real, inicio_ms
    
    def guardar_audio(self, audio, filepath, formato='PCM_U8'):
        """
        Guardar audio en formato 8-bit
        
        Args:
            audio: Señal de audio
            filepath: Ruta del archivo
            formato: Formato de codificación (default: PCM_U8)
        """
        sf.write(filepath, audio, self.fs, subtype=formato)
        self._log(f"✓ Audio guardado: {filepath}")
    
    def guardar_matriz_csv(self, audio, filepath):
        """
        Guardar matriz de audio en CSV
        
        Args:
            audio: Señal de audio
            filepath: Ruta del archivo CSV
        """
        np.savetxt(filepath, audio, delimiter=',', header='muestras_8bit')
        self._log(f"Matriz {len(audio)}x1 guardada en CSV.")
    
    def transcribir_audio(self, filepath_wav, idioma="es-MX"):
        """
        Transcribir audio a texto usando Google Speech Recognition
        
        Args:
            filepath_wav: Ruta del archivo .wav
            idioma: Código de idioma (default: es-MX)
            
        Returns:
            str: Texto transcrito o None si falla
        """
        r = sr.Recognizer()
        try:
            with sr.AudioFile(filepath_wav) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=idioma)
                self._log(f"Transcripción: {text}")
                return text
        except Exception as e:
            self._log("⚠ No se pudo transcribir (habla más claro o aumenta duración).")
            return None
    
    def sintetizar_voz(self, audio_original):
        """
        Síntesis voz-a-voz con máxima calidad en 8-bit
        
        Args:
            audio_original: Señal de audio procesada
            
        Returns:
            numpy array con audio sintetizado
        """
        self._log("Generando síntesis optimizada para 8kHz...")
        
        # PASO 1: Pitch Shift con precisión optimizada
        y_synth = librosa.effects.pitch_shift(
            audio_original, 
            sr=self.fs, 
            n_steps=2.0,
            bins_per_octave=12
        )
        self._log("✓ Pitch shift de 2.0 semitonos aplicado (bins_per_octave=12)")
        
        # PASO 2: Normalización a -1dB
        target_db = -1.0
        normalization_factor = 10 ** (target_db / 20.0)  # ≈ 0.8913
        
        if np.max(np.abs(y_synth)) > 0:
            y_synth_normalized = y_synth / np.max(np.abs(y_synth)) * normalization_factor
        else:
            y_synth_normalized = y_synth
        
        self._log(f"✓ Normalizado a {target_db}dB (factor: {normalization_factor:.4f})")
        
        # PASO 3: Dithering para reducir distorsión de cuantización
        dither_amplitude = 1.0 / (2 * 256)  # Medio LSB
        dither_noise = np.random.uniform(
            -dither_amplitude, 
            dither_amplitude, 
            len(y_synth_normalized)
        )
        y_synth_dithered = y_synth_normalized + dither_noise
        self._log(f"✓ Dithering aplicado (amplitud: {dither_amplitude:.6f})")
        
        # PASO 4: Clip para evitar overflow
        y_synth_final = np.clip(y_synth_dithered, -1.0, 1.0)
        
        return y_synth_final
    
    def reproducir_audio(self, audio):
        """
        Reproducir audio
        
        Args:
            audio: Señal de audio a reproducir
        """
        self._log("🔊 Reproduciendo síntesis optimizada...")
        sd.play(audio, self.fs)
        sd.wait()
    
    def proceso_completo(self, nombre_base, duracion, output_folder=None):
        """
        Proceso completo de grabación, procesamiento y síntesis
        
        Args:
            nombre_base: Nombre base para los archivos
            duracion: Duración de la grabación en segundos
            output_folder: Carpeta de salida (opcional)
            
        Returns:
            dict con rutas de archivos generados y señal procesada
        """
        if output_folder is None:
            output_folder = self.output_folder
        
        # Definir rutas de archivos
        import os
        ruta_wav = os.path.join(output_folder, f"{nombre_base}.wav")
        ruta_csv = os.path.join(output_folder, f"{nombre_base}_matriz.csv")
        ruta_txt = os.path.join(output_folder, f"{nombre_base}.txt")
        ruta_synth = os.path.join(output_folder, f"{nombre_base}_synth.wav")
        
        try:
            # 1. Warm-up del buffer
            self.warmup_audio_buffer()
            
            # 2. Cuenta regresiva
            self.cuenta_regresiva()
            
            # 3. Grabación con pre-roll
            audio_raw = self.grabar_audio(duracion)
            
            # 4. Procesamiento
            audio_procesado, duracion_real, inicio_ms = self.procesar_audio(audio_raw)
            self.current_audio_processed = audio_procesado
            
            # 5. Guardar audio original
            self.guardar_audio(audio_procesado, ruta_wav)
            
            # 6. Guardar matriz CSV
            self.guardar_matriz_csv(audio_procesado, ruta_csv)
            
            # 7. Transcripción
            texto = self.transcribir_audio(ruta_wav)
            if texto:
                with open(ruta_txt, "w", encoding='utf-8') as f:
                    f.write(texto)
            
            # 8. Síntesis
            audio_synth = self.sintetizar_voz(audio_procesado)
            
            # 9. Guardar síntesis
            self.guardar_audio(audio_synth, ruta_synth)
            self._log("✓ Audio sintetizado guardado en 8-bit PCM_U8")
            
            # 10. Reproducir síntesis
            self.reproducir_audio(audio_synth)
            
            self._log("✓ Proceso completado exitosamente.")
            
            return {
                'wav': ruta_wav,
                'csv': ruta_csv,
                'txt': ruta_txt,
                'synth': ruta_synth,
                'audio_procesado': audio_procesado,
                'duracion': duracion_real,
                'inicio_ms': inicio_ms,
                'success': True
            }
            
        except Exception as e:
            self._log(f"⚠ SEÑAL ERROR CRÍTICO: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'audio_procesado': self.current_audio_processed
            }
