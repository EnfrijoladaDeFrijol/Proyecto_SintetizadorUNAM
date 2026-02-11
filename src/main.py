"""
Main - Interfaz Gráfica del Sintetizador de Voz
Frontend con CustomTkinter que utiliza AudioEngine para todo el procesamiento
"""

import customtkinter as ctk
import os
import threading
import numpy as np

# Importaciones para gráficas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display

# Importar el motor de audio
from audio_engine import AudioEngine


# Configuración visual global
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")
plt.style.use('seaborn-v0_8-whitegrid')


class SintetizadorApp(ctk.CTk):
    """
    Aplicación GUI para el Sintetizador de Voz 8kHz/8-bit
    """
    
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.title("Voice Synth Studio | Pro 8kHz 8-bit")
        self.geometry("950x850")
        self.configure(fg_color="#F5F5F7")
        
        # Configurar carpeta de salida
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.normpath(os.path.join(script_dir, "..", "data", "raw"))
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Inicializar el motor de audio
        self.audio_engine = AudioEngine(
            sample_rate=8000,
            output_folder=self.output_folder
        )
        
        # Configurar callbacks del motor de audio
        self.audio_engine.set_log_callback(self.log)
        self.audio_engine.set_status_callback(self.actualizar_estado_boton)
        
        # Crear interfaz
        self._crear_ui()
    
    def _crear_ui(self):
        """Crear todos los elementos de la interfaz"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # 1. HEADER
        self._crear_header()
        
        # 2. PANEL DE CONTROL
        self._crear_panel_control()
        
        # 3. SECCIÓN DE GRÁFICAS
        self._crear_seccion_graficas()
        
        # 4. LOG / CONSOLA
        self._crear_consola_log()
    
    def _crear_header(self):
        """Crear sección de encabezado"""
        self.header_frame = ctk.CTkFrame(
            self, 
            fg_color="#FFFFFF", 
            height=80, 
            corner_radius=0
        )
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        ctk.CTkLabel(
            self.header_frame, 
            text="Sintetizador Voz-a-Voz (8kHz/8-bit)", 
            font=("Segoe UI", 24, "bold"), 
            text_color="#1D1D1F"
        ).pack(pady=20, padx=30, side="left")
    
    def _crear_panel_control(self):
        """Crear panel de control con entradas y botón principal"""
        self.control_frame = ctk.CTkFrame(
            self, 
            fg_color="#FFFFFF", 
            corner_radius=15
        )
        self.control_frame.grid(row=1, column=0, padx=30, pady=20, sticky="ew")
        self.control_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Entrada de Nombre
        self.frame_name = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.frame_name.grid(row=0, column=0, padx=10, pady=15, sticky="ew")
        
        ctk.CTkLabel(
            self.frame_name, 
            text="Nombre Archivo", 
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")
        
        self.entry_nombre = ctk.CTkEntry(self.frame_name, height=35)
        self.entry_nombre.pack(fill="x", pady=5)
        self.entry_nombre.insert(0, "grabacion_final")
        
        # Entrada de Duración
        self.frame_dur = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.frame_dur.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        
        ctk.CTkLabel(
            self.frame_dur, 
            text="Duración (seg)", 
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")
        
        self.entry_duracion = ctk.CTkEntry(self.frame_dur, height=35)
        self.entry_duracion.pack(fill="x", pady=5)
        self.entry_duracion.insert(0, "3")
        
        # Botón Principal
        self.btn_grabar = ctk.CTkButton(
            self.control_frame, 
            text="GRABAR Y SINTETIZAR", 
            command=self.iniciar_proceso,
            height=45, 
            fg_color="#007AFF", 
            font=("Segoe UI", 14, "bold")
        )
        self.btn_grabar.grid(row=0, column=2, padx=20, pady=15, sticky="ew")
    
    def _crear_seccion_graficas(self):
        """Crear sección de gráficas (forma de onda y espectrograma)"""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=2, column=0, padx=30, pady=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure((0, 1), weight=1)
        
        # Gráfica de forma de onda
        self.fig_wave, self.ax_wave = plt.subplots(figsize=(8, 2), dpi=100)
        self.fig_wave.patch.set_facecolor('#F5F5F7')
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.main_container)
        self.canvas_wave.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=5)
        
        # Gráfica de espectrograma
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(8, 2), dpi=100)
        self.fig_spec.patch.set_facecolor('#F5F5F7')
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.main_container)
        self.canvas_spec.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=5)
    
    def _crear_consola_log(self):
        """Crear consola de log"""
        self.textbox_log = ctk.CTkTextbox(
            self, 
            height=100, 
            fg_color="#FFFFFF", 
            font=("Consolas", 12)
        )
        self.textbox_log.grid(row=3, column=0, padx=30, pady=20, sticky="ew")
        self.log("Listo para procesar voz a 8000Hz y 8-bits.")
    
    def log(self, mensaje):
        """
        Agregar mensaje al log (thread-safe)
        
        Args:
            mensaje: Texto a mostrar en el log
        """
        def _update():
            self.textbox_log.configure(state="normal")
            self.textbox_log.insert("end", f"{mensaje}\n")
            self.textbox_log.see("end")
            self.textbox_log.configure(state="disabled")
        
        # Asegurar que se ejecute en el hilo principal
        self.after(0, _update)
    
    def actualizar_estado_boton(self, status, color=None, text=None):
        """
        Actualizar estado del botón (thread-safe)
        
        Args:
            status: Estado ('preparing', 'recording', 'processing', 'ready')
            color: Color del botón (opcional)
            text: Texto del botón (opcional)
        """
        def _update():
            if color:
                self.btn_grabar.configure(fg_color=color)
            if text:
                self.btn_grabar.configure(text=text)
        
        self.after(0, _update)
    
    def actualizar_graficas(self, signal):
        """
        Actualizar gráficas con nueva señal de audio
        
        Args:
            signal: Señal de audio (numpy array)
        """
        # Gráfica de forma de onda
        self.ax_wave.clear()
        self.ax_wave.plot(signal, color='#007AFF', linewidth=0.5)
        self.ax_wave.set_title(
            "Señal 8-bit / 8kHz (Tiempo)", 
            fontsize=10, 
            fontweight='bold'
        )
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlabel("Muestras")
        self.ax_wave.set_ylabel("Amplitud")
        self.ax_wave.grid(True, alpha=0.3)
        
        # Espectrograma
        self.ax_spec.clear()
        self.ax_spec.set_title(
            "Espectrograma Telefónico (0-4kHz)", 
            fontsize=10, 
            fontweight='bold'
        )
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        librosa.display.specshow(
            S, 
            sr=self.audio_engine.fs, 
            x_axis='time', 
            y_axis='hz', 
            ax=self.ax_spec
        )
        self.ax_spec.set_ylim(0, 4000)  # Límite Nyquist para 8kHz
        
        # Redibujar
        self.canvas_wave.draw()
        self.canvas_spec.draw()
    
    def iniciar_proceso(self):
        """
        Iniciar proceso de grabación y síntesis en hilo separado
        """
        # Validar entrada
        nombre = self.entry_nombre.get().strip()
        if not nombre:
            self.log("⚠ SEÑAL ERROR: Nombre inválido.")
            return
        
        try:
            duracion = float(self.entry_duracion.get() or 3)
        except ValueError:
            self.log("⚠ SEÑAL ERROR: Duración inválida.")
            return
        
        # Deshabilitar botón
        self.btn_grabar.configure(state="disabled", text="PREPARANDO...")
        
        # Ejecutar en hilo separado
        threading.Thread(
            target=self._ejecutar_proceso_audio,
            args=(nombre, duracion),
            daemon=True
        ).start()
    
    def _ejecutar_proceso_audio(self, nombre, duracion):
        """
        Ejecutar proceso completo de audio (corre en hilo separado)
        
        Args:
            nombre: Nombre base para archivos
            duracion: Duración de grabación en segundos
        """
        try:
            # Ejecutar proceso completo en el motor de audio
            resultado = self.audio_engine.proceso_completo(
                nombre_base=nombre,
                duracion=duracion,
                output_folder=self.output_folder
            )
            
            # Si fue exitoso, actualizar gráficas
            if resultado['success'] and resultado.get('audio_procesado') is not None:
                self.after(0, lambda: self.actualizar_graficas(resultado['audio_procesado']))
        
        except Exception as e:
            self.log(f"⚠ ERROR EN PROCESO: {str(e)}")
        
        finally:
            # Restaurar botón al estado original
            self.after(0, lambda: self.btn_grabar.configure(
                state="normal",
                text="GRABAR Y SINTETIZAR",
                fg_color="#007AFF"
            ))


def main():
    """Punto de entrada de la aplicación"""
    app = SintetizadorApp()
    app.mainloop()


if __name__ == "__main__":
    main()