import customtkinter as ctk
import tkinter as tk
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import os
import threading
import numpy as np

# --- IMPORTACIONES PARA GRAFICAR ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display

# --- CONFIGURACIÓN VISUAL GLOBAL (MODO CLARO) ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Configurar estilo claro para Matplotlib
plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpio y claro

class SintetizadorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la Ventana
        self.title("Voice Synth Studio | Pro")
        self.geometry("900x800")
        self.configure(fg_color="#F5F5F7") # Gris ultra claro de fondo
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.normpath(os.path.join(script_dir, "..", "data", "raw"))
        os.makedirs(self.output_folder, exist_ok=True)

        self.fs = 44100
        self._crear_ui()

    def _crear_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) 

        # 1. HEADER / BARRA SUPERIOR
        self.header_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", height=80, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        self.lbl_titulo = ctk.CTkLabel(self.header_frame, text="Análisis de Señal de Voz", 
                                       font=("Segoe UI", 24, "bold"), text_color="#1D1D1F")
        self.lbl_titulo.pack(pady=20, padx=30, side="left")

        # 2. PANEL DE CONTROL (INPUTS Y BOTÓN)
        self.control_frame = ctk.CTkFrame(self, fg_color="#FFFFFF", corner_radius=15)
        self.control_frame.grid(row=1, column=0, padx=30, pady=20, sticky="ew")
        
        # Grid dentro del panel de control
        self.control_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Input Nombre
        self.frame_name = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.frame_name.grid(row=0, column=0, padx=20, pady=15, sticky="ew")
        ctk.CTkLabel(self.frame_name, text="Nombre del Archivo", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.entry_nombre = ctk.CTkEntry(self.frame_name, placeholder_text="Muestra...", height=35, border_width=1)
        self.entry_nombre.pack(fill="x", pady=5)
        self.entry_nombre.insert(0, "grabacion_01")

        # Input Duración
        self.frame_dur = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.frame_dur.grid(row=0, column=1, padx=20, pady=15, sticky="ew")
        ctk.CTkLabel(self.frame_dur, text="Duración (seg)", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.entry_duracion = ctk.CTkEntry(self.frame_dur, height=35, border_width=1)
        self.entry_duracion.pack(fill="x", pady=5)
        self.entry_duracion.insert(0, "3")

        # Botón Grabar
        self.btn_grabar = ctk.CTkButton(self.control_frame, text="INICIAR CAPTURA", 
                                       command=self.iniciar_grabacion_hilo,
                                       height=45, fg_color="#007AFF", hover_color="#0056b3",
                                       font=("Segoe UI", 14, "bold"))
        self.btn_grabar.grid(row=0, column=2, padx=20, pady=15, sticky="ew")

        # 3. SECCIÓN DE VISUALIZACIÓN (GRÁFICAS)
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=2, column=0, padx=30, pady=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure((0, 1), weight=1)

        self.inicializar_matplotlib_canvases()

        # 4. LOG / CONSOLA (ESTILO MINIMALISTA)
        self.textbox_log = ctk.CTkTextbox(self, height=80, fg_color="#FFFFFF", border_width=1,
                                         font=("Consolas", 12), text_color="#424245")
        self.textbox_log.grid(row=3, column=0, padx=30, pady=20, sticky="ew")
        self.log("Sistema listo para captura de audio.")

    def inicializar_matplotlib_canvases(self):
        # Gráfica de Onda
        self.fig_wave, self.ax_wave = plt.subplots(figsize=(8, 2.5), dpi=100)
        self.fig_wave.patch.set_facecolor('#F5F5F7') 
        self.ax_wave.set_facecolor('#FFFFFF')
        self.ax_wave.set_title("Forma de Onda (Tiempo)", fontsize=10, fontweight='bold', color="#1D1D1F")
        self.fig_wave.tight_layout()

        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.main_container)
        self.canvas_wave.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        # Espectrograma
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(8, 2.5), dpi=100)
        self.fig_spec.patch.set_facecolor('#F5F5F7')
        self.ax_spec.set_facecolor('#FFFFFF')
        self.ax_spec.set_title("Espectrograma (Frecuencia)", fontsize=10, fontweight='bold', color="#1D1D1F")
        self.fig_spec.tight_layout()
        
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.main_container)
        self.canvas_spec.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    def log(self, mensaje):
        timestamp = datetime.now().strftime("%H:%M")
        self.textbox_log.configure(state="normal")
        self.textbox_log.insert("end", f"• [{timestamp}] {mensaje}\n")
        self.textbox_log.see("end")
        self.textbox_log.configure(state="disabled")

    def iniciar_grabacion_hilo(self):
        nombre = self.entry_nombre.get().strip()
        if not nombre:
            self.log("Error: Ingrese un nombre.")
            return
        
        self.btn_grabar.configure(state="disabled", text="ESCUCHANDO...", fg_color="#FF9500") # Naranja mientras graba
        self.ax_wave.clear()
        self.ax_spec.clear()
        self.canvas_wave.draw()
        self.canvas_spec.draw()
        
        t = threading.Thread(target=self.proceso_grabacion_completo, args=(nombre,))
        t.start()

    def proceso_grabacion_completo(self, nombre_base):
        audio_data = None
        try:
            duracion = float(self.entry_duracion.get() or 3)
            ruta_wav = os.path.join(self.output_folder, f"{nombre_base}.wav")
            ruta_txt = os.path.join(self.output_folder, f"{nombre_base}.txt")

            audio_data = sd.rec(int(duracion * self.fs), samplerate=self.fs, channels=1, dtype='float32')
            sd.wait()
            audio_data = audio_data.flatten()

            sf.write(ruta_wav, audio_data, self.fs)
            self.log(f"Archivo '{nombre_base}.wav' guardado.")

            # Transcripción
            r = sr.Recognizer()
            with sr.AudioFile(ruta_wav) as source:
                audio_listo = r.record(source)
                try:
                    texto = r.recognize_google(audio_listo, language="es-MX")
                    with open(ruta_txt, "w", encoding='utf-8') as f: f.write(texto)
                    self.log(f"Texto detectado: \"{texto}\"")
                except:
                    self.log("No se detectó voz clara.")

        except Exception as e:
            self.log(f"Error: {e}")
        
        finally:
            self.after(0, lambda: self.finalizar_proceso_gui(audio_data))

    def finalizar_proceso_gui(self, audio_data):
        self.btn_grabar.configure(state="normal", text="INICIAR CAPTURA", fg_color="#007AFF")
        if audio_data is not None and len(audio_data) > 0:
            self.actualizar_graficas(audio_data)

    def actualizar_graficas(self, audio_signal):
        # Graficar Onda
        self.ax_wave.clear()
        self.ax_wave.set_title("Forma de Onda (Tiempo)", fontsize=10, fontweight='bold')
        times = np.linspace(0, len(audio_signal) / self.fs, num=len(audio_signal))
        self.ax_wave.plot(times, audio_signal, color='#007AFF', linewidth=0.5) # Azul Apple
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.grid(True, alpha=0.3)
        
        # Graficar Espectrograma
        self.ax_spec.clear()
        self.ax_spec.set_title("Espectrograma (Frecuencia)", fontsize=10, fontweight='bold')
        D = librosa.stft(audio_signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=self.fs, x_axis='time', y_axis='hz', 
                                     ax=self.ax_spec, cmap='viridis') # Viridis es muy legible en blanco
        self.ax_spec.set_ylim(0, 8000)

        self.canvas_wave.draw()
        self.canvas_spec.draw()

if __name__ == "__main__":
    app = SintetizadorApp()
    app.mainloop()

    