import gradio as gr
import subprocess

def iniciar_detector():
    try:
        subprocess.Popen(["python", "app_estres.py"])
        return "Sistema iniciado. Revise la ventana de la cámara."
    except Exception as e:
        return f"Error al iniciar: {e}"

demo = gr.Interface(
    fn=iniciar_detector,
    inputs=None,
    outputs="text",
    title="Sistema de detección de Estrés y Fatiga",
    description="Aplicación de análisis facial para detectar indicadores de cansancio, estrés y somnolencia mediante visión por computadora."
)

demo.launch()
