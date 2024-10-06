from flask import Flask, render_template

# Importa altre librerie necessarie da detect.py
from detect import detect  # Assicurati che detect.py contenga la funzione detect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analizza')
def analizza():
    # Chiama la funzione detect con i parametri desiderati
    try:  # view_img True se vuoi visualizzare i risultati in tempo reale
        detect(source='./images/', weights='./YOLOv7x_640.pt', imgsz=640, conf_thres=0.10, save_txt=True, view_img=False)
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    return "Analisi completata!"

if __name__ == '__main__':
    app.run(debug=True)
