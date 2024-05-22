import numpy as np
import joblib
import gradio as gr

def load_css(file_name):
    with open(file_name) as f:
        return f"<style>{f.read()}</style>"

# Se reciben los valores y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    preds = model.predict([x_in])
    return preds[0]

# Definir la función de predicción para Gradio
def predict(mes, dia, hora, aparcamiento, precipitaciones):
    # Mapeo de opciones visibles a valores reales
    mes_mapping = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
    hora_mapping = {'00:00':0,'01:00':1,'02:00':2,'03:00':3,'04:00':4,'05:00':5,'06:00':6,'07:00':7,'08:00':8,
                    '09:00':9,'10:00':10,'11:00':11,'12:00':12,'13:00':13,'14:00':14,'15:00':15,'16:00':16,'17:00':17,
                    '18:00':18,'19:00':19,'20:00':20,'21:00':21,'22:00':22,'23:00':23}
    aparcamiento_mapping = {'RINCÓN': 1, 'ELDER': 2, 'SAN BERNARDO': 3, 'SANAPÚ': 4, 'MATA': 5, 'VEGUETA': 6, 'METROPOL': 7}
    precipitaciones_mapping = {'Sin lluvia':0, 'lluvioso':1}
    laboral_festivo_mapping = {'Festivo':0, 'Laboral':1}

    # Definir el path del modelo basado en la selección del usuario
    # MODEL_PATH = f'./modelos_aparcamientos_entrenados/xgb_trained_model_{aparcamiento.lower()}.pkl'
    MODEL_PATH = f'./modelos_aparcamientos_entrenados/rf_trained_model_{aparcamiento.lower()}.pkl'


    # rf_trained_model_MATA_pickle.pkl

    # Cargar el modelo
    model = joblib.load(MODEL_PATH)

    # Determinar si es festivo o laboral según el mes y el día seleccionados
    def es_festivo(mes, dia):
        # Festivos y fines de semana de 2023
        festivos = [
            (1, 1),(1, 6),  # Enero
            (1, 7),(1, 8),(1, 14),(1, 15),
            (1, 21),(1, 22),(1, 28),(1, 29),
            
            (2, 2),         # Febrero
            (2, 4),(2, 5),(2, 11),(2, 12),
            (2, 18),(2, 19),(2, 25),(2, 26),
            
            (3, 4),         # Marzo
            (3, 5),(3, 11),(3, 12),(3, 17),
            (3, 18),(3, 19),(3, 25),(3, 26),
            
            (4, 1),         # Abril
            (4, 2),(4, 8),(4, 9),(4, 15),(4, 16),
            (4, 22),(4, 23),(4, 29),(4, 30),
            
            (5, 1),         # Mayo
            (5, 6),(5, 7),(5, 13),(5, 14),(5, 20),
            (5, 21),(5, 27),(5, 28),(5, 30),
            
            (6, 3),         # Junio
            (6, 4),(6, 10),(6, 11),(6, 17),(6, 18),
            (6, 24),(6, 25),
            
            (7, 1),         # Julio
            (7, 2),(7, 8),(7, 9),(7, 15),(7, 16),
            (7, 22),(7, 23),(7, 29),(7, 30),
            
            (8, 5),         # Agosto
            (8, 6),(8, 12),(8, 13),(8, 15),(8, 19),
            (8, 20),(8, 26),(8, 27),
            
            (9, 2),         # Septiembre
            (9, 3),(9, 9),(9, 10),(9, 16),(9, 17),
            (9, 23),(9, 24),(9, 30),
            
            (10, 1),        # Octubre
            (10, 7),(10, 8),(10, 12),(10, 14),(10, 15),
            (10, 21),(10, 22),(10, 28),(10, 29),
            
            (11, 1),        # Noviembre
            (11, 4),(11, 5),(11, 11),(11, 12),(11, 18),
            (11, 19),(11, 25),(11, 26),
            
            (12, 2),        # Diciembre
            (12, 3),(12, 6),(12, 8),(12, 9),(12, 10),
            (12, 16),(12, 17),(12, 23),(12, 24), (12, 25),
            (12, 30),(12, 31)
        ]
        return (mes, dia) in festivos

    if es_festivo(mes_mapping[mes], int(dia)):
        laboral_festivo = 'Festivo'
    else:
        laboral_festivo = 'Laboral'

    x_in = [int(mes_mapping[mes]), int(dia), int(hora_mapping[hora]), int(precipitaciones_mapping[precipitaciones]), int(laboral_festivo_mapping[laboral_festivo])]
    predictS = round(int(model_prediction(x_in, model)))
    # predictS = max(round(int(model_prediction(x_in, model))), 0)
    # predictS = abs(round(int(model_prediction(x_in, model))))
    # Asegurarse de que predictS no sea negativo
    # if predictS < 0:
    #     predictS = 0

    resultado = f'La predicción de ocupación en {aparcamiento}, para el día {dia} de {mes} a las {hora} horas, es de: {predictS} vehículos'
    detalle = f"A su vez hay que tener en cuenta que es un día {laboral_festivo} y {precipitaciones}."
    detalle_2 = f"\n El modelo usado es -> {MODEL_PATH}"

    return resultado, detalle, detalle_2

# Configuración de la interfaz de Gradio
mes_options = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
dia_options = [str(d) for d in range(1, 32)]
hora_options = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
aparcamiento_options = ['RINCÓN', 'ELDER', 'SAN BERNARDO', 'SANAPÚ', 'MATA', 'VEGUETA', 'METROPOL']
precipitaciones_options = ['Sin lluvia', 'lluvioso']

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(mes_options, label="Elija el mes que quiere ver:"),
        gr.Dropdown(dia_options, label="Elija el día que quiere ver:"),
        gr.Dropdown(hora_options, label="Elija la hora:"),
        gr.Dropdown(aparcamiento_options, label="Elija el aparcamiento:"),
        gr.Dropdown(precipitaciones_options, label="Elija la estimación climatológica:")
    ],
    outputs=[
        gr.Textbox(label="Predicción de ocupación"),
        gr.Textbox(label="Detalles adicionales")
    ],
    title="Predicción de ocupación de aparcamiento",
    description="Ingrese los detalles para obtener una predicción de ocupación de aparcamiento."
)

# if __name__ == "__main__":
#     iface.launch()

if __name__ == "__main__":
    iface.launch(share=True)

