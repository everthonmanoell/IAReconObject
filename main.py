# Importação das bibliotecas
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, Listbox, Scrollbar


# Definição do conjunto de dados
dataset = tf.keras.datasets.cifar10  # Utilize o dataset de sua preferência (CIFAR-10, COCO, etc.)

# Carregamento e pré-processamento das imagens
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Definição da arquitetura da rede neural convolucional (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(x_train, y_train, epochs=10)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Acurácia:', test_acc)

# Função para classificar uma nova imagem
def classificar_imagem(imagem_caminho):
    # Carregamento e pré-processamento da imagem
    imagem = cv2.imread(imagem_caminho)
    imagem = cv2.resize(imagem, (32, 32))
    imagem = imagem.astype('float32') / 255.0
    imagem = np.expand_dims(imagem, axis=0)

    # Predição da classe
    predicao = model.predict(imagem)
    classe_predita = np.argmax(predicao)

    # Retorno da classe predita
    return classe_predita


# Interface gráfica para carregar e classificar uma imagem

def carregar_imagens():
    global imagem_caminhos
    imagem_caminhos = filedialog.askopenfilenames(title="Selecionar imagens")
    for imagem_caminho in imagem_caminhos:
        lista_imagens.insert("end", imagem_caminho)


def classificar():
    for imagem_caminho in imagem_caminhos:
        classe_predita = classificar_imagem(imagem_caminho)
        lista_classes.insert("end", classe_predita)


# Criação da interface gráfica
app = Tk()
app.title("Classificador de Imagens")

lista_imagens = Listbox(app, width=50, height=10)
lista_imagens.pack(side="left", fill="both", expand=True)

scrollbar_imagens = Scrollbar(app, orient="vertical", command=lista_imagens.yview)
scrollbar_imagens.pack(side="right", fill="y")

lista_imagens.config(yscrollcommand=scrollbar_imagens.set)

carregar_button = Button(app, text="Carregar imagens", command=carregar_imagens)
carregar_button.pack()

classificar_button = Button(app, text="Classificar", command=classificar)
classificar_button.pack()

lista_classes = Listbox(app, width=20, height=10)
lista_classes.pack()

app.mainloop()
