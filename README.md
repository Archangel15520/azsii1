# Васильев Г.М. ББМО-02-23

# Практика 1: Установка окружения и настройка фреймворков для анализа защищенности ИИ

CleverHans — это ещё одна популярная библиотека для работы с враждебными атаками, разработанная для проведения тестирования и защиты моделей от атакующих примеров. CleverHans был одной из первых библиотек, ориентированных на adversarial attacks, и изначально был создан командой Google Brain.

Foolbox — это библиотека для проведения атак на модели машинного обучения с целью тестирования их устойчивости к adversarial attacks (враждебным атакам). Она предоставляет удобные методы для создания атакующих примеров, которые могут вызывать некорректное поведение модели.
  
  ---

**WARNING(Важная информация): 1. Все работы по данному предмету можно найти по ссылке: https://github.com/Archangel15520/AZSII-REPO/tree/main**

**2. В коде используется ранее обученная модель на датасете MNIST, которую можно найти в закрепе к данному проекту.**

**3. Сылка на выполненую работу в среде google colab: https://colab.research.google.com/drive/1hhY8c2kcZOTbC5AR2JwprqDSMS7mMgTt#scrollTo=-fwC81hqUcHp** 

  ---


# Установка необходимых библиотек:

```
!pip install foolbox
!pip install cleverhans
```

# Проверка установленных библиотек:

```
import tensorflow as tf
import torch
import foolbox
import cleverhans

print(f"TensorFlow version: {tf.__version__}") # Проверка версии TensorFlow
print(f"PyTorch version: {torch.__version__}") # Проверка версии PyTorch
print(f"Foolbox version: {foolbox.__version__}") # Проверка версии Foolbox
print(f"CleverHans version: {cleverhans.__version__}") # Проверка версии CleverHans

```
![image](https://github.com/Archangel15520/azsii1/blob/main/screemshot/1.JPG)

# Загрузка и обучение простой модели на датасете MNIST:

```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # Загрузка датасета MNIST
# Нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0
# Преобразование меток в one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Создание модели
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Компиляция модели

model.fit(train_images, train_labels, epochs=5) # Обучение модели

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

```

![image](https://github.com/Archangel15520/azsii1/blob/main/screemshot/2.JPG)

# Сохранение модели

```

#Сохранение модели в рабочую директорию google colab
model.save('mnist_model.h5')
#скачивание файла на локальный компьютер сразу
from google.colab import files
files.download('mnist_model.h5')

```

![image](https://github.com/Archangel15520/azsii1/blob/main/screemshot/3.JPG)
