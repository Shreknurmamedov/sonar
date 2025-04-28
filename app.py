import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import yaml
import os
import base64

# Функция для конвертации локального изображения в Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Функция для добавления фонового изображения
def add_background(image_path):
    image_base64 = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    h1, h2, p, label {{
        color: white;
    }}
    .stButton>button {{
        background-color: #007BFF; /* Голубая кнопка */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #0056b3; /* Темнее при наведении */
    }}
    .stFileUploader>div>div {{
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px dashed white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }}
    /* Стили для бокового меню */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.5); /* Прозрачный фон */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); /* Тень */
    }}
    [data-testid="stSidebar"] * {{
        color: white !important; /* Белый цвет текста */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Укажите путь к вашему локальному изображению
add_background("/Users/imac/Desktop/Diplom/projects/new_data2/assets/background.jpg")

# Загрузка конфигурации
with open('sonar-seg.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Проверка существования модели
model_path = "runs/segment/train/weights/best.pt"
if not os.path.exists(model_path):
    st.error(f"Модель {model_path} не найдена. Убедитесь, что файл существует.")
else:
    model = YOLO(model_path)

# Функция для предсказания
def predict(image):
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Преобразование в RGB, если изображение черно-белое
    results = model.predict(image)
    return results[0].plot()

# Главная страница
st.title("Подводный Радар 🌊")
st.markdown("<h2 style='color: white;'>Распознавание препятствий и дна на эхолотных снимках</h2>", unsafe_allow_html=True)

# Информация о классах
st.sidebar.header("Классы")
class_names = config['names']
for class_id, class_name in class_names.items():
    st.sidebar.markdown(f"<p style='color: white;'>{class_id}: {class_name}</p>", unsafe_allow_html=True)

# Загрузка изображения
uploaded_image = st.file_uploader("Загрузите эхолотный снимок", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Открытие изображения
    image = Image.open(uploaded_image)
    st.image(image, caption="Загруженное изображение", width=400)

    # Преобразование изображения в NumPy массив
    image_np = np.array(image)

    # Кнопка для выполнения предсказания
    if st.button("Сделать предсказание"):
        with st.spinner("Обработка изображения..."):
            result_image = predict(image_np)
            st.image(result_image, caption="Результат предсказания", width=400)
