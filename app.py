import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import yaml
import os

# Стиль Стримлита
st.set_page_config(
    page_title="Подводный Радар",
    page_icon="🌊",
    layout="wide"
)

# Загрузка конфига
with open('sonar-seg.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Путь к модели
model_path = "runs/segment/train/weights/best.pt"
if not os.path.exists(model_path):
    st.error(f"Модель {model_path} не найдена. Убедитесь, что файл существует.")
else:
    # Загрузка модели
    model = YOLO(model_path)

# Функция для предсказания
def predict(image):
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Преобразование в RGB, если изображение черно-белое
    results = model.predict(image)
    return results[0].plot()

# Функция для добавления фонового изображения(????------)
def add_background(image_path):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{image_path}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .stImage {{
        max-width: 50%;
        margin: auto;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def add_icons():
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)

# Добавление фонового изображения(?????-----)
add_background("/Users/imac/Desktop/Diplom/projects/new_data2/interface")


add_icons()

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
        # Получение результата предсказания
        result_image = predict(image_np)
        st.image(result_image, caption="Результат предсказания", width=400)

