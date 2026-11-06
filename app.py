import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import urllib.request
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- СКАЧИВАНИЕ МОДЕЛЕЙ С GOOGLE ДИСКА ---
def download_models():
    # ВСТАВЬТЕ СЮДА ВАШИ ID ИЗ ССЫЛОК (без кавычек)
    # Пример: если ссылка https://drive.google.com/file/d/1AbCdEfGh/view...
    # То ID = 1AbCdEfGh
    urls = {
         "pneumonia_model_covid_final.pth": "18ZvC2AF1s5pyW4yjN0iYd59Xtxhf_Gy",
        "bones_model_final.pth": "1pSWEJVCO3icbW49E0KZpclVSk8_7aJ5",
        "mri_model_perfect.pth": "1nr_f3KWkZ0Z3YfPF1kkl_owSvwN19wI"
    }
    
    for filename, file_id in urls.items():
        if not os.path.exists(filename):
            print(f"Скачиваем {filename}...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(url, filename)
            print(f"{filename} скачан!")

download_models()
# ------------------------------------------

st.set_page_config(
    page_title="MedAI Pro | Диагностика",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    h1 { font-size: 3rem !important; color: #0B3D91; font-weight: 700; margin-bottom: 20px;}
    h2 { font-size: 2.2rem !important; color: #0B3D91; font-weight: 600; }
    h3 { font-size: 1.6rem !important; color: #0B3D91; font-weight: 600; margin-top: 20px;}
    [data-testid="stSidebar"] { background-color: #1e293b; color: white; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: white !important; }
    .stButton>button {
        background-color: #0B3D91; color: white; border-radius: 8px; font-weight: 600; font-size: 1.1rem;
        padding: 0.7rem 1.5rem; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease; width: 100%; margin-top: 10px;
    }
    .stButton>button:hover { background-color: #1565C0; box-shadow: 0 6px 8px rgba(0,0,0,0.2); color: white; }
    .marker-normal { color: #2E7D32; font-weight: bold; font-size: 1.3em; }
    .marker-pathology { color: #C62828; font-weight: bold; font-size: 1.3em; }
    img { border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    </style>
""", unsafe_allow_html=True)

# Пути к моделям теперь просто названия файлов (они скачаются в папку с сайтом)
PATH_PNEUMONIA = "pneumonia_model_covid_final.pth"
PATH_MRI = "mri_model_perfect.pth"
PATH_BONES = "bones_model_final.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_type):
    if model_type == "Рентген легких":
        if not os.path.exists(PATH_PNEUMONIA): return None, None, None
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3) 
        model.load_state_dict(torch.load(PATH_PNEUMONIA, map_location=torch.device('cpu')))
        classes = ['COVID-19', 'Норма', 'Пневмония']
        status = ['Патология', 'Норма', 'Патология']
    elif model_type == "Рентген костей":
        if not os.path.exists(PATH_BONES): return None, None, None
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(PATH_BONES, map_location=torch.device('cpu')))
        classes = ['Перелом', 'Норма']
        status = ['Патология', 'Норма']
    elif model_type == "МРТ головного мозга":
        if not os.path.exists(PATH_MRI): return None, None, None
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load(PATH_MRI, map_location=torch.device('cpu')))
        classes = ['Глиома', 'Менингиома', 'Опухоли нет', 'Опухоль гипофиза']
        status = ['Патология', 'Патология', 'Норма', 'Патология']
    
    model.eval()
    return model, classes, status

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>MedAI Pro</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9em;'>Система поддержки врачебных решений</p>", unsafe_allow_html=True)
    st.markdown("---")
    modality = st.radio("**Выберите тип исследования:**", ("Рентген легких", "Рентген костей", "МРТ головного мозга"))
    st.markdown("---")
    st.markdown("<div style='height: 35vh;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8em; color: #94a3b8; text-align: center;'>Версия 1.0<br>Разработано для ВКР<br><b>Чебаковой Ольгой Илдаровной</b></p>", unsafe_allow_html=True)

st.markdown(f"<h1>{modality}</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.3rem; color: #555; margin-bottom: 30px;'>Загрузите медицинское изображение для автоматического анализа нейросетью.</p>", unsafe_allow_html=True)

model, classes, status = load_model(modality)

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Выберите файл для загрузки...", type=["jpg", "jpeg", "png"], key=f"uploader_{modality}")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        st.markdown("### Исходный снимок")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("### Результаты анализа")
        if st.button("Запустить анализ"):
            with st.spinner('Нейросеть обрабатывает снимок...'):
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_class = torch.argmax(probabilities).item()
                result_name = classes[predicted_class]
                result_status = status[predicted_class]
                confidence = probabilities[predicted_class].item() * 100

                if result_status == 'Норма':
                    bg_color = "#E8F5E9"; border_color = "#2E7D32"; text_color = "#1B5E20"; title_text = "Патологий не выявлено"
                else:
                    bg_color = "#FFEBEE"; border_color = "#C62828"; text_color = "#B71C1C"; title_text = "Обнаружена патология"

                st.markdown(f"""
                    <div style="background-color: {bg_color}; border-left: 8px solid {border_color}; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: {text_color}; margin: 0; padding-bottom: 5px;">Заключение: {title_text}</h4>
                        <p style="color: {text_color}; margin: 0; font-size: 1.5rem; font-weight: bold;">{result_name}</p>
                        <p style="color: {text_color}; margin: 5px 0 0 0; font-size: 1.1rem;">Уверенность модели: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Вероятности по классам:")
                prob_df = pd.DataFrame({
                    "Класс": classes,
                    "Вероятность": [p.item() * 100 for p in probabilities],
                    "Статус": status
                }).sort_values(by="Вероятность", ascending=False)
                
                for index, row in prob_df.iterrows():
                    is_class_normal = row['Статус'] == 'Норма'
                    marker = '<span class="marker-normal">●</span>' if is_class_normal else '<span class="marker-pathology">●</span>'
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #ddd;">
                            <div style="font-size: 1.2rem;">{marker} &nbsp; <b>{row['Класс']}</b></div>
                            <div style="font-size: 1.2rem; color: #333;">{row['Вероятность']:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div style="background-color: #FFF9C4; border: 2px solid #FBC02D; border-radius: 8px; padding: 15px; margin-top: 25px;">
                        <p style="color: #F57F17; font-size: 1.1rem; font-weight: bold; margin: 0;">⚠️ Внимание: Заключение предварительное. Требуется консультация врача.</p>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.info("Пожалуйста, загрузите снимок для анализа.")