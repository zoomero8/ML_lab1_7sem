import gradio as gr
import onnxruntime as rt
from torchvision import transforms
from PIL import Image
import torch
from pathlib import Path

# --- Конфигурация ---
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.onnx"
CLASS_NAMES = ['balls', 'cars', 'dogs']

# --- Загрузка модели ---
# Создаем сессию для инференса ONNX модели
sess = rt.InferenceSession(str(MODEL_PATH))
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# --- Трансформации для изображения ---
# Важно: эти трансформации должны быть АНАЛОГИЧНЫ валидационным из этапа обучения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image: Image.Image):
    """Принимает PIL Image, возвращает словарь с вероятностями классов."""
    # Предобработка изображения
    img_tensor = transform(image).unsqueeze(0)

    # Запуск модели
    outputs = sess.run([output_name], {input_name: img_tensor.numpy()})[0]

    # Постобработка для получения вероятностей
    probabilities = torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1)[0]

    # Формирование результата
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences


# --- Создание интерфейса Gradio ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение"),
    outputs=gr.Label(num_top_classes=3, label="Результат"),
    title="Классификатор: Мячи, Машины, Собаки",
    description="Загрузите изображение, чтобы определить, к какому классу оно относится. Модель на основе ResNet18.",
)

# --- Запуск приложения ---
if __name__ == "__main__":
    iface.launch()