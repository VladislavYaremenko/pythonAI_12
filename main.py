import torch
from torchvision import models
from torchvision.models import ResNet50_Weights, AlexNet_Weights
from PIL import Image
from pathlib import Path


# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ПОРIВНЯННЯ RESNET50 I ALEXNET")
print("=" * 60)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"Вiдеокарта: {torch.cuda.get_device_name(0)}")


# ==========================================
# 2. ЗАВАНТАЖЕННЯ RESNET50
# ==========================================

print("\nЗавантаження ResNet50")

resnet_weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=resnet_weights)

resnet.eval()
resnet.to(device)

resnet_transform = resnet_weights.transforms()
resnet_categories = resnet_weights.meta["categories"]


# ==========================================
# 3. ЗАВАНТАЖЕННЯ ALEXNET
# ==========================================

print("Завантаження AlexNet")

alexnet_weights = AlexNet_Weights.DEFAULT
alexnet = models.alexnet(weights=alexnet_weights)

alexnet.eval()
alexnet.to(device)

alexnet_transform = alexnet_weights.transforms()
alexnet_categories = alexnet_weights.meta["categories"]


# ==========================================
# 4. ПЕРЕВIРКА ЗОБРАЖЕННЯ
# ==========================================

def predict(image, model, transform, categories):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities, 5, dim=1
        )

    results = []

    for probability, index in zip(
        top_probabilities[0], top_indices[0]
    ):
        category = categories[index.item()]
        confidence = probability.item() * 100

        results.append((category, confidence))

    return results


# ==========================================
# 5. ОБРОБКА ФОТО
# ==========================================

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    file for file in IMAGES_DIR.iterdir()
    if file.is_file() and file.suffix.lower() in image_extensions
]

if not images:
    print("\nВ images нема фото!")
    print("Додай зображення")
    raise SystemExit

report = []

report.append("ПОРIВНЯННЯ RESNET50 И ALEXNET")
report.append("=" * 60)
report.append(f"Кiлькiсть зображеннь: {len(images)}")
report.append(f"Device: {device}")
report.append("")

for image_path in images:
    print(f"\nОбробка: {image_path.name}")

    try:
        image = Image.open(image_path).convert("RGB")

        # ResNet50
        resnet_results = predict(
            image,
            resnet,
            resnet_transform,
            resnet_categories
        )

        # AlexNet
        alexnet_results = predict(
            image,
            alexnet,
            alexnet_transform,
            alexnet_categories
        )

        report.append("=" * 60)
        report.append(f"ЗОБРАЖЕННЯ: {image_path.name}")
        report.append("=" * 60)

        report.append("\nRESNET50 - ТОП-5:")

        for i, (category, confidence) in enumerate(
            resnet_results, start=1
        ):
            line = f"{i}. {category} — {confidence:.2f}%"
            report.append(line)
            print("ResNet50:", line)

        report.append("\nALEXNET - ТОП-5:")

        for i, (category, confidence) in enumerate(
            alexnet_results, start=1
        ):
            line = f"{i}. {category} — {confidence:.2f}%"
            report.append(line)
            print("AlexNet:", line)

        report.append("")

    except Exception as error:
        message = f"помилка при обробцi {image_path.name}: {error}"
        print(message)
        report.append(message)


# ==========================================
# 6. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТУ
# ==========================================

results_file = RESULTS_DIR / "results.txt"

with open(results_file, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\n" + "=" * 60)
print("ОБРОБКА ЗАВЕРШЕНА")
print(f"Результати збереженi: {results_file}")
print("=" * 60)
