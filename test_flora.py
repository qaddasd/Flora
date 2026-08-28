import torch
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYVISTA_OFF_SCREEN"] = "true"

from flora.v3_model import FloraV3

print("🧠 Flora Brain Encoding Model")
print("=" * 50)

# Загрузка модели
checkpoint_path = "checkpoints/best-epoch=052-val/pearson_r=0.7278.ckpt"
if not os.path.exists(checkpoint_path):
    print(f"❌ Чекпоинт не найден: {checkpoint_path}")
    exit(1)

print(f"✅ Загрузка весов: {checkpoint_path}")
model = FloraV3.load_from_checkpoint(checkpoint_path, map_location="cpu")
model.eval()

print(f"✅ Модель загружена успешно!")
print(f"   Параметры: ~14M")
print(f"   Модальности: Текст + Аудио + Видео → Активность мозга")

# Тестовый инференс с фиктивными данными
print("\n🧪 Тестовый запуск...")
batch_size = 1
timesteps = 5

# Создаем тестовые тензоры (в реальности здесь будут реальные фичи из видео)
dummy_text = torch.randn(batch_size, timesteps, 384)  # MiniLM embeddings
dummy_audio = torch.randn(batch_size, timesteps, 384)  # Whisper embeddings  
dummy_video = torch.randn(batch_size, timesteps, 640)  # MobileViT embeddings

print(f"   Входные данные: {dummy_text.shape}, {dummy_audio.shape}, {dummy_video.shape}")

with torch.no_grad():
    output = model(dummy_text, dummy_audio, dummy_video)
    brain_prediction = output["prediction"]

print(f"   Выходные данные: {brain_prediction.shape}")
print(f"   (5 временных шагов × 400 регионов мозга)")

print("\n✅ Flora готова к работе!")
print("\n💡 Для использования с реальными видео:")
print("   1. Положи видео в папку ./videos/")
print("   2. Запусти: python flora/extract_features_v3.py")
print("   3. Обработай извлеченные фичи через модель")
