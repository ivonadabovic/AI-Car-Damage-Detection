# test_onnx_model.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("🔍 TESTIRANJE ONNX MODELA")
print("=" * 60)

# 1. Provjeri da li model postoji
if not os.path.exists("best.onnx"):
    print("❌ best.onnx NIJE PRONAĐEN!")
    exit(1)

print(f"✅ best.onnx pronađen ({os.path.getsize('best.onnx') / (1024*1024):.2f} MB)")

# 2. Učitaj model
try:
    session = ort.InferenceSession("best.onnx")
    print("✅ Model uspješno učitan!")
    
    print("\n📋 Ulazni node-ovi:")
    for inp in session.get_inputs():
        print(f"   {inp.name}: {inp.shape} ({inp.type})")
    
    print("\n📋 Izlazni node-ovi:")
    for out in session.get_outputs():
        print(f"   {out.name}: {out.shape} ({out.type})")
    
except Exception as e:
    print(f"❌ Greška pri učitavanju: {e}")
    exit(1)

# 3. Testiraj na slici
print("\n" + "=" * 60)
print("🧪 TESTIRANJE NA SLICI")
print("=" * 60)

# Uzmi sliku iz foldera
image_files = [f for f in os.listdir(".") if f.endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("❌ Nema slika u folderu!")
    print("📌 Uploadujte neku sliku ili koristite car.jpg")
    exit(1)

image_path = image_files[0]
print(f"📸 Slika: {image_path}")

try:
    # Učitaj sliku
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 640))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_chw = np.transpose(img_array, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    print(f"📊 Ulazni tensor: {img_batch.shape}")
    
    # Pokreni model
    inputs = {session.get_inputs()[0].name: img_batch}
    outputs = session.run(None, inputs)
    
    print(f"✅ Model pokrenut!")
    
    # Analiziraj rezultate
    output_data = outputs[0]
    print(f"\n📊 Izlazni tensor: {output_data.shape}")
    print(f"   Min: {output_data.min():.4f}")
    print(f"   Max: {output_data.max():.4f}")
    print(f"   Mean: {output_data.mean():.4f}")
    
    # Pronađi najbolju detekciju
    num_classes = 6
    num_detections = output_data.shape[2]
    
    print(f"\n📊 Broj detekcija: {num_detections}")
    
    class_names = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
    class_names_sr = ["Udubljenje", "Ogrebotina", "Pukotina", "Razbijeno staklo", "Oštećen far", "Probušena guma"]
    
    detections = []
    threshold = 0.3
    
    for i in range(num_detections):
        x1 = output_data[0, 0, i]
        y1 = output_data[0, 1, i]
        x2 = output_data[0, 2, i]
        y2 = output_data[0, 3, i]
        
        for c in range(num_classes):
            conf = output_data[0, 4 + c, i]
            if conf > threshold:
                detections.append({
                    'class': c,
                    'class_name': class_names[c],
                    'class_name_sr': class_names_sr[c],
                    'confidence': conf,
                    'box': (x1, y1, x2, y2)
                })
    
    # Sortiraj po confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n🎯 Pronađeno {len(detections)} detekcija:")
    for i, det in enumerate(detections[:10]):
        print(f"   {i+1}. {det['class_name_sr']} ({det['class_name']}) - {det['confidence']*100:.1f}%")
        print(f"      Box: ({det['box'][0]:.2f}, {det['box'][1]:.2f}) -> ({det['box'][2]:.2f}, {det['box'][3]:.2f})")
    
    if len(detections) == 0:
        print("   ❌ Nema detekcija!")

except Exception as e:
    print(f"❌ Greška pri testiranju: {e}")

print("\n" + "=" * 60)
print("✅ TESTIRANJE ZAVRŠENO!")