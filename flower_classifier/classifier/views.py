from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from torchvision import transforms as T
from PIL import Image
import os
import onnxruntime as ort
import numpy as np

# إعداد التحويلات
mean, std, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
test_transformations = T.Compose([
    T.Resize((size, size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# تحميل نموذج ONNX
onnx_model_path = "classifier/model.onnx"
try:
    ort_session = ort.InferenceSession(onnx_model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the ONNX model: {e}")

# دالة لتحميل الصورة والتنبؤ
def predict_image(image_path):
    try:
        # تحميل الصورة وتحويلها
        img = Image.open(image_path).convert("RGB")
        img = test_transformations(img).unsqueeze(0).numpy()

        # تشغيل التنبؤ باستخدام ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)

        # استخراج الفئة المتوقعة
        predicted_label = np.argmax(ort_outs[0], axis=1)[0]
        return predicted_label
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

# دالة رفع الصورة
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)

            # التنبؤ بالصورة
            image_path = os.path.join("media", filename)
            prediction = predict_image(image_path)

            # أسماء ومعلومات الفئات
            flower_data = [
                {"ar": "زهرة الأقحوان", "en": "Daisy", "info_ar": "زهرة بيضاء جميلة ترمز إلى النقاء.", "info_en": "A beautiful white flower symbolizing purity."},
                {"ar": "زهرة الهندباء", "en": "Dandelion", "info_ar": "زهرة برية صفراء معروفة بقدرتها على الانتشار.", "info_en": "A wild yellow flower known for its ability to spread."},
                {"ar": "زهرة الورد", "en": "Rose", "info_ar": "رمز الحب والجمال.", "info_en": "A symbol of love and beauty."},
                {"ar": "زهرة عباد الشمس", "en": "Sunflower", "info_ar": "زهرة تتبع الشمس طوال اليوم.", "info_en": "A flower that tracks the sun throughout the day."},
                {"ar": "زهرة التوليب", "en": "Tulip", "info_ar": "زهرة معروفة بجمالها ورائحتها العطرة.", "info_en": "A beautiful flower known for its fragrance."},
            ]

            predicted_class = flower_data[prediction]
            return render(request, 'result.html', {
                'predicted_class': f"{predicted_class['ar']} ({predicted_class['en']})",
                'flower_info_ar': predicted_class['info_ar'],
                'flower_info_en': predicted_class['info_en'],
                'image_url': uploaded_file_url,
            })
        except Exception as e:
            return render(request, 'index.html', {'error': f"An error occurred: {e}"})

    return render(request, 'index.html')
