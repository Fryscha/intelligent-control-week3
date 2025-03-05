import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model yang telah dilatih
model = load_model('cnn_model.h5')

# Load label kelas
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Ganti dengan label kelas yang sesuai

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas
    pred = model.predict(img)
    class_id = np.argmax(pred)
    conf = pred[0][class_id] * 100  # Mengubah ke persentase

    # Tampilkan hasil
    label = f'Class: {class_labels[class_id]}: {conf:.2f}%'
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
