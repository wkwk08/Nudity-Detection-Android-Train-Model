import cv2, numpy as np, tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="hsv_skin_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Classify one HSV pixel
def classify_pixel(hsv_pixel):
    input_data = np.array([hsv_pixel], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0][0] > 0.5  # Threshold at 0.5

# Apply to full image
def detect_skin_pixels(img):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_norm = hsv / 255.0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for y in range(hsv.shape[0]):
        for x in range(hsv.shape[1]):
            pixel = hsv_norm[y, x]
            if classify_pixel(pixel):
                mask[y, x] = 1

    return mask