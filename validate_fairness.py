from data_loader import load_fitzpatrick_data
import tensorflow as tf
import numpy as np
from collections import defaultdict

interpreter = tf.lite.Interpreter(model_path="skin_detector.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

images, tones = load_fitzpatrick_data()
results = defaultdict(list)

for img, tone in zip(images, tones):
    input_data = np.expand_dims(img.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    skin_pixels = np.sum(output_data > 0.5)
    results[tone].append(skin_pixels)

for tone in sorted(results.keys()):
    avg = np.mean(results[tone])
    print(f"Tone {tone}: Avg detected skin pixels = {avg:.2f}")
