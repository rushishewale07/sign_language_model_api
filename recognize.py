import cv2
import base64
import os
import time
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="8cNkNcwVfGQMHWyLhFyo"
)

# Roboflow model ID
MODEL_ID = "asl-alphabet-dataset/1"

# Temp image path
temp_dir = r'C:\Users\DELL\OneDrive\Desktop\temp_images'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 👇 Folder where alphabet images will be saved
save_base_dir = r'C:\Users\DELL\OneDrive\Desktop\sign_language[1]\model'
if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)

# Webcam start
cap = cv2.VideoCapture(0)

# Track last saved time per sign
last_saved = {}

# Cooldown time (in seconds) before same sign can be saved again
save_cooldown = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture webcam frame.")
        break

    # Save temp frame
    temp_image_path = os.path.join(temp_dir, "temp.jpg")
    cv2.imwrite(temp_image_path, frame)

    # Encode to base64
    with open(temp_image_path, "rb") as img_file:
        img_bytes = img_file.read()
        encoded_image = base64.b64encode(img_bytes).decode("utf-8")

    # Send to Roboflow
    result = CLIENT.infer(encoded_image, model_id=MODEL_ID)

    # If we got predictions
    if result['predictions']:
        # Get most confident prediction
        top_pred = sorted(result['predictions'], key=lambda x: x['confidence'], reverse=True)[0]

        class_name = top_pred['class']
        confidence = top_pred['confidence']

        # Only proceed if confident enough
        if confidence > 0.5:
            x, y = int(top_pred['x']), int(top_pred['y'])
            width, height = int(top_pred['width']), int(top_pred['height'])

            x1 = max(0, x - width // 2)
            y1 = max(0, y - height // 2)
            x2 = min(frame.shape[1], x + width // 2)
            y2 = min(frame.shape[0], y + height // 2)

            # Draw green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Show label
            label = f"{class_name} ({confidence * 100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            # Save image if high confidence
            if confidence > 0.85:
                current_time = time.time()
                last_time = last_saved.get(class_name, 0)

                if current_time - last_time > save_cooldown:
                    # Create folder for the letter
                    class_folder = os.path.join(save_base_dir, class_name)
                    if not os.path.exists(class_folder):
                        os.makedirs(class_folder)

                    # Create filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{class_name}_{timestamp}.jpg"
                    image_path = os.path.join(class_folder, image_filename)

                    # Save full frame image
                    cv2.imwrite(image_path, frame)

                    print(f"[📸] Saved: {image_path}")
                    last_saved[class_name] = current_time
    else:
        # Show no hand detected message
        cv2.putText(frame, "No hand detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # Show window
    cv2.imshow("ASL Hand Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
