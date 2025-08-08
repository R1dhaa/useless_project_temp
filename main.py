# Air2Art: Air Drawing → AI Image
# Hackathon-Optimized, No AR Animation

import cv2
import mediapipe as mp
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import ScribbleDetector
from PIL import Image

# ----------------------------
# 1️⃣ Device Detection
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ----------------------------
# 2️⃣ AI Model Setup
# ----------------------------
print("[INFO] Loading AI models... (first time takes longer)")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

scribble_detector = ScribbleDetector.from_pretrained("lllyasviel/Annotators")

# ----------------------------
# 3️⃣ Mediapipe Hand Tracker
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# 4️⃣ Canvas Setup
# ----------------------------
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
last_x, last_y = None, None

# ----------------------------
# 5️⃣ Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)

ai_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for generated image

prompt = "A realistic painting of a cute bird"  # Default prompt

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])  # Index finger tip
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            if drawing:
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), 5)
                last_x, last_y = x, y
            else:
                last_x, last_y = None, None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge sketch with webcam view
    combined_view = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Combine left (sketch) and right (AI image) into one display
    display = np.hstack((combined_view, ai_image))

    cv2.putText(display, "Press 'd' to draw, 'g' to generate, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Air2Art - Sketch | AI Image", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        drawing = not drawing  # Toggle drawing mode

    elif key == ord('g'):
        print("[INFO] Generating AI image...")
        # Convert canvas to scribble
        pil_image = Image.fromarray(canvas)
        scribble = scribble_detector(pil_image)

        # Run AI model
        result = pipe(prompt, image=scribble).images[0]
        ai_image = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        ai_image = cv2.resize(ai_image, (640, 480))

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
