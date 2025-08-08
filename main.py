import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import clip  # pip install ftfy regex tqdm transformers torch torchvision

# --------- Device ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --------- Load CLIP model ---------
print("[INFO] Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --------- Candidate categories for CLIP zero-shot ---------
CATEGORIES = [
    "cat", "car", "house", "airplane", "flower", "dog", "tree", "bicycle", "person",
    "bird", "boat", "chair", "clock", "dog", "elephant", "guitar", "hat", "keyboard"
]

def classify_sketch_with_clip(img_np):
    pil_img = Image.fromarray(img_np).convert("RGB")
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)

    text_inputs = clip.tokenize(CATEGORIES).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        best_idx = similarity.argmax().item()
        return CATEGORIES[best_idx]

# --------- Load Stable Diffusion + ControlNet ---------
print("[INFO] Loading Stable Diffusion + ControlNet models...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

# --------- Mediapipe hand tracking ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --------- Canvas & states ---------
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False
last_pos = None
ai_image = np.zeros((480, 640, 3), dtype=np.uint8)

# --------- Webcam loop ---------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            if drawing and last_pos is not None:
                cv2.line(canvas, last_pos, (x, y), (255, 255, 255), 5)
            last_pos = (x, y)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        last_pos = None

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    display = np.hstack((combined, ai_image))

    cv2.putText(display, "d: toggle draw | g: generate | c: clear | q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Air2Art", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        drawing = not drawing

    elif key == ord('c'):
        canvas.fill(0)
        ai_image.fill(0)

    elif key == ord('g'):
        print("[INFO] Classifying sketch with CLIP...")
        detected_label = classify_sketch_with_clip(canvas)
        print(f"[INFO] Detected: {detected_label}")

        prompt = f"A stunning, lifelike digital painting of a {detected_label}, ultra-realistic details, sharp focus, natural shadows, and vibrant colors"
        pil_image = Image.fromarray(canvas)
        print(f"[INFO] Generating AI image with prompt: {prompt}")

        # Resize input image to 384x384 (good balance quality & speed)
        small_pil = pil_image.resize((384, 384))

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = pipe(
                    prompt,
                    image=small_pil,
                    num_inference_steps=25,
                    height=384,
                    width=384,
                    controlnet_conditioning_scale=1.0  # default, can try 0.8 for speedup
                ).images[0]

        # Resize result back to display size
        ai_image = cv2.cvtColor(np.array(result.resize((640, 480))), cv2.COLOR_RGB2BGR)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
