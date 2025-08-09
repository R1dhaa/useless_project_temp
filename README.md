<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# Air2Art  🎯


## Basic Details
### Team Name: SERENDIPITY


### Team Members
- Member 1: Fatma Ridha - SCMS School of Engineering and Technology
- Member 2: Gayathri Sudheesh - SCMS School of Engineering and Technology

### Project Description
Air2Art lets you draw invisible sketches in mid-air with your finger — then instantly turns your airy doodles into stunning AI-generated artworks. It’s like magic painting, no canvas required!

### The Problem (that doesn't exist)
Why draw on paper or screens when you can just wave your hand and draw in the air? The problem is, there’s no easy way to turn those air drawings into real art… yet!

### The Solution (that nobody asked for)
You draw in the air, and the AI turns your sketch into a real picture—all using your webcam and some smart code. No mess, no paper, just fun!

## Technical Details
### Technologies/Components Used
For Software:
- Language used: Python 3.x
- Libraries used:
    - OpenCV - for webcam capture, image processing and GUI display
    - MediaPipe - for real time hand landmark detection and tracking
    - OpenAI CLIP - for zero-shot image classification
- Tools used:
    - Python package manager(pip) for installating dependencies
    - Hugging Face CLI for authentication and model downloading
    - Webcam for capturing live video

For Hardware:
- Main components
    -NVIDIA GPU(CUDA-capable) for fast Stable Diffusion and CLIP inference 
  

### Implementation
For Software:
- Hand Tracking
- Sketch classification
- Image generator
- User interface
  
# Installation
pip install opencv-python mediapipe torch torchvision diffusers transformers accelerate pillow ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git  # For CLIP model

# Run
python main.py


### Project Documentation
For Software:

# Screenshots (Add at least 3)
![img1](https://github.com/user-attachments/assets/63f49829-91d8-4d62-b1a0-cf51d66c4431)

The image shows a hand-tracking application called "Air2Art" that uses a skeletal model of a hand to interpret gestures for drawing and generating art.

![img2](https://github.com/user-attachments/assets/06899d15-be3e-4213-be6c-5d924a8caadf)

This image shows a program's console output where an AI model classified a user's sketch as a "boat" and is now using a detailed text prompt to generate a digital painting of a boat.

![img3](https://github.com/user-attachments/assets/88662852-5a6e-42f1-a5df-15a208c8cabd)

This image shows the "Air2Art" application interface, displaying a user's simple line drawing of a boat on the left and the detailed, AI-generated digital painting that resulted from it on the right.

# Diagrams
<img width="673" height="717" alt="flowchart" src="https://github.com/user-attachments/assets/f1a0cd8c-933f-4335-a867-779fb075be12" />

Flowchart


# Build Photos

![img3](https://github.com/user-attachments/assets/c5e67387-f930-4fd7-9e26-50a35b0c4e55)

The final build of the "Air2Art" application is a gesture-based drawing tool that uses AI to classify a user's simple sketch and then generates a corresponding detailed, high-quality digital painting.

### Project Demo
# Video
https://drive.google.com/file/d/1SqwFnhdCBGQWCOdJkRJ_2F5GGnLyXyLV/view?usp=sharing

The video showcases an "Air2Art" application where a user draws a simple boat in the air with their hand. The software's AI then classifies the drawing and generates a high-quality, realistic digital painting of a boat, presenting the transformation from a basic sketch to a refined piece of art.


## Team Contributions
- Fatma Ridha: Contributed to ideation, made commits, contributed to coding, fixed bugs.
- Gayathri Sudheesh: Made commits, contributed to coding, and wrote documentation.
  

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)



