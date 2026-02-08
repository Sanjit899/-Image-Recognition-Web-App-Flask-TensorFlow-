🖼️ Image Recognition Web App (Flask + TensorFlow)

A Flask-based web application that performs offline image recognition using a pre-trained MobileNetV2 model from TensorFlow.
The app allows users to upload an image through a secure web interface and receive the top-3 predicted labels with confidence scores.

🔐 Includes CSRF protection
🌙 Dark mode UI
🚫 No external APIs required
🌐 Deployable on platforms like Render.

🚀 Features

Upload an image via web browser

Offline image classification using TensorFlow

Top-3 predictions with confidence percentages

Dark-themed responsive UI

CSRF protection using Flask-WTF

Secure file upload validation

Production-ready structure (Gunicorn compatible).


🛠️ Tech Stack

Backend: Flask (Python)

ML Model: TensorFlow – MobileNetV2 (ImageNet)

Frontend: HTML, CSS (Dark Mode)

Security: CSRF Protection (Flask-WTF)

Deployment: Render / Gunicorn

Environment: Python Virtual Environment (venv).



⚙️ Installation & Setup (Local)

1️⃣ Clone the repository
image-recognition-flask

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows


3️⃣ Install dependencies
write  pip  installl   and  write all  the  libraries  that are used in this project  it will  downloaded .

▶️ Run the Application
python app.py
Open your browser and go to:- http://127.0.0.1:5000

🔐 Security Features

CSRF protection enabled using Flask-WTF

File upload validation (image formats only)

Secret key configuration supported via environment variables



🧠 Future Improvements

Object detection (YOLO)

User authentication

Rate limiting

Drag-and-drop uploads

Docker containerization.


📄 License
This project is for educational and demonstration purposes.

