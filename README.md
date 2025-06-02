# AI Customer Support Chatbot

A full-stack web application featuring an AI-powered chatbot designed to classify customer support queries into predefined categories. This project demonstrates skills in Natural Language Processing, model training, and web development.


## 🌟 Features

- **NLP-Powered Intent Classification:** The chatbot uses a TensorFlow model to classify user queries into one of five categories: Refund, Technical Issue, Cancellation, Product Inquiry, or Billing.
- **Interactive Web Interface:** A clean, responsive chat interface built with React.
- **Full-Stack Architecture:** A decoupled frontend and backend system.
- **Node.js Backend:** An Express.js server provides an API endpoint to serve predictions from the Python model.
- **Python AI Core:** The prediction logic is handled by a Python script that loads a pre-trained Keras/TensorFlow model.

## 🛠️ Tech Stack

- **Frontend:** React, Axios
- **Backend:** Node.js, Express.js
- **AI / Machine Learning:** Python, TensorFlow (Keras), NLTK, Scikit-learn

## 📂 Project Structure

```
.
├── backend/            # Node.js Express server
├── frontend/           # React user interface
├── venv/               # Python virtual environment
├── chatbot_model.h5    # The trained TensorFlow model
├── tokenizer.pickle    # Keras Tokenizer object
├── label_encoder.pickle# Scikit-learn LabelEncoder object
├── predict.py          # Python script for making predictions
└── train_chatbot.py    # Python script for training the model
```

## 🚀 Running Locally

To run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Setup the Python Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Setup the Backend:**
    ```bash
    cd backend
    npm install
    cd ..
    ```

4.  **Setup the Frontend:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

5.  **Run the Application:**
    - In one terminal, start the backend server:
      ```bash
      cd backend && node index.js
      ```
    - In a second terminal, start the frontend application:
      ```bash
      cd frontend && npm start
      ```

The application will be available at `http://localhost:3000`.
