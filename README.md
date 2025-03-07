# MAI - AI-powered Educational Assistant

## Overview
MAI is an AI-powered educational assistant built with FastAPI and OpenAI's GPT-4. It integrates Retrieval-Augmented Generation (RAG) for knowledge retrieval, speech-to-text conversion using OpenAI's Whisper, and text-to-speech synthesis for audio responses. Additionally, MAI supports audio re-encoding to ensure compatibility with Unity and other applications.

## Features
- **Retrieval-Augmented Generation (RAG):** Enhances chatbot responses by retrieving relevant knowledge base documents using FAISS.
- **OpenAI GPT-4 Integration:** Provides intelligent responses based on user queries.
- **Speech-to-Text (STT):** Converts spoken words to text using OpenAI's Whisper model.
- **Text-to-Speech (TTS):** Converts text responses to high-quality WAV audio output.
- **Speech-to-Speech (S2S):** Enables end-to-end voice interaction by processing user speech and responding with generated audio.
- **Persistent Chat History:** Logs user queries and responses for analysis.
- **User Data Management:** Stores user interactions for improved personalization (optional).

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)

### Setup
1. Clone the repository:
   ```sh
   git clone [<repository_url>](https://github.com/Matin-Ar/MAI-Robot.git)
   cd MAI-Robot
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file and set your OpenAI API key:
   ```sh
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

5. Run the application:
   ```sh
   uvicorn MAI:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints
### 1. Speech-to-Text (STT)
- **Endpoint:** `POST /speech-to-text`
- **Description:** Converts speech from an audio file to text.
- **Request Body:**
  ```json
  {
    "audio_file": "path/to/audio.wav",
    "language": "en"
  }
  ```
- **Response:**
  ```json
  {"text": "Transcribed text"}
  ```

### 2. Text-to-Speech (TTS)
- **Endpoint:** `POST /text-to-speech`
- **Description:** Converts text to speech and returns the audio file path.
- **Request Body:**
  ```json
  {"text": "Hello, world!"}
  ```
- **Response:**
  ```json
  {"audio_file": "path/to/output.wav"}
  ```

### 3. Generate Response
- **Endpoint:** `POST /generate-response`
- **Description:** Generates a response using OpenAI GPT with RAG.
- **Request Body:**
  ```json
  {"text": "What is AI?"}
  ```
- **Response:**
  ```json
  {"response": "AI stands for Artificial Intelligence..."}
  ```

### 4. Speech-to-Speech
- **Endpoint:** `POST /speech-to-speech`
- **Description:** Converts speech to text, generates a response, and converts it back to speech.
- **Request Body:**
  ```json
  {
    "audio_file": "path/to/audio.wav",
    "language": "en"
  }
  ```
- **Response:**
  ```json
  {
    "user_text": "What is AI?",
    "bot_response": "AI stands for Artificial Intelligence...",
    "audio_file": "path/to/output.wav"
  }
  ```

## File Structure
```
├── documents/                 # Knowledge base documents
│   ├── user_data.txt          # User interaction data (optional)
├── MAI.py                     # Main application script
├── history.txt                 # Chat history log
├── requirements.txt            # Dependencies
├── .env                        # API key storage
```

## Dependencies
- `fastapi`
- `uvicorn`
- `openai`
- `pydantic`
- `faiss`
- `numpy`
- `sentence-transformers`
- `pydub`
- `python-dotenv`

## 📜 License
This project is **open-source** under the **MIT License**. Feel free to use, modify, and distribute it.

## 🤝 Contributing
If you want to improve the project:
1. **Fork the repository**
2. **Make changes in a new branch**
3. **Submit a pull request**

## 📬 Contact
- **Author:** Matin
- **GitHub:** [Matin-Ar](https://github.com/Matin-Ar)
- **VR MAI Robot Repo:** [VR-MAI-Robot](https://github.com/Matin-Ar/VR-MAI-Robot)
