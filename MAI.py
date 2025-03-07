import os
import base64
from datetime import datetime

import faiss
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment  # Import pydub for audio processing

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# File paths for knowledge base, chat history, and user data
KNOWLEDGE_BASE_DIR = "documents"
HISTORY_FILE = "history.txt"  # Saved in the root directory for logging
USER_DATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "user_data.txt")  # Saved in the documents directory for RAG

# Ensure the documents directory exists
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# Global flag to enable/disable user_data usage
USE_USER_DATA = False  # Set to False to disable user_data


class RAGSystem:
    """Class to encapsulate the RAG system's state and methods."""

    def __init__(self, knowledge_base_dir):
        """
        Initialize the RAG system.

        Args:
            knowledge_base_dir (str): Directory containing the knowledge base documents.
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.documents = []
        self.index = None
        self.model = None
        self._initialize()

    def _initialize(self):
        """Load documents and create the FAISS index."""
        self.documents = self.load_documents()
        if self.documents:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            document_embeddings = self.model.encode(self.documents)
            dimension = document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(document_embeddings)

    def load_documents(self):
        """Load all text files from the specified directory."""
        documents = []
        for filename in os.listdir(self.knowledge_base_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.knowledge_base_dir, filename), "r", encoding="utf-8") as file:
                    documents.append(file.read())
        return documents

    def update_faiss_index(self, new_documents):
        """
        Update the FAISS index with new documents.

        Args:
            new_documents (list): List of new documents to add.
        """
        if not new_documents:
            return
        new_embeddings = self.model.encode(new_documents)
        self.index.add(new_embeddings)
        self.documents.extend(new_documents)

    def retrieve_documents(self, query, top_k=2):
        """
        Retrieve the top-k most relevant documents based on the query.

        Args:
            query (str): User query.
            top_k (int): Number of documents to retrieve.

        Returns:
            list: Retrieved documents.
        """
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs


def generate_response(query, retrieved_docs, language="en"):
    """
    Generate a personalized response using OpenAI GPT based on the retrieved documents.

    Args:
        query (str): User query.
        retrieved_docs (list): Retrieved documents.
        language (str): Language for the response.

    Returns:
        str: Generated response.
    """
    # Define a static system message for MAI Robot
    system_message = (
        f"You are MAI Robot, a sophisticated AI-powered educational assistant created by Matin. "
        f"Your purpose is to assist users in learning and exploring various educational topics. "
        f"Always maintain a friendly, knowledgeable, and engaging tone in all your interactions. "
        f"Respond in {language.upper()} unless otherwise instructed."
    )

    # Combine retrieved documents into a context string
    context = "\n".join(retrieved_docs)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


def generate_response_with_audio(query, retrieved_docs, language="en", output_file="response.wav"):
    """
    Generate a response with audio using GPT-4o-audio-preview.

    Args:
        query (str): User query.
        retrieved_docs (list): Retrieved documents.
        language (str): Language for the response.
        output_file (str): Path to save the audio file.

    Returns:
        tuple: Generated text and audio file path.
    """
    # Define a static system message for MAI Robot
    system_message = (
        f"You are MAI Robot, a sophisticated AI-powered educational assistant created by Matin. "
        f"Your purpose is to assist users in learning and exploring various educational topics. "
        f"Always maintain a friendly, knowledgeable, and engaging tone in all your interactions. "
        f"Respond in {language.upper()} unless otherwise instructed."
    )

    # Combine retrieved documents into a context string
    context = "\n".join(retrieved_docs)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},  # Customize voice and format
            messages=[
                {"role": "system", "content": system_message},  # Use the MAI Robot system message
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
        )

        # Extract the generated text from the transcript field
        generated_text = completion.choices[0].message.audio.transcript.strip()
        # Extract the audio data
        audio_data = base64.b64decode(completion.choices[0].message.audio.data)  # Decode base64 audio

        # Save the raw audio data to a temporary file
        temp_file = "temp_response.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Re-encode the audio file using pydub to ensure compatibility with Unity
        audio = AudioSegment.from_wav(temp_file)
        audio = audio.set_frame_rate(44100).set_channels(1)  # Convert to 44,100 Hz, mono
        audio.export(output_file, format="wav")  # Export as a standard WAV file

        # Clean up the temporary file
        os.remove(temp_file)

        return generated_text, os.path.abspath(output_file)  # Return the full path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response with audio: {str(e)}")


def speech_to_text(audio_file, language="en"):
    """
    Convert speech from an audio file to text using OpenAI's Whisper model.

    Args:
        audio_file (str): Path to the audio file.
        language (str): Language of the audio.

    Returns:
        str: Transcribed text.
    """
    try:
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                response_format="text",
                language=language,
            )
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting speech to text: {str(e)}")


def text_to_speech(text, language="en", output_file="response.wav"):
    """
    Convert text to speech using OpenAI TTS and save it as a WAV audio file.

    Args:
        text (str): Text to convert.
        language (str): Language of the text.
        output_file (str): Path to save the audio file.

    Returns:
        str: Path to the saved audio file.
    """
    # Map language to appropriate voice (if needed)
    voice_map = {
        "en": "alloy",  # Default English voice
        "fa": "alloy",   # Example Persian voice (you may need to adjust this)
    }
    voice = voice_map.get(language, "alloy")  # Default to "alloy" if language is unsupported

    try:
        # Use OpenAI TTS to generate the audio in MP3 format
        temp_mp3_file = "temp_response.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=text,
        ) as response:
            with open(temp_mp3_file, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

        # Convert the MP3 file to WAV using pydub
        audio = AudioSegment.from_mp3(temp_mp3_file)
        audio = audio.set_frame_rate(44100).set_channels(1)  # Convert to 44,100 Hz, mono
        audio.export(output_file, format="wav")  # Export as a standard WAV file

        # Clean up the temporary MP3 file
        os.remove(temp_mp3_file)

        return os.path.abspath(output_file)  # Return the full path to the WAV file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting text to speech: {str(e)}")


def save_chat_history(user_input, bot_response):
    """
    Save the conversation history to a file.

    Args:
        user_input (str): User input.
        bot_response (str): Bot response.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "a", encoding="utf-8") as file:
        file.write(f"{timestamp} - User: {user_input}\n")
        file.write(f"{timestamp} - Bot: {bot_response}\n\n")


def update_user_data(user_input, bot_response):
    """
    Update the user data file with new information.

    Args:
        user_input (str): User input.
        bot_response (str): Bot response.
    """
    if USE_USER_DATA:  # Only save user data if the flag is enabled
        with open(USER_DATA_FILE, "a", encoding="utf-8") as file:
            file.write(f"User Input: {user_input}\n")
            file.write(f"Bot Response: {bot_response}\n\n")


# Initialize FastAPI app
app = FastAPI()

# Initialize RAG system
rag_system = RAGSystem(KNOWLEDGE_BASE_DIR)


# Define request models
class SpeechRequest(BaseModel):
    audio_file: str
    language: str = "en"


class TextRequest(BaseModel):
    text: str


class SpeechToSpeechRequest(BaseModel):
    audio_file: str
    language: str = "en"


# API endpoint for speech-to-text
@app.post("/speech-to-text")
async def speech_to_text_api(request: SpeechRequest):
    """Convert speech from an audio file to text."""
    try:
        text = speech_to_text(request.audio_file, request.language)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for text-to-speech
@app.post("/text-to-speech")
async def text_to_speech_api(request: TextRequest):
    """Convert text to speech and return the audio file path."""
    try:
        output_file = text_to_speech(request.text)
        return {"audio_file": output_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for generating a response
@app.post("/generate-response")
async def generate_response_api(request: TextRequest):
    """Generate a response using OpenAI GPT with RAG."""
    try:
        retrieved_docs = rag_system.retrieve_documents(request.text)
        response = generate_response(request.text, retrieved_docs)
        save_chat_history(request.text, response)
        update_user_data(request.text, response)  # Only updates if USE_USER_DATA is True
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for speech-to-speech-v2 (renamed from original speech-to-speech)
@app.post("/speech-to-speech-v2")
async def speech_to_speech_v2_api(request: SpeechToSpeechRequest):
    """Convert speech to text, generate a response with audio, and return both."""
    try:
        # Step 1: Convert speech to text
        user_text = speech_to_text(request.audio_file, request.language)
        if not user_text:
            raise HTTPException(status_code=400, detail="Failed to convert speech to text.")

        # Step 2: Retrieve relevant documents
        retrieved_docs = rag_system.retrieve_documents(user_text)

        # Step 3: Generate response with audio
        bot_response, audio_file_path = generate_response_with_audio(user_text, retrieved_docs, language=request.language)

        # Step 4: Save chat history and update user data
        save_chat_history(user_text, bot_response)
        update_user_data(user_text, bot_response)  # Only updates if USE_USER_DATA is True

        # Step 5: Add new user input and bot response to the FAISS index
        if USE_USER_DATA:  # Only update the FAISS index if the flag is enabled
            rag_system.update_faiss_index([f"User Input: {user_text}", f"Bot Response: {bot_response}"])

        return {
            "user_text": user_text,
            "bot_response": bot_response,
            "audio_file": audio_file_path,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# API endpoint for speech-to-speech (new implementation)
@app.post("/speech-to-speech")
async def speech_to_speech_api(request: SpeechToSpeechRequest):
    """Convert speech to text, generate a response, and convert it back to speech."""
    try:
        # Step 1: Convert speech to text
        user_text = speech_to_text(request.audio_file, request.language)
        if not user_text:
            raise HTTPException(status_code=400, detail="Failed to convert speech to text.")

        # Step 2: Retrieve relevant documents
        retrieved_docs = rag_system.retrieve_documents(user_text)

        # Step 3: Generate response
        bot_response = generate_response(user_text, retrieved_docs, language=request.language)

        # Step 4: Convert response to speech
        audio_file_path = text_to_speech(bot_response, language=request.language)

        # Step 5: Save chat history and update user data
        save_chat_history(user_text, bot_response)
        update_user_data(user_text, bot_response)  # Only updates if USE_USER_DATA is True

        # Step 6: Add new user input and bot response to the FAISS index
        if USE_USER_DATA:  # Only update the FAISS index if the flag is enabled
            rag_system.update_faiss_index([f"User Input: {user_text}", f"Bot Response: {bot_response}"])

        return {
            "user_text": user_text,
            "bot_response": bot_response,
            "audio_file": audio_file_path,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)