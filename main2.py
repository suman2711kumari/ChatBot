from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import json
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


for model in genai.list_models():
    print(model.name)

print("Listing available models...\n")

for model in genai.list_models():
    print(model.name, model.supported_generation_methods)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Gemini model
try:
    gemini_model = genai.GenerativeModel('gemini-3.6-flash')
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    gemini_model = None

# Load local models
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Serve static files (your frontend) from 'static' folder
#app.mount("/static", StaticFiles(directory="chatbot/static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load merged lexicon once at startup
try:
    with open('merged_nrc_lexicon.json', 'r', encoding='utf-8') as f:
        merged_lexicon = json.load(f)
except FileNotFoundError:
    print("Warning: merged_nrc_lexicon.json not found. Emotion detection will be limited.")
    merged_lexicon = {}

# Initialize DB
def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            message TEXT,
            reply TEXT,
            depression_score REAL,
            model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Define which emotions you consider related to depression
DEPRESSION_EMOTIONS = {'sadness', 'fear', 'anger', 'disgust'}

emotion_reply_map = {
    'sad': "I'm sorry to hear you're feeling sad. Would you like to talk more about it?",
    'depressed': "It sounds like you're going through a tough time. Remember, you're not alone.",
    'anxious': "Anxiety can be overwhelming. Try to take deep breaths and focus on the present moment.",
    'lonely': "Feeling lonely is hard. I'm here to listen whenever you want to share.",
    'suicidal': "I'm really concerned about your safety. Please consider reaching out to a mental health professional or a trusted person.",
    'default': "Thank you for sharing. Can you tell me more about how you're feeling?",
    'sadness': "I'm sorry to hear you're feeling sad. Would you like to talk more about it?",
    'fear': "It sounds like you're worried. I'm here to listen.",
    'anger': "I understand you're upset. Let's talk about it.",
    'disgust': "That sounds difficult. I'm here for you.",
}

CRITICAL_PHRASES = [
    "i want to die",
    "i want to kill myself",
    "i am going to hurt myself",
    "i can't go on",
    "i feel hopeless",
]

HELPLINE_MESSAGE = (
    "I'm really sorry you're feeling this way. "
    "Please consider reaching out to a mental health professional or helpline immediately. "
    "You are not alone."
)

# Download NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

lemmatizer = WordNetLemmatizer()

def lemmatize_word(word):
    return lemmatizer.lemmatize(word)

def detect_dominant_emotion(text: str) -> str:
    if not merged_lexicon:
        return None
    
    words = re.findall(r'\b\w+\b', text.lower())
    emotion_scores = {}
    for word in words:
        if word in merged_lexicon:
            for emotion, weight in merged_lexicon[word].items():
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + weight
    if not emotion_scores:
        return None
    return max(emotion_scores, key=emotion_scores.get)

analyzer = SentimentIntensityAnalyzer()

def calculate_depression_score(text: str) -> float:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    score = 0.0

    for word in words:
        lemma = lemmatize_word(word)
        if lemma in merged_lexicon:
            emotions = merged_lexicon[lemma]
            for emotion, weight in emotions.items():
                if emotion in DEPRESSION_EMOTIONS:
                    score += weight

    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] < -0.3:
        score += abs(sentiment['compound']) * 5

    return score

def check_critical_message(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in CRITICAL_PHRASES)

def should_use_gemini(user_message: str) -> bool:
    """
    Always use Gemini for all non-rule-based responses
    """
    return True

async def generate_gemini_reply(user_message: str, user_id: str) -> tuple[str, str]:
    """
    Generate reply using Gemini API
    Returns (reply, model_used)
    """
    if not gemini_model:
        return "I'm having trouble accessing my knowledge base right now. Please try again later.", "gemini_error"
    
    try:
        # Create a prompt that maintains the supportive chatbot character
        prompt = f"""You are a supportive and empathetic chatbot designed to help people with their questions and concerns. 
        Please respond to the following message in a helpful, caring, and informative way. 
        Keep your response concise but thorough, and maintain a warm, supportive tone.

        User message: {user_message}

        Response:"""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), "gemini"
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to local model
        return generate_local_llm_reply(user_id, user_message), "dialogpt_fallback"

chat_histories = {}

def generate_local_llm_reply(user_id: str, user_message: str) -> str:
    """
    Generate reply using local DialoGPT model
    """
    global chat_histories
    try:
        input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
        if user_id in chat_histories:
            bot_input_ids = torch.cat([chat_histories[user_id], input_ids], dim=-1)
        else:
            bot_input_ids = input_ids
        
        chat_histories[user_id] = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        reply = tokenizer.decode(chat_histories[user_id][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return reply.strip() if reply.strip() else "I understand. Can you tell me more?"
    except Exception as e:
        print(f"Local model error: {e}")
        return "I'm here to listen. Can you tell me more about what's on your mind?"

async def generate_reply(user_id, text):

    print("Entered generate_reply")

    if check_critical_message(text):
        print("Critical response")
        return HELPLINE_MESSAGE, "crisis_response"

    dominant_emotion = detect_dominant_emotion(text)
    print("Dominant emotion:", dominant_emotion)

    if dominant_emotion and dominant_emotion in emotion_reply_map:
        print("Emotion template")
        return emotion_reply_map[dominant_emotion], "emotion_template"

    print("Calling Gemini...")
    return await generate_gemini_reply(text, user_id)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    user_id = chat_req.user_id
    message = chat_req.message

    if not user_id or not message:
        return JSONResponse(status_code=400, content={"error": "user_id and message required"})

    depression_score = calculate_depression_score(message)
    reply, model_used = await generate_reply(user_id, message)

    # Store in database with model information
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO messages (user_id, message, reply, depression_score, model_used, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, message, reply, depression_score, model_used, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

    return {
        "reply": reply, 
        "depression_score": depression_score,
        "model_used": model_used
    }

@app.get("/admin/stats")
async def admin_stats():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    since = datetime.now(timezone.utc) - timedelta(hours=24)

    # Get depression trend data grouped by hour
    c.execute('''
        SELECT DATE(timestamp) as date,
               strftime('%H:00', timestamp) as hour,  
               AVG(depression_score) as avg_depression
        FROM messages
        WHERE timestamp >= ?
        AND depression_score IS NOT NULL
        GROUP BY date, hour
        ORDER BY date, hour
    ''', (since.isoformat(),))
    rows = c.fetchall()

    dates = []
    avg_scores = []

    for row in rows:
        dates.append(f"{row[0]} {row[1]}")
        avg_scores.append(round(row[2], 2) if row[2] is not None else 0)
        
    # If no data, provide sample data
    if not dates:
        current_time = datetime.now()
        dates = [f"{current_time.strftime('%Y-%m-%d')} {i:02d}:00" for i in range(max(0, current_time.hour-5), current_time.hour+1)]
        avg_scores = [0] * len(dates)
    
    # Total distinct active users in last hour
    since_hour = datetime.now(timezone.utc) - timedelta(hours=1)
    c.execute('''
        SELECT COUNT(DISTINCT user_id)
        FROM messages
        WHERE timestamp >= ?
        ''', (since_hour.isoformat(),))
    total_active_users = c.fetchone()[0]

    # Model usage statistics
    c.execute('''
        SELECT model_used, COUNT(*) as count
        FROM messages
        WHERE timestamp >= ?
        GROUP BY model_used
    ''', (since.isoformat(),))
    model_stats = dict(c.fetchall())

    conn.close()

    return {
        "total_active_users": total_active_users,
        "depression_trend": {
            "dates": dates,
            "avg_scores": avg_scores
        },
        "model_usage": model_stats
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    gemini_status = "available" if gemini_model else "unavailable"
    return {
        "status": "healthy",
        "gemini_api": gemini_status,
        "local_model": "available"
    }

# Serve frontend index.html at root
@app.get("/")
async def root():
    #index_path = os.path.join("chatbot", "static", "chatbot.html")
    index_path = os.path.join("static", "chatbot.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"message": "Chatbot API is running. Frontend not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)