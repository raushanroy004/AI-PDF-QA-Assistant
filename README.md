📘 AI PDF Q&A Assistant (RAG)
Upload any PDF → Ask questions → Get instant AI-powered answers

🚀 Live Demo:
👉 https://pdf-expert-ai.streamlit.app/

📦 Tech Stack: Streamlit · Python · FAISS · Groq · HuggingFace · LangChain-style RAG
🎯 Use Case: Students, Researchers, Developers, Lawyers, Anyone reading long PDFs

🌟 Features

✔ Upload any PDF (Notes, Books, Research Papers, Reports)
✔ Automatic Text Extraction
✔ Smart Text Cleaning & Chunking
✔ FAISS Vector Search for fast document retrieval
✔ Groq-powered RAG answers (short + long explanation)
✔ Ask by typing or using voice
✔ Live microphone input support
✔ Dark UI theme (clean & modern)
✔ No local model dependency – fully cloud-based
✔ Deployed on Streamlit Cloud

🧠 How It Works

This project uses a lightweight RAG (Retrieval-Augmented Generation) pipeline:

PDF → Extract Text → Clean → Chunk → Embeddings → FAISS Index
                           ↓
                    User Question
                           ↓
                Top Relevant Chunks Retrieved
                           ↓
          Sent to Groq LLM → AI Answer (short + long)

🚀 Tech Stack
Component	Technology
App Framework	Streamlit
Embeddings	sentence-transformers / all-MiniLM-L6-v2
Vector DB	FAISS
LLM (RAG Answering)	Groq (mixtral-8x7b or llama3-8b)
Audio STT	Whisper (HuggingFace)
Optional TTS	Groq TTS (can be enabled later)
📂 Project Structure
ai-pdf-qa-assistant/
│
├── app.py                      # Main Streamlit App
│
├── utils/
│   ├── pdf_reader.py           # Extract text from PDF
│   ├── text_processing.py      # Clean & chunk text
│   ├── embeddings.py           # Create embeddings + FAISS index
│   ├── rag_pipeline.py         # RAG answer generator
│   ├── voice_utils.py          # STT using Whisper
│   ├── tts_utils.py            # TTS (optional)
│   └── __init__.py
│
├── requirements.txt
├── README.md  (THIS FILE)
└── .streamlit/secrets.toml     # Streamlit secrets (cloud only)

⚙️ Installation (Local Setup)
1️⃣ Clone the repository
git clone https://github.com/YOUR-USERNAME/ai-pdf-qa-assistant.git
cd ai-pdf-qa-assistant

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add environment variables

Create .streamlit/secrets.toml:

GROQ_API_KEY = "your_groq_key_here"
HF_API_KEY = "your_huggingface_key_here"

5️⃣ Run the app
streamlit run app.py

☁️ Deployment on Streamlit Cloud

Push repo to GitHub

Go to https://share.streamlit.io

Create new app → Select repo

Add secrets:

GROQ_API_KEY = "your_groq_key_here"
HF_API_KEY = "your_huggingface_key_here"


Click Deploy 🎉

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

📜 License

MIT License © 2025 YOUR NAME

⭐ Support The Project

If you like this project, please ⭐ the repository!
Your support encourages development of more AI tools 😊.
