# 🎵 Recofy - AI Music Recommender

An intelligent music recommendation system that analyzes your Spotify listening history and generates personalized song recommendations using RAG (Retrieval-Augmented Generation) and machine learning.

## ✨ Features

- **Real-time Spotify Integration** - Fetches your recently played tracks
- **AI-Powered Recommendations** - Uses sentence transformers and vector similarity
- **Last.fm Enrichment** - Adds metadata and listening statistics
- **RAG-based Suggestions** - Leverages Groq LLM for contextual recommendations
- **Modern UI** - Dark theme Spotify-like interface built with Next.js
- **Error Handling** - Comprehensive error notifications and logging

## 🚀 Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **Sentence Transformers** - Text embeddings for music similarity
- **Pinecone** - Vector database for similarity search
- **Groq** - Fast LLM inference for recommendations
- **Spotipy** - Spotify Web API integration
- **Scikit-learn** - Feature scaling and preprocessing

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lucide Icons** - Beautiful icon library

## 📋 Prerequisites

- Python 3.12+
- Node.js 18+
- Spotify Developer Account
- Last.fm API Account
- Groq API Key
- Pinecone Account

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Music-Recommender-RAG-UI.git
cd Music-Recommender-RAG-UI
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup
```bash
cd frontend/react
npm install
```

### 4. Environment Variables
Update `.env` with your credentials:
- Spotify Client ID & Secret
- Last.fm API Key & Secret
- Groq API Key
- Pinecone API Key

## 🚀 Running Locally

### Start Backend
```bash
uvicorn main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend/react
npm run dev
```

Visit `http://localhost:3000` to use the application.

## 🌐 Deployment

### Backend (Render)
1. Connect GitHub repository to Render
2. Use `render.yaml` configuration
3. Add environment variables in Render dashboard

### Frontend (Vercel)
1. Connect GitHub repository to Vercel
2. Set root directory to `frontend/react`
3. Add `NEXT_PUBLIC_API_URL` environment variable

### CI/CD Pipeline
GitHub Actions automatically deploys on push to `main` branch.

## 📊 How It Works

1. **Data Collection** - Fetches recent Spotify tracks and Last.fm metadata
2. **Feature Engineering** - Creates embeddings for artists, albums, and AI-generated tags
3. **Vector Search** - Uses Pinecone to find similar songs based on listening patterns
4. **RAG Enhancement** - Groq LLM generates contextual recommendations
5. **Spotify Integration** - Searches and displays playable results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Spotify Web API
- Last.fm API
- Hugging Face Transformers
- Pinecone Vector Database
- Groq AI Platform