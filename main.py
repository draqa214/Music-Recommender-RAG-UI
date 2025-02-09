from fastapi import FastAPI
from app_v2_3 import router
from fastapi.middleware.cors import CORSMiddleware
app= FastAPI()

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change "*" to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {"message": "Welcome to the Music Recommender API"}
