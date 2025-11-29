from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

import torch
import torch.nn as nn
import pickle

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentiment Analysis App", version="1.0.0")

# Allow your local React dev server
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # ✅ This must include "OPTIONS"
    allow_headers=["*"],
)

# -----------------------------
# Config
# -----------------------------
# IMPORTANT: Update your DB URL (include your actual postgres password)
# Example: postgresql://postgres:YOUR_PASSWORD@localhost:5432/sentiment_app
DATABASE_URL = "postgresql://postgres:29/7/2025time@localhost:5432/sentiment_app"

# JWT settings (use your own secret in production)
SECRET_KEY = "CHANGE_THIS_TO_A_RANDOM_LONG_SECRET"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Model/vocab paths (already provided by you)
VOCAB_PATH = r"D:\Machine Learning\Recurrent Neural Network\model\vocab.pkl"
MODEL_PATH = r"D:\Machine Learning\Recurrent Neural Network\model\sentiment_model.pth"

# -----------------------------
# Database setup (SQLAlchemy)
# -----------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    review_text = Column(Text, nullable=False)
    sentiment = Column(String(10), nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# Auth setup
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username)
    if not user:
        raise credentials_exception
    return user

# -----------------------------
# ML model (LSTM) loading
# -----------------------------
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# Load vocab
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(len(vocab)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Tokenizer (same as training)
def simple_tokenizer(text: str):
    return text.lower().split()

# -----------------------------
# FastAPI app + schemas
# -----------------------------


class RegisterInput(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    probability: float

# -----------------------------
# Routes
# -----------------------------
@app.post("/register", status_code=201)
def register(input: RegisterInput, db: Session = Depends(get_db)):
    existing = get_user_by_username(db, input.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=input.username, password_hash=hash_password(input.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered successfully"}

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=access_token)

@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Prepare input
    tokens = simple_tokenizer(input.text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens][:200]
    if len(ids) < 200:
        ids += [vocab["<pad>"]] * (200 - len(ids))
    X = torch.tensor([ids]).to(device)

    # Predict
    with torch.no_grad():
        prob = model(X).item()

    sentiment = "positive" if prob >= 0.5 else "negative"

    # Store prediction
    record = Prediction(
        user_id=current_user.id,
        review_text=input.text,
        sentiment=sentiment,
        probability=float(prob),
    )
    db.add(record)
    db.commit()

    return PredictResponse(sentiment=sentiment, probability=float(prob))

@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "created_at": current_user.created_at.isoformat(),
    }

@app.get("/predictions")
def list_predictions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "text": r.review_text,
            "sentiment": r.sentiment,
            "probability": r.probability,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]

@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    prediction = db.query(Prediction).filter_by(id=prediction_id, user_id=current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(prediction)
    db.commit()
    return {"message": "Prediction deleted"}