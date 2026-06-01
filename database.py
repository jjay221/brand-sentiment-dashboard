
# ============================================================
# database.py — Sets up PostgreSQL database tables
# Run this ONCE to create all tables in Supabase
# ============================================================

import os
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
Base   = declarative_base()

# ============================================================
# TABLE 1: Reddit Posts
# ============================================================
class RedditPost(Base):
    __tablename__ = 'reddit_posts'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    brand         = Column(String(100), nullable=False)
    title         = Column(String(500))
    content       = Column(Text)
    upvotes       = Column(Integer)
    comments_count = Column(Integer)
    posted_date   = Column(String(100))
    url           = Column(String(500), unique=True)
    subreddit     = Column(String(100))

    # Sentiment scores
    vader_sentiment  = Column(String(20))
    vader_score      = Column(Float)
    bert_sentiment   = Column(String(20))
    bert_score       = Column(Float)

    # Metadata
    scraped_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<RedditPost {self.brand}: {self.title[:50]}>'


# ============================================================
# TABLE 2: Reddit Comments
# ============================================================
class RedditComment(Base):
    __tablename__ = 'reddit_comments'

    id              = Column(Integer, primary_key=True, autoincrement=True)
    post_url        = Column(String(500))
    brand           = Column(String(100))
    comment_text    = Column(Text)
    comment_upvotes = Column(Integer)
    comment_depth   = Column(Integer)  # 0 = top level, 1 = reply, etc.

    # Sentiment
    vader_sentiment = Column(String(20))
    vader_score     = Column(Float)
    bert_sentiment  = Column(String(20))
    bert_score      = Column(Float)

    # Metadata
    scraped_at      = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<RedditComment {self.brand}: {self.comment_text[:50]}>'


# ============================================================
# TABLE 3: Scrape History (tracks when scrapes happened)
# ============================================================
class ScrapeHistory(Base):
    __tablename__ = 'scrape_history'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    brand         = Column(String(100))
    posts_scraped = Column(Integer)
    comments_scraped = Column(Integer)
    status        = Column(String(50))  # success, failed, partial
    error_message = Column(Text)
    scraped_at    = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ScrapeHistory {self.brand} at {self.scraped_at}>'


# ============================================================
# Create all tables
# ============================================================
def create_tables():
    print('Creating database tables...')
    Base.metadata.create_all(engine)
    print('Tables created successfully!')
    print()
    print('Tables in database:')
    print('  - reddit_posts')
    print('  - reddit_comments')
    print('  - scrape_history')

# Create session factory
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == '__main__':
    create_tables()

    
