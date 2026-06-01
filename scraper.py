
# ============================================================
# scraper.py — Reddit scraper with comments
# Scrapes posts AND comments for all brands
# Saves everything to Supabase database
# ============================================================

import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session
from database import RedditPost, RedditComment, ScrapeHistory, SessionLocal

load_dotenv()

# ============================================================
# Configuration
# ============================================================
headers = {
    'user-agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

brands = {
    'Kachava':  'https://old.reddit.com/search/?q=kachava&sort=new',
    'Huel':     'https://old.reddit.com/search/?q=huel+shake&sort=new',
    'AG1':      'https://old.reddit.com/search/?q=AG1+athletic+greens&sort=new',
    'Soylent':  'https://old.reddit.com/search/?q=soylent+shake&sort=new',
    'Orgain':   'https://old.reddit.com/search/?q=orgain+shake&sort=new'
}

analyzer = SentimentIntensityAnalyzer()

# ============================================================
# Sentiment Helper
# ============================================================
def get_vader_sentiment(text):
    if not text or text in ['N/A', 'loading...', 'comment']:
        return 'N/A', 0.0
    score    = analyzer.polarity_scores(str(text))
    compound = score['compound']
    if compound >= 0.05:    label = 'Positive'
    elif compound <= -0.05: label = 'Negative'
    else:                   label = 'Neutral'
    return label, round(compound, 4)


# ============================================================
# Extract Comments from a Thread
# ============================================================
def extract_comments(thread_soup, brand, post_url, max_comments=50):
    comments = []

    # Find all comment divs
    comment_divs = thread_soup.find_all('div', class_='usertext-body')

    # Skip index 0 — that's the original post body
    for i, div in enumerate(comment_divs[1:max_comments+1]):
        comment_text = div.get_text(separator=' ', strip=True)

        if not comment_text or len(comment_text) < 10:
            continue

        # Get comment upvotes
        parent = div.find_parent('div', class_='entry')
        upvotes = 0
        if parent:
            score_tag = parent.find('span', class_='score')
            if score_tag:
                try:
                    upvotes = int(score_tag.get_text().split()[0])
                except:
                    upvotes = 0

        # Get comment depth
        depth = 0
        parent_div = div.find_parent('div', class_='child')
        if parent_div:
            depth = 1

        # Sentiment
        vader_label, vader_score = get_vader_sentiment(comment_text)

        comments.append({
            'post_url':        post_url,
            'brand':           brand,
            'comment_text':    comment_text,
            'comment_upvotes': upvotes,
            'comment_depth':   depth,
            'vader_sentiment': vader_label,
            'vader_score':     vader_score,
            'bert_sentiment':  None,  # Will add BERT later
            'bert_score':      None
        })

    return comments


# ============================================================
# Save Post to Database
# ============================================================
def save_post(db: Session, post_data: dict):
    # Check if post already exists
    existing = db.query(RedditPost).filter_by(url=post_data['url']).first()

    if existing:
        # Update existing post
        for key, value in post_data.items():
            setattr(existing, key, value)
        existing.updated_at = datetime.utcnow()
        db.commit()
        return False  # Not new
    else:
        # Create new post
        post = RedditPost(**post_data)
        db.add(post)
        db.commit()
        return True  # New post


# ============================================================
# Save Comments to Database
# ============================================================
def save_comments(db: Session, comments: list):
    saved = 0
    for comment_data in comments:
        comment = RedditComment(**comment_data)
        db.add(comment)
        saved += 1
    db.commit()
    return saved


# ============================================================
# Main Scraper Function
# ============================================================
def scrape_brand(brand_name, brand_url, db: Session):
    print(f'\n{"="*50}')
    print(f'Scraping: {brand_name}')
    print(f'{"="*50}')

    posts_scraped    = 0
    comments_scraped = 0
    error_message    = None

    try:
        # Get search results page
        response = requests.get(brand_url, headers=headers)
        print(f'Status: {response.status_code}')

        if response.status_code != 200:
            raise Exception(f'Failed to load page: {response.status_code}')

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract thread links
        thread_links = []
        for tag in soup.find_all('a', attrs={'data-event-action': 'search'}):
            link = tag['href']
            if link.startswith('/'):
                link = 'https://old.reddit.com' + link
            thread_links.append(link)

        if len(thread_links) == 0:
            for tag in soup.find_all('a', class_='search-title'):
                link = tag['href']
                if link.startswith('/'):
                    link = 'https://old.reddit.com' + link
                thread_links.append(link)

        print(f'Found {len(thread_links)} threads')

        # Scrape each thread
        for i, link in enumerate(thread_links, start=1):
            try:
                thread_response = requests.get(link, headers=headers)

                if thread_response.status_code != 200:
                    print(f'  [{i}] Failed: {thread_response.status_code}')
                    continue

                thread_soup = BeautifulSoup(thread_response.text, 'html.parser')

                # Extract post data
                title_tag = thread_soup.find('a', class_='title')
                title     = title_tag.get_text(strip=True) if title_tag else 'N/A'

                content_tag = thread_soup.find('div', class_='expando')
                content     = content_tag.get_text(separator=' ', strip=True) if content_tag else 'N/A'

                comments_tag = thread_soup.find('a', attrs={'data-event-action': 'comments'})
                if comments_tag:
                    comments_text  = comments_tag.get_text(strip=True)
                    comments_count = int(comments_text.split()[0]) if comments_text else 0
                else:
                    comments_count = 0

                upvotes_tag = (
                    thread_soup.find('div', class_='score unvoted') or
                    thread_soup.find('div', class_='score likes')
                )
                try:
                    upvotes = int(upvotes_tag.get_text(strip=True)) if upvotes_tag else 0
                except:
                    upvotes = 0

                time_tag    = thread_soup.find('time')
                posted_date = time_tag['datetime'] if (time_tag and time_tag.has_attr('datetime')) else 'N/A'

                # Get subreddit
                subreddit_tag = thread_soup.find('a', class_='subreddit')
                subreddit     = subreddit_tag.get_text(strip=True) if subreddit_tag else 'N/A'

                # Sentiment
                vader_label, vader_score = get_vader_sentiment(content)

                # Save post
                post_data = {
                    'brand':           brand_name,
                    'title':           title,
                    'content':         content,
                    'upvotes':         upvotes,
                    'comments_count':  comments_count,
                    'posted_date':     posted_date,
                    'url':             link,
                    'subreddit':       subreddit,
                    'vader_sentiment': vader_label,
                    'vader_score':     vader_score,
                    'bert_sentiment':  None,
                    'bert_score':      None
                }

                is_new = save_post(db, post_data)
                posts_scraped += 1

                # Extract and save comments
                comments = extract_comments(thread_soup, brand_name, link)
                saved    = save_comments(db, comments)
                comments_scraped += saved

                status = '✓ NEW' if is_new else '↻ UPDATED'
                print(f'  [{i}/{len(thread_links)}] {status}: {title[:50]}... ({saved} comments)')

                time.sleep(2)  # Be polite

            except Exception as e:
                print(f'  [{i}] Error: {e}')
                continue

    except Exception as e:
        error_message = str(e)
        print(f'Error scraping {brand_name}: {e}')

    # Save scrape history
    history = ScrapeHistory(
        brand            = brand_name,
        posts_scraped    = posts_scraped,
        comments_scraped = comments_scraped,
        status           = 'success' if not error_message else 'failed',
        error_message    = error_message,
        scraped_at       = datetime.utcnow()
    )
    db.add(history)
    db.commit()

    print(f'\n{brand_name} complete!')
    print(f'  Posts: {posts_scraped}')
    print(f'  Comments: {comments_scraped}')

    return posts_scraped, comments_scraped


# ============================================================
# Run Full Scrape
# ============================================================
def run_full_scrape():
    print('='*50)
    print('  BRAND SENTIMENT SCRAPER')
    print(f'  Started: {datetime.utcnow()}')
    print('='*50)

    db = SessionLocal()
    total_posts    = 0
    total_comments = 0

    try:
        for brand_name, brand_url in brands.items():
            posts, comments  = scrape_brand(brand_name, brand_url, db)
            total_posts     += posts
            total_comments  += comments

    finally:
        db.close()

    print('\n' + '='*50)
    print('  SCRAPE COMPLETE!')
    print(f'  Total posts:    {total_posts}')
    print(f'  Total comments: {total_comments}')
    print(f'  Finished: {datetime.utcnow()}')
    print('='*50)



if __name__ == '__main__':

    # ============================================================
    # On-Demand Scraper — for any brand searched from dashboard
    # ============================================================

    def scrape_brand_on_demand(brand_name: str, max_posts: int = 25):
        """
        Scrape Reddit for any brand name on demand.
        Called directly from app.py when a user searches a new brand.
        Returns (posts_scraped, comments_scraped, error_message)
        """
        # Check if already scraped in last 24 hours
        db = SessionLocal()
        try:
            from sqlalchemy import text as sql_text
            result = db.execute(sql_text('''
                SELECT COUNT(*) FROM scrape_history
                WHERE brand = :brand
                AND scraped_at > NOW() - INTERVAL '24 hours'
                AND status = 'success'
            '''), {'brand': brand_name}).scalar()

            if result > 0:
                print(f'{brand_name} already scraped in last 24hrs, using cache')
                return 0, 0, None  # Cached — no scrape needed

            # Build Reddit search URL dynamically
            query = brand_name.lower().replace(' ', '+')
            brand_url = f'https://old.reddit.com/search/?q={query}&sort=new&limit=25'

            posts, comments = scrape_brand(brand_name, brand_url, db)
            return posts, comments, None

        except Exception as e:
            return 0, 0, str(e)
        finally:
            db.close()

    run_full_scrape()


