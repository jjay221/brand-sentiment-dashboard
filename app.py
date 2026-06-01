
# ============================================================
# app.py — Main Dashboard Connected to Supabase Database
# Run: python app.py
# Then open: http://127.0.0.1:8050
# ============================================================

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import re
from collections import Counter

load_dotenv()

# ============================================================
# Database Connection
# ============================================================
DATABASE_URL = os.getenv('DATABASE_URL')
engine       = create_engine(DATABASE_URL)

def load_posts():
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM reddit_posts', conn)
    df['upvotes']        = pd.to_numeric(df['upvotes'],        errors='coerce')
    df['comments_count'] = pd.to_numeric(df['comments_count'], errors='coerce')
    df['posted_date']    = pd.to_datetime(df['posted_date'],   errors='coerce')
    df['month']          = df['posted_date'].dt.to_period('M').astype(str)
    return df

def load_comments():
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM reddit_comments', conn)
    df['comment_upvotes'] = pd.to_numeric(df['comment_upvotes'], errors='coerce')
    return df

def load_scrape_history():
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM scrape_history ORDER BY scraped_at DESC', conn)
    return df

def load_brands():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT DISTINCT brand FROM reddit_posts ORDER BY brand'))
        return [row[0] for row in result]

# ============================================================
# Helper Functions
# ============================================================
stop_words = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'i','my','me','it','is','was','be','have','has','had','this','that',
    'are','so','just','not','do','did','if','up','out','as','by','from',
    'you','your','we','they','them','their','its','been','will','can','all',
    'about','get','got','more','also','back','one','when','what','how','no',
    'now','into','like','over','after','previous','next','view','grid',
    'he','she','his','her','some','would','which','there','than','then',
    'very','really','new','good','even','still','most','don','much',
    'loading','comment','nan','http','https','www','null'
])

colors = {
    'Positive': '#2ecc71',
    'Negative': '#e74c3c',
    'Neutral':  '#95a5a6',
    'N/A':      '#3498db'
}

def get_top_words(text_series, n=15):
    all_text = ' '.join(text_series.dropna().astype(str).tolist()).lower()
    words    = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered = [w for w in words if w not in stop_words]
    return Counter(filtered).most_common(n)

# ============================================================
# Initialize App
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server  # For Render deployment

# ============================================================
# Layout
# ============================================================
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H1('🏆 Brand Sentiment Intelligence Dashboard',
# Intro Card
dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4('🔍 What is BrandPulse?',
                                style={'color': '#3498db',
                                       'fontWeight': 'bold'}),
                        html.P(
                            'BrandPulse is a free AI-powered brand '
                            'intelligence tool that analyzes what real '
                            'people are saying about nutrition and meal '
                            'replacement brands on Reddit. We collect '
                            'hundreds of posts and comments daily, then '
                            'use artificial intelligence to give you '
                            'real-time consumer insights.',
                            style={'color': '#bdc3c7', 'fontSize': '15px'}
                        ),
                    ], width=8),
                    dbc.Col([
                        html.H5('What you can discover:',
                                style={'color': 'white'}),
                        html.Ul([
                            html.Li('✅ Is Reddit positive or negative about this brand?',
                                   style={'color': '#bdc3c7', 'marginBottom': '5px'}),
                            html.Li('💬 What are people actually saying in comments?',
                                   style={'color': '#bdc3c7', 'marginBottom': '5px'}),
                            html.Li('📊 How does this brand compare to competitors?',
                                   style={'color': '#bdc3c7', 'marginBottom': '5px'}),
                            html.Li('🔮 What topics do people discuss most?',
                                   style={'color': '#bdc3c7', 'marginBottom': '5px'}),
                        ], style={'paddingLeft': '20px'}),
                        html.P(
                            '📌 Tracking: Kachava | Huel | AG1 | Soylent | Orgain',
                            style={'color': '#f39c12', 'fontWeight': 'bold',
                                   'marginTop': '10px'}
                        ),
                    ], width=4),
                ]),
            ])
        ], style={'background': '#1a252f',
                  'borderLeft': '4px solid #3498db',
                  'marginBottom': '20px'})
    ])
], className='mb-3'),
                    style={'color': '#3498db', 'fontWeight': 'bold', 'marginTop': '20px'}),
            html.P('Real-time Reddit competitor analysis — Powered by AI',
                   style={'color': '#95a5a6', 'fontSize': '16px'}),
            html.Hr(style={'borderColor': '#2c3e50'})
        ])
    ]),

    # KPI Cards - loaded dynamically
    html.Div(id='kpi-cards', className='mb-4'),

    # Filters Row
    dbc.Row([
        dbc.Col([
            html.Label('Filter by Brand:', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='brand-filter',
                options=[],  # Loaded dynamically
                value='All',
                clearable=False,
                style={'color': 'black'}
            )
        ], width=3),

        dbc.Col([
            html.Label('Select Metric:', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='metric-filter',
                options=[
                    {'label': 'Upvotes',   'value': 'upvotes'},
                    {'label': 'Comments',  'value': 'comments_count'}
                ],
                value='upvotes',
                clearable=False,
                style={'color': 'black'}
            )
        ], width=3),

        dbc.Col([
            html.Label('Data Source:', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='source-filter',
                options=[
                    {'label': 'Posts Only',    'value': 'posts'},
                    {'label': 'Comments Only', 'value': 'comments'},
                    {'label': 'Both',          'value': 'both'}
                ],
                value='both',
                clearable=False,
                style={'color': 'black'}
            )
        ], width=3),

        dbc.Col([
            html.Label('Refresh Data:', style={'color': 'white', 'fontWeight': 'bold'}),
            html.Br(),
            dbc.Button('🔄 Refresh', id='refresh-btn', color='primary', n_clicks=0)
        ], width=3),
    ], className='mb-4'),

    # Main Tabs
    dbc.Tabs([

        # ── Tab 1: Overview ──────────────────────────────────
        dbc.Tab(label='📊 Overview', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='sentiment-bar'),  width=6),
                dbc.Col(dcc.Graph(id='sentiment-pie'),  width=6),
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='positive-rate'),  width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 2: Engagement ─────────────────────────────────
        dbc.Tab(label='💬 Engagement', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='engagement-scatter'), width=8),
                dbc.Col(dcc.Graph(id='avg-metrics'),        width=4),
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='timeline'),           width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 3: Comments Analysis ──────────────────────────
        dbc.Tab(label='💭 Comments Analysis', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='comment-sentiment'),  width=6),
                dbc.Col(dcc.Graph(id='comment-volume'),     width=6),
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='comment-words'),      width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 4: Word Analysis ──────────────────────────────
        dbc.Tab(label='🔤 Word Analysis', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select Brand:',
                               style={'color': 'white', 'fontWeight': 'bold', 'marginTop': '15px'}),
                    dcc.Dropdown(
                        id='word-brand-filter',
                        options=[],
                        value=None,
                        clearable=False,
                        style={'color': 'black'}
                    )
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='word-pos'),  width=6),
                dbc.Col(dcc.Graph(id='word-neg'),  width=6),
            ], className='mt-3'),
        ]),

        # ── Tab 5: Posts Explorer ─────────────────────────────
        dbc.Tab(label='🔍 Posts Explorer', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='posts-table'), width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 6: Comments Explorer ──────────────────────────
        dbc.Tab(label='📝 Comments Explorer', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='comments-table'), width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 7: Brand Manager ──────────────────────────────
        dbc.Tab(label='⚙️ Brand Manager', children=[
            dbc.Row([
                dbc.Col([
                    html.H4('Manage Brands', style={'color': 'white', 'marginTop': '20px'}),
                    html.P('Add or remove brands to track. The scraper will collect Reddit posts and comments for each brand.',
                           style={'color': '#95a5a6'}),

                    # Add new brand
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('➕ Add New Brand', style={'color': 'white'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label('Brand Name:', style={'color': 'white'}),
                                    dbc.Input(id='new-brand-name',
                                             placeholder='e.g. Premier Protein',
                                             type='text', className='mb-2')
                                ], width=4),
                                dbc.Col([
                                    html.Label('Reddit Search URL:', style={'color': 'white'}),
                                    dbc.Input(id='new-brand-url',
                                             placeholder='https://old.reddit.com/search/?q=premier+protein&sort=new',
                                             type='text', className='mb-2')
                                ], width=6),
                                dbc.Col([
                                    html.Label(' ', style={'color': 'white'}),
                                    html.Br(),
                                    dbc.Button('Add Brand', id='add-brand-btn',
                                              color='success', n_clicks=0)
                                ], width=2),
                            ]),
                            html.Div(id='add-brand-output',
                                    style={'color': '#2ecc71', 'marginTop': '10px'})
                        ])
                    ], style={'background': '#1a252f', 'marginBottom': '20px'}),

                    # Current brands table
                    html.H5('Current Brands:', style={'color': 'white'}),
                    html.Div(id='brands-table')

                ], width=10),
            ], className='mt-3'),
        ]),

        # ── Tab 8: Scrape History ─────────────────────────────
        dbc.Tab(label='📅 Scrape History', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='history-table'), width=12),
            ], className='mt-3'),
        ]),

        # ── Tab 9: Predict Post ───────────────────────────────
        dbc.Tab(label='🤖 Predict Post', children=[
            dbc.Row([
                dbc.Col([
                    html.H4('Will your post get High or Low upvotes?',
                            style={'color': 'white', 'marginTop': '20px'}),
                    html.Label('Post Title:', style={'color': 'white'}),
                    dbc.Input(id='pred-title',
                             placeholder='Enter post title...',
                             type='text', className='mb-2'),
                    html.Label('Post Content:', style={'color': 'white'}),
                    dbc.Textarea(id='pred-content',
                                placeholder='Enter post content...',
                                style={'height': '120px'}, className='mb-2'),
                    html.Label('Brand:', style={'color': 'white'}),
                    dcc.Dropdown(
                        id='pred-brand',
                        options=[],
                        value=None,
                        clearable=False,
                        style={'color': 'black', 'marginBottom': '10px'}
                    ),
                    dbc.Button('🔮 Predict!', id='predict-btn',
                              color='primary', className='mb-3', n_clicks=0),
                    html.Div(id='prediction-output',
                            style={'fontSize': '18px', 'fontWeight': 'bold',
                                   'color': 'white', 'padding': '20px',
                                   'background': '#1a252f', 'borderRadius': '10px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='sentiment-gauge')
                ], width=6),
            ], className='mt-3'),
        ]),

# ── Tab 10: Search Any Brand ──────────────────────────
        dbc.Tab(label='🔍 Search Any Brand', children=[
            dbc.Row([
                dbc.Col([
                    html.H4('Search Any Brand on Reddit',
                            style={'color': 'white', 'marginTop': '20px'}),
                    html.P('Type any brand name and we\'ll scrape Reddit for sentiment analysis in real-time.',
                           style={'color': '#95a5a6'}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id='search-brand-input',
                                placeholder='e.g. Nike, Protein World, Gorilla Mind...',
                                type='text',
                                style={'fontSize': '16px'}
                            )
                        ], width=7),
                        dbc.Col([
                            dbc.Button('🔍 Search Reddit',
                                       id='search-brand-btn',
                                       color='primary',
                                       size='lg',
                                       n_clicks=0)
                        ], width=2),
                    ], className='mb-3'),
                    html.Div(id='search-brand-status',
                             style={'color': '#f39c12', 'fontSize': '16px',
                                    'marginBottom': '15px'}),
                    html.Div(id='search-brand-results'),
                ], width=10),
            ], className='mt-3'),
        ]),
    ]),

    # Auto refresh every 5 minutes


    dcc.Interval(id='interval', interval=300000, n_intervals=0)

], fluid=True, style={'backgroundColor': '#0d1117', 'minHeight': '100vh'})


# ============================================================
# Callbacks
# ============================================================

# Load brand options on startup
@app.callback(
    Output('brand-filter',      'options'),
    Output('brand-filter',      'value'),
    Output('word-brand-filter', 'options'),
    Output('word-brand-filter', 'value'),
    Output('pred-brand',        'options'),
    Output('pred-brand',        'value'),
    Input('interval',           'n_intervals'),
    Input('refresh-btn',        'n_clicks')
)
def load_brand_options(n, clicks):
    brands   = load_brands()
    options  = [{'label': 'All Brands', 'value': 'All'}] + \
               [{'label': b, 'value': b} for b in brands]
    w_opts   = [{'label': b, 'value': b} for b in brands]
    first    = brands[0] if brands else None
    return options, 'All', w_opts, first, w_opts, first


# KPI Cards
@app.callback(
    Output('kpi-cards', 'children'),
    Input('brand-filter', 'value'),
    Input('interval',     'n_intervals')
)
def update_kpis(brand, n):
    df       = load_posts()
    df_com   = load_comments()

    if brand != 'All':
        df     = df[df['brand'] == brand]
        df_com = df_com[df_com['brand'] == brand]

    total_posts    = len(df)
    total_comments = len(df_com)
    pos_pct        = round(len(df[df['vader_sentiment'] == 'Positive']) / max(total_posts, 1) * 100, 1)
    neg_pct        = round(len(df[df['vader_sentiment'] == 'Negative']) / max(total_posts, 1) * 100, 1)
    avg_up         = round(df['upvotes'].mean(), 1) if total_posts > 0 else 0

    card_style = {'textAlign': 'center', 'padding': '15px',
                  'borderRadius': '10px', 'background': '#1a252f'}

    return dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(total_posts, className='card-title', style={'color': '#3498db'}),
            html.P('Total Posts', className='card-text')
        ])], style=card_style), width=2),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(total_comments, className='card-title', style={'color': '#9b59b6'}),
            html.P('Total Comments', className='card-text')
        ])], style=card_style), width=2),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(f'{pos_pct}%', className='card-title', style={'color': '#2ecc71'}),
            html.P('Positive Rate', className='card-text')
        ])], style=card_style), width=2),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(f'{neg_pct}%', className='card-title', style={'color': '#e74c3c'}),
            html.P('Negative Rate', className='card-text')
        ])], style=card_style), width=2),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(avg_up, className='card-title', style={'color': '#f39c12'}),
            html.P('Avg Upvotes', className='card-text')
        ])], style=card_style), width=2),

        dbc.Col(dbc.Card([dbc.CardBody([
            html.H2(len(load_brands()), className='card-title', style={'color': '#1abc9c'}),
            html.P('Brands Tracked', className='card-text')
        ])], style=card_style), width=2),
    ])


# Sentiment Bar
@app.callback(Output('sentiment-bar', 'figure'), Input('brand-filter', 'value'))
def update_sentiment_bar(brand):
    df      = load_posts()
    if brand != 'All':
        df  = df[df['brand'] == brand]
    summary = df.groupby(['brand', 'vader_sentiment']).size().reset_index(name='Count')
    fig     = px.bar(summary, x='brand', y='Count', color='vader_sentiment',
                     color_discrete_map=colors, barmode='group',
                     title='Post Sentiment by Brand', template='plotly_dark')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Sentiment Pie
@app.callback(Output('sentiment-pie', 'figure'), Input('brand-filter', 'value'))
def update_pie(brand):
    df     = load_posts()
    if brand != 'All':
        df = df[df['brand'] == brand]
    counts = df['vader_sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    fig    = px.pie(counts, names='Sentiment', values='Count',
                    color='Sentiment', color_discrete_map=colors,
                    title=f'Overall Sentiment — {brand}',
                    template='plotly_dark', hole=0.4)
    fig.update_layout(paper_bgcolor='#0d1117')
    return fig


# Positive Rate
@app.callback(Output('positive-rate', 'figure'), Input('brand-filter', 'value'))
def update_pos_rate(brand):
    df       = load_posts()
    total    = df.groupby('brand').size()
    positive = df[df['vader_sentiment'] == 'Positive'].groupby('brand').size()
    rate     = (positive / total * 100).fillna(0).reset_index()
    rate.columns = ['Brand', 'Positive Rate (%)']
    rate     = rate.sort_values('Positive Rate (%)', ascending=False)
    fig      = px.bar(rate, x='Brand', y='Positive Rate (%)',
                      color='Positive Rate (%)', color_continuous_scale='RdYlGn',
                      title='Positive Sentiment Rate by Brand (%)',
                      template='plotly_dark', text='Positive Rate (%)')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.add_hline(y=50, line_dash='dash', line_color='white', annotation_text='50% line')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Engagement Scatter
@app.callback(Output('engagement-scatter', 'figure'),
              Input('brand-filter', 'value'),
              Input('metric-filter', 'value'))
def update_scatter(brand, metric):
    df  = load_posts()
    if brand != 'All':
        df = df[df['brand'] == brand]
    df  = df.dropna(subset=['upvotes', 'comments_count'])
    fig = px.scatter(df, x='upvotes', y='comments_count',
                     color='brand', symbol='vader_sentiment',
                     hover_data=['title', 'brand', 'vader_sentiment'],
                     title='Upvotes vs Comments by Brand',
                     template='plotly_dark', opacity=0.8)
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Avg Metrics
@app.callback(Output('avg-metrics', 'figure'), Input('metric-filter', 'value'))
def update_avg(metric):
    df   = load_posts()
    avg  = df.groupby('brand')[metric].mean().reset_index()
    avg.columns = ['Brand', 'Average']
    avg  = avg.sort_values('Average', ascending=False)
    fig  = px.bar(avg, x='Average', y='Brand', orientation='h',
                  title=f'Avg {metric} by Brand', template='plotly_dark',
                  color='Average', color_continuous_scale='Blues', text='Average')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Timeline
@app.callback(Output('timeline', 'figure'), Input('brand-filter', 'value'))
def update_timeline(brand):
    df       = load_posts()
    if brand != 'All':
        df   = df[df['brand'] == brand]
    df       = df.dropna(subset=['month'])
    timeline = df.groupby(['month', 'brand']).size().reset_index(name='Posts')
    fig      = px.line(timeline, x='month', y='Posts', color='brand',
                       title='Posts Over Time', template='plotly_dark', markers=True)
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Comment Sentiment
@app.callback(Output('comment-sentiment', 'figure'), Input('brand-filter', 'value'))
def update_comment_sentiment(brand):
    df      = load_comments()
    if brand != 'All':
        df  = df[df['brand'] == brand]
    summary = df.groupby(['brand', 'vader_sentiment']).size().reset_index(name='Count')
    fig     = px.bar(summary, x='brand', y='Count', color='vader_sentiment',
                     color_discrete_map=colors, barmode='group',
                     title='Comment Sentiment by Brand', template='plotly_dark')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Comment Volume
@app.callback(Output('comment-volume', 'figure'), Input('brand-filter', 'value'))
def update_comment_volume(brand):
    df    = load_comments()
    vol   = df.groupby('brand').size().reset_index(name='Comments')
    vol   = vol.sort_values('Comments', ascending=False)
    fig   = px.bar(vol, x='brand', y='Comments',
                   title='Total Comments by Brand', template='plotly_dark',
                   color='Comments', color_continuous_scale='Viridis', text='Comments')
    fig.update_traces(textposition='outside')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig


# Comment Words
@app.callback(Output('comment-words', 'figure'), Input('brand-filter', 'value'))
def update_comment_words(brand):
    df      = load_comments()
    if brand != 'All':
        df  = df[df['brand'] == brand]
    words   = get_top_words(df['comment_text'], 20)
    if not words:
        return go.Figure()
    wdf     = pd.DataFrame(words, columns=['Word', 'Count'])
    fig     = px.bar(wdf, x='Count', y='Word', orientation='h',
                     title=f'Top Words in Comments — {brand}',
                     template='plotly_dark', color='Count',
                     color_continuous_scale='Blues')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f', height=600)
    return fig


# Word Analysis
@app.callback(
    Output('word-pos', 'figure'),
    Output('word-neg', 'figure'),
    Input('word-brand-filter', 'value')
)
def update_words(brand):
    if not brand:
        return go.Figure(), go.Figure()

    df      = load_posts()
    df_com  = load_comments()
    dff     = df[df['brand'] == brand]
    dfc     = df_com[df_com['brand'] == brand]

    pos_text = pd.concat([
        dff[dff['vader_sentiment'] == 'Positive']['content'],
        dfc[dfc['vader_sentiment'] == 'Positive']['comment_text']
    ])
    neg_text = pd.concat([
        dff[dff['vader_sentiment'] == 'Negative']['content'],
        dfc[dfc['vader_sentiment'] == 'Negative']['comment_text']
    ])

    pos_words = get_top_words(pos_text, 12)
    neg_words = get_top_words(neg_text, 12)

    if pos_words:
        pw_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
        fig1  = px.bar(pw_df, x='Count', y='Word', orientation='h',
                       title=f'{brand} — Top Words in POSITIVE Posts + Comments',
                       template='plotly_dark', color='Count',
                       color_continuous_scale='Greens')
    else:
        fig1 = go.Figure()
        fig1.update_layout(title=f'{brand} — No Positive Content')

    if neg_words:
        nw_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
        fig2  = px.bar(nw_df, x='Count', y='Word', orientation='h',
                       title=f'{brand} — Top Words in NEGATIVE Posts + Comments',
                       template='plotly_dark', color='Count',
                       color_continuous_scale='Reds')
    else:
        fig2 = go.Figure()
        fig2.update_layout(title=f'{brand} — No Negative Content')

    fig1.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    fig2.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
    return fig1, fig2


# Posts Table
@app.callback(Output('posts-table', 'figure'), Input('brand-filter', 'value'))
def update_posts_table(brand):
    df   = load_posts()
    if brand != 'All':
        df = df[df['brand'] == brand]
    dff  = df[['brand', 'title', 'vader_sentiment', 'upvotes',
               'comments_count', 'subreddit', 'posted_date']].copy()
    dff['title']       = dff['title'].astype(str).str[:80]
    dff['posted_date'] = dff['posted_date'].astype(str).str[:10]

    color_map  = {'Positive': '#2ecc71', 'Negative': '#e74c3c',
                  'Neutral': '#95a5a6', 'N/A': '#3498db'}
    sent_colors = [color_map.get(s, 'white') for s in dff['vader_sentiment']]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dff.columns),
                    fill_color='#2c3e50',
                    font=dict(color='white', size=12), align='left'),
        cells=dict(values=[dff[c] for c in dff.columns],
                   fill_color=['#1a252f'] * len(dff.columns),
                   font=dict(color=['white', 'white', sent_colors,
                                    'white', 'white', 'white', 'white'], size=11),
                   align='left', height=28)
    )])
    fig.update_layout(title=f'Posts — {brand}', template='plotly_dark',
                      paper_bgcolor='#0d1117', height=600)
    return fig


# Comments Table
@app.callback(Output('comments-table', 'figure'), Input('brand-filter', 'value'))
def update_comments_table(brand):
    df   = load_comments()
    if brand != 'All':
        df = df[df['brand'] == brand]
    dff  = df[['brand', 'comment_text', 'vader_sentiment',
               'comment_upvotes', 'comment_depth']].head(100).copy()
    dff['comment_text'] = dff['comment_text'].astype(str).str[:100]

    color_map   = {'Positive': '#2ecc71', 'Negative': '#e74c3c',
                   'Neutral': '#95a5a6', 'N/A': '#3498db'}
    sent_colors = [color_map.get(s, 'white') for s in dff['vader_sentiment']]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dff.columns),
                    fill_color='#2c3e50',
                    font=dict(color='white', size=12), align='left'),
        cells=dict(values=[dff[c] for c in dff.columns],
                   fill_color=['#1a252f'] * len(dff.columns),
                   font=dict(color=['white', 'white', sent_colors,
                                    'white', 'white'], size=11),
                   align='left', height=28)
    )])
    fig.update_layout(title=f'Comments — {brand} (showing first 100)',
                      template='plotly_dark', paper_bgcolor='#0d1117', height=600)
    return fig


# Brands Table
@app.callback(Output('brands-table', 'children'), Input('refresh-btn', 'n_clicks'))
def update_brands_table(n):
    try:
        brands = load_brands()
        rows   = []
        for brand in brands:
            df         = load_posts()
            brand_df   = df[df['brand'] == brand]
            post_count = len(brand_df)
            df_com     = load_comments()
            com_count  = len(df_com[df_com['brand'] == brand])
            rows.append(html.Tr([
                html.Td(brand,      style={'color': 'white', 'padding': '10px'}),
                html.Td(post_count, style={'color': '#3498db', 'padding': '10px'}),
                html.Td(com_count,  style={'color': '#9b59b6', 'padding': '10px'}),
                html.Td('✅ Active', style={'color': '#2ecc71', 'padding': '10px'}),
            ]))

        return dbc.Table(
            [html.Thead(html.Tr([
                html.Th('Brand',    style={'color': 'white'}),
                html.Th('Posts',    style={'color': 'white'}),
                html.Th('Comments', style={'color': 'white'}),
                html.Th('Status',   style={'color': 'white'}),
            ]))] + [html.Tbody(rows)],
            bordered=True, dark=True, hover=True, responsive=True
        )
    except Exception as e:
        return html.P(f'Loading brands... {str(e)}',
                     style={'color': 'white'})
        brand_df  = df[df['brand'] == brand]
        post_count = len(brand_df)
        df_com    = load_comments()
        com_count  = len(df_com[df_com['brand'] == brand])
        rows.append(html.Tr([
            html.Td(brand,      style={'color': 'white', 'padding': '10px'}),
            html.Td(post_count, style={'color': '#3498db', 'padding': '10px'}),
            html.Td(com_count,  style={'color': '#9b59b6', 'padding': '10px'}),
            html.Td('✅ Active', style={'color': '#2ecc71', 'padding': '10px'}),
        ]))

    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th('Brand',    style={'color': 'white'}),
            html.Th('Posts',    style={'color': 'white'}),
            html.Th('Comments', style={'color': 'white'}),
            html.Th('Status',   style={'color': 'white'}),
        ]))] + [html.Tbody(rows)],
        bordered=True, dark=True, hover=True, responsive=True
    )


# Add Brand
@app.callback(
    Output('add-brand-output', 'children'),
    Input('add-brand-btn', 'n_clicks'),
    State('new-brand-name', 'value'),
    State('new-brand-url',  'value'),
    prevent_initial_call=True
)
def add_brand(n_clicks, name, url):
    if not name or not url:
        return '⚠️ Please enter both brand name and URL!'
    return f'✅ Brand "{name}" added! Run the scraper to collect data.'


# Scrape History Table
@app.callback(Output('history-table', 'figure'), Input('interval', 'n_intervals'))
def update_history(n):
    df  = load_scrape_history()
    fig = go.Figure(data=[go.Table(
        header=dict(values=['ID', 'Brand', 'Posts', 'Comments', 'Status', 'Error', 'Scraped At'],
                    fill_color='#2c3e50',
                    font=dict(color='white', size=12), align='left'),
        cells=dict(values=[df['id'], df['brand'], df['posts_scraped'],
                           df['comments_scraped'], df['status'],
                           df['error_message'].fillna('None'),
                           df['scraped_at'].astype(str).str[:19]],
                   fill_color=['#1a252f'] * 7,
                   font=dict(color='white', size=11),
                   align='left', height=28)
    )])
    fig.update_layout(title='Scrape History', template='plotly_dark',
                      paper_bgcolor='#0d1117', height=400)
    return fig


# Predict Post
@app.callback(
    Output('prediction-output', 'children'),
    Output('sentiment-gauge',   'figure'),
    Input('predict-btn', 'n_clicks'),
    State('pred-title',   'value'),
    State('pred-content', 'value'),
    State('pred-brand',   'value'),
    prevent_initial_call=True
)
def predict_post(n_clicks, title, content, brand):
    if not title and not content:
        return 'Please enter a title and content!', go.Figure()

    analyzer  = SentimentIntensityAnalyzer()
    full_text = f'{title or ""} {content or ""}'
    scores    = analyzer.polarity_scores(full_text)
    compound  = scores['compound']

    if compound >= 0.05:
        sentiment   = 'POSITIVE'
        color       = '#2ecc71'
        emoji       = '🟢'
        upvote_pred = 'HIGH' if compound > 0.3 else 'MODERATE'
    elif compound <= -0.05:
        sentiment   = 'NEGATIVE'
        color       = '#e74c3c'
        emoji       = '🔴'
        upvote_pred = 'LOW'
    else:
        sentiment   = 'NEUTRAL'
        color       = '#95a5a6'
        emoji       = '⚪'
        upvote_pred = 'MODERATE'

    output = [
        html.P(f'{emoji} Sentiment: {sentiment}',
               style={'color': color, 'fontSize': '22px'}),
        html.P(f'📈 Predicted Upvotes: {upvote_pred}',
               style={'color': '#f39c12'}),
        html.P(f'🏷️ Brand: {brand}',
               style={'color': 'white'}),
        html.P(f'📊 Compound Score: {compound:.3f}',
               style={'color': '#3498db'}),
        html.P(f'😊 Positive: {scores["pos"]:.2f}  '
               f'😐 Neutral: {scores["neu"]:.2f}  '
               f'😞 Negative: {scores["neg"]:.2f}',
               style={'color': '#95a5a6'})
    ]

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=round((compound + 1) * 50, 1),
        title={'text': 'Sentiment Score', 'font': {'color': 'white'}},
        delta={'reference': 50},
        gauge={
            'axis':  {'range': [0, 100], 'tickcolor': 'white'},
            'bar':   {'color': color},
            'steps': [
                {'range': [0,  40], 'color': '#2c1810'},
                {'range': [40, 60], 'color': '#2c2c1a'},
                {'range': [60, 100], 'color': '#1a2c1a'}
            ],
            'threshold': {
                'line':      {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value':     50
            }
        }
    ))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0d1117',
                      font={'color': 'white'}, height=350)
    return output, fig

# Search Any Brand Callback
@app.callback(
    Output('search-brand-status', 'children'),
    Output('search-brand-results', 'children'),
    Input('search-brand-btn', 'n_clicks'),
    State('search-brand-input', 'value'),
    prevent_initial_call=True
)
def search_any_brand(n_clicks, brand_name):
    if not brand_name or not brand_name.strip():
        return '⚠️ Please enter a brand name.', ''

    brand_name = brand_name.strip().title()

    try:
        from scraper import scrape_brand_on_demand

        # Check if cached
        with engine.connect() as conn:
            cached = conn.execute(text('''
                SELECT COUNT(*) FROM scrape_history
                WHERE brand = :brand
                AND scraped_at > NOW() - INTERVAL '24 hours'
                AND status = 'success'
            '''), {'brand': brand_name}).scalar()

        if cached > 0:
            status_msg = f'✅ Showing cached results for "{brand_name}" (scraped in last 24hrs)'
        else:
            posts, comments, error = scrape_brand_on_demand(brand_name, max_posts=25)
            if error:
                return f'❌ Error: {error}', ''
            status_msg = f'✅ Done! Found {posts} posts and {comments} comments for "{brand_name}"'

        # Load results from DB
        df = load_posts()
        df_com = load_comments()
        brand_df = df[df['brand'] == brand_name]
        brand_com = df_com[df_com['brand'] == brand_name]

        if len(brand_df) == 0:
            return f'⚠️ No data found for "{brand_name}" on Reddit.', ''

        # Build results
        total = len(brand_df)
        pos = len(brand_df[brand_df['vader_sentiment'] == 'Positive'])
        neg = len(brand_df[brand_df['vader_sentiment'] == 'Negative'])
        neu = len(brand_df[brand_df['vader_sentiment'] == 'Neutral'])
        pos_pct = round(pos / total * 100, 1)
        neg_pct = round(neg / total * 100, 1)

        card_style = {'textAlign': 'center', 'padding': '15px',
                      'borderRadius': '10px', 'background': '#1a252f'}

        results = html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H3(total, style={'color': '#3498db'}),
                    html.P('Posts Found')
                ])], style=card_style), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H3(len(brand_com), style={'color': '#9b59b6'}),
                    html.P('Comments')
                ])], style=card_style), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H3(f'{pos_pct}%', style={'color': '#2ecc71'}),
                    html.P('Positive')
                ])], style=card_style), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H3(f'{neg_pct}%', style={'color': '#e74c3c'}),
                    html.P('Negative')
                ])], style=card_style), width=2),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H3(round(brand_df['upvotes'].mean(), 1),
                            style={'color': '#f39c12'}),
                    html.P('Avg Upvotes')
                ])], style=card_style), width=2),
            ], className='mb-4'),

            dbc.Row([
                dbc.Col(dcc.Graph(
                    figure=px.pie(
                        values=[pos, neg, neu],
                        names=['Positive', 'Negative', 'Neutral'],
                        color_discrete_map=colors,
                        title=f'Sentiment — {brand_name}',
                        template='plotly_dark',
                        hole=0.4
                    ).update_layout(paper_bgcolor='#0d1117')
                ), width=5),
                dbc.Col(dcc.Graph(
                    figure=px.bar(
                        brand_df.groupby('vader_sentiment').size().reset_index(name='Count'),
                        x='vader_sentiment', y='Count',
                        color='vader_sentiment',
                        color_discrete_map=colors,
                        title=f'Post Breakdown — {brand_name}',
                        template='plotly_dark'
                    ).update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#1a252f')
                ), width=7),
            ]),

            html.H5(f'Top Posts for {brand_name}',
                    style={'color': 'white', 'marginTop': '20px'}),
            dbc.Table(
                [html.Thead(html.Tr([
                    html.Th('Title', style={'color': 'white'}),
                    html.Th('Sentiment', style={'color': 'white'}),
                    html.Th('Upvotes', style={'color': 'white'}),
                    html.Th('Subreddit', style={'color': 'white'}),
                ]))] + [html.Tbody([
                    html.Tr([
                        html.Td(str(row['title'])[:80],
                                style={'color': 'white', 'fontSize': '12px'}),
                        html.Td(row['vader_sentiment'],
                                style={'color': colors.get(row['vader_sentiment'], 'white')}),
                        html.Td(row['upvotes'], style={'color': 'white'}),
                        html.Td(row['subreddit'], style={'color': '#95a5a6'}),
                    ]) for _, row in brand_df.head(15).iterrows()
                ])],
                bordered=True, dark=True, hover=True,
                responsive=True, size='sm'
            ),
        ])

        return status_msg, results

    except Exception as e:
        return f'❌ Unexpected error: {str(e)}', ''
# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
