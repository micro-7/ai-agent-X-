# twitter_agent.py
import os
import time
import json
from typing import List, Optional
from dotenv import load_dotenv

# LangChain / OpenAI — modern packages
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Tweepy for Twitter API v2 (posting + reading mentions)
import tweepy

# ------------ CONFIG ------------
load_dotenv()  # loads .env into environment

TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")  # optional for some endpoints

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Behavior flags
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "20"))  # seconds between polls
STATE_FILE = os.getenv("STATE_FILE", "agent_state.json")
PREVIEW_ONLY = os.getenv("PREVIEW_ONLY", "true").lower() in ("1", "true", "yes")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")  # no Twitter calls if True

# Basic blacklist / moderation keywords (simple local filter)
BLACKLIST_KEYWORDS = {"suicide", "bomb", "terror", "explosive", "kill", "attack"}  # extend as needed

# ------------ HELPERS ------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_mention_id": None}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def contains_blacklisted(text: str) -> bool:
    txt = text.lower()
    return any(k in txt for k in BLACKLIST_KEYWORDS)

# ------------ LLM SETUP ------------
if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")

# Create an OpenAI LLM instance. model_name and args are adjustable.
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name="gpt-4o-mini")

PROMPT = PromptTemplate.from_template(
    """You are a helpful, concise, human-sounding assistant that replies to Twitter mentions.
Rules:
- Reply in one to three sentences.
- Use a friendly, natural tone. Avoid sounding like a bot.
- If the mention contains a question, answer directly. If it's feedback, thank them and respond succinctly.
- If content is abusive, refuse politely.
- Keep replies <= 280 characters.

Mention: {mention_text}
Context: {context}
Generate the text to post as a reply (no quoting).
"""
)

def generate_reply(mention_text: str, context: Optional[str] = "") -> str:
    prompt_text = PROMPT.format(mention_text=mention_text, context=context)
    resp = llm.invoke(prompt_text)
    reply = resp.strip()
    if len(reply) > 270:
        reply = reply[:267] + "..."
    return reply


# ------------ TWITTER CLIENT (tweepy) ------------
def make_twitter_client():
    if not all([TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
        raise RuntimeError("Twitter API keys/tokens missing in environment.")
    client = tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
        bearer_token=TWITTER_BEARER_TOKEN
    )
    return client

# Get authenticated user id (needed for mentions)
def get_own_user_id(client: tweepy.Client) -> str:
    if os.getenv("TWITTER_USERNAME"):
        me = client.get_user(username=os.getenv("TWITTER_USERNAME"))
    else:
        me = client.get_me()
    return str(me.data.id)

# Fetch mentions since last_mention_id (returns list of tweet dicts)
def fetch_mentions(client: tweepy.Client, user_id: str, since_id: Optional[str]) -> List[dict]:
    mentions = []
    params = {"id": user_id, "max_results": 50}  # tweepy uses 'id' for get_users_mentions positional kw in some versions
    # Use since_id only if provided
    try:
        if since_id:
            resp = client.get_users_mentions(id=user_id, since_id=since_id, max_results=50, tweet_fields=["author_id","text","created_at","id"])
        else:
            resp = client.get_users_mentions(id=user_id, max_results=50, tweet_fields=["author_id","text","created_at","id"])
    except TypeError:
        # Fallback in case of older/newer tweepy signatures
        resp = client.get_users_mentions(user_id, max_results=50, tweet_fields=["author_id","text","created_at","id"])
    if resp and resp.data:
        mentions = [dict(id=str(t.id), text=t.text, author_id=str(t.author_id), created_at=str(t.created_at)) for t in resp.data]
        mentions = list(reversed(mentions))  # process oldest-first
    return mentions

def post_reply(client: tweepy.Client, text: str, in_reply_to_tweet_id: str) -> dict:
    resp = client.create_tweet(text=text, in_reply_to_tweet_id=in_reply_to_tweet_id)
    return resp.data

# ------------ MAIN LOOP ------------
def run_agent():
    state = load_state()
    last_id = state.get("last_mention_id")

    if MOCK_MODE:
        print("MOCK_MODE enabled — no Twitter API calls will be made.")
        demo_mention = {"id": "999999", "text": "@you How do I reset my password?", "author_id": "1111", "created_at": "now"}
        mentions = [demo_mention] if last_id != demo_mention["id"] else []
    else:
        client = make_twitter_client()
        user_id = get_own_user_id(client)
        mentions = fetch_mentions(client, user_id, last_id)

    if not mentions:
        print("No new mentions.")
        return

    for m in mentions:
        mention_id = str(m["id"])
        mention_text = m["text"]
        print(f"Processing mention {mention_id}: {mention_text}")

        # Moderation / blacklist
        if contains_blacklisted(mention_text):
            fallback = "I can't assist with that request. If this is an emergency, please contact local authorities."
            reply_text = fallback
            print(f"Blacklisted content detected; replying with fallback.")
        else:
            try:
                reply_text = generate_reply(mention_text, context="")
            except Exception as e:
                print(f"LLM generation failed: {e}")
                reply_text = "Sorry, something went wrong while composing a reply. Please try again later."

        # Preview or post
        if PREVIEW_ONLY or MOCK_MODE:
            print("---- PREVIEW ----")
            print(f"Would reply to {mention_id} with:\n{reply_text}")
            print("-----------------")
        else:
            try:
                r = post_reply(client, text=reply_text, in_reply_to_tweet_id=mention_id)
                print(f"Posted reply id: {r['id'] if isinstance(r, dict) and 'id' in r else r}")
            except Exception as e:
                print(f"Failed to post reply: {e}")

        # update last processed id
        last_id = mention_id
        state["last_mention_id"] = last_id
        save_state(state)

# ------------ ENTRY POINT ------------
if __name__ == "__main__":
    print("Starting Twitter LangChain agent. Preview only:", PREVIEW_ONLY, "Mock mode:", MOCK_MODE)
    try:
        while True:
            try:
                run_agent()
            except Exception as e:
                print("Agent run error:", e)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("Agent stopped by user.")
