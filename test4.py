# twitter_agent_full_gemini.py
"""
Single-file Twitter agent (mock by default) — LangChain (Gemini) + Tweepy + persistent DB + moderation + human-review CLI.
Converted LLM backend from OpenAI -> Google Gemini while keeping all other libraries and logic unchanged.
Important: set GEMINI_API_KEY in your .env (replaces OPENAI_API_KEY usage).
Required packages (same as before, plus langchain-google-genai if you plan to use LangChain's Gemini wrapper):
    pip install tweepy langchain-openai langchain-core langchain-google-genai python-dotenv sqlalchemy tenacity
Note: LangChain + Gemini wrapper versions must be compatible; if you prefer the direct google.generativeai client, swap accordingly.
"""
import os
import time
import sys
from typing import Optional, List, Dict
from datetime import datetime, timezone
from dotenv import load_dotenv

# LangChain Gemini wrapper (replace OpenAI LLM with Gemini LLM)
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Tweepy (Twitter API v2)
import tweepy

# SQLAlchemy 2.x
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# retries
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Keep Twitter/other env names same as original
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")  # optional; used to resolve own id

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("1", "true", "yes")
PREVIEW_ONLY = os.getenv("PREVIEW_ONLY", "true").lower() in ("1", "true", "yes")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "20"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///agent.db")

# immediate blacklist
BLACKLIST_KEYWORDS = {"suicide", "bomb", "terror", "explosive", "kill", "attack"}

# ---------------- DB SETUP ----------------
Base = declarative_base()


class Mention(Base):
    __tablename__ = "mentions"
    mention_id = Column(String, primary_key=True, index=True)
    author_id = Column(String, index=True)
    text = Column(Text)
    reply = Column(Text, nullable=True)
    status = Column(String)  # new/pending/approved/posted/blocked/rejected/failed
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    moderated_at = Column(DateTime(timezone=True), nullable=True)
    replied_at = Column(DateTime(timezone=True), nullable=True)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base.metadata.create_all(engine)

# ---------------- LLM (Gemini via LangChain) SETUP ----------------
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in environment for LLM generation (replaces OPENAI_API_KEY)")

# create LangChain Gemini LLM instance (model_name adjustable)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.6)

# before
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.6)

# replace with a supported model alias (examples)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0.6)
# or
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=GEMINI_API_KEY, temperature=0.6)
# or for cheapest/small-footprint:
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=GEMINI_API_KEY, temperature=0.6)


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

# ---------------- Helpers ----------------
def contains_blacklisted(text: str) -> bool:
    txt = (text or "").lower()
    return any(k in txt for k in BLACKLIST_KEYWORDS)


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(4), retry=retry_if_exception_type(Exception))
def call_llm(prompt_text: str) -> str:
    """
    Invoke LangChain Gemini wrapper and normalize response to a string.
    Handles common response shapes returned by the Gemini wrapper.
    """
    # Try invoke (most LangChain wrappers expose invoke)
    resp = None
    try:
        resp = llm.invoke(prompt_text)
    except TypeError:
        # Some versions expect a dict or different call, try generate if available
        try:
            resp = llm.generate(prompt_text)
        except Exception:
            resp = None

    # Normalize to text
    out = ""
    if resp is None:
        raise RuntimeError("LLM returned no response object")

    # resp might be a string
    if isinstance(resp, str):
        out = resp.strip()
    else:
        # Try common properties for LangChain Google GenAI wrapper
        # 1) resp.content -> list with .text
        content = getattr(resp, "content", None)
        if content:
            try:
                # content may be list-like
                first = content[0]
                out = getattr(first, "text", None) or getattr(first, "content", None) or str(first)
                out = str(out).strip()
            except Exception:
                out = str(content).strip()
        else:
            # 2) resp.generations (LangChain common)
            gens = getattr(resp, "generations", None)
            if gens:
                try:
                    out = gens[0][0].text.strip()
                except Exception:
                    out = str(gens).strip()
            else:
                # 3) try resp.text or resp.output_text
                out = getattr(resp, "text", None) or getattr(resp, "output_text", None) or getattr(resp, "result", None) or str(resp)
                out = str(out).strip()

    if len(out) > 270:
        out = out[:267] + "..."
    return out


def moderate_text(text: str) -> str:
    t = (text or "").lower()
    if contains_blacklisted(t):
        return "unsafe"
    if "?" in t or any(w in t for w in ("how ", "what ", "why ", "when ", "where ", "who ")):
        return "safe"
    return "review"


# ---------------- DB operations ----------------
def save_new_mention(mention_id: str, author_id: str, text: str) -> bool:
    session = SessionLocal()
    try:
        existing = session.get(Mention, mention_id)
        if existing:
            return False
        m = Mention(mention_id=mention_id, author_id=author_id, text=text, status="new")
        session.add(m)
        session.commit()
        return True
    finally:
        session.close()


def process_mention_db(mention_id: str):
    session = SessionLocal()
    try:
        m = session.get(Mention, mention_id)
        if not m or m.status != "new":
            return
        mod = moderate_text(m.text)
        m.moderated_at = datetime.now(timezone.utc)
        if mod == "unsafe":
            m.status = "blocked"
            session.commit()
            print(f"Mention {mention_id} blocked by moderation.")
            return

        prompt_text = PROMPT.format(mention_text=m.text, context="")
        try:
            gen = call_llm(prompt_text)
            m.reply = gen
            m.status = "pending"
            session.commit()
            print(f"Mention {mention_id} generated and set to pending.")
            return
        except RetryError as re:
            last_exc = None
            try:
                last_exc = re.last_attempt.exception()
            except Exception:
                last_exc = re
            m.status = "failed"
            session.commit()
            print(f"LLM RetryError for mention {mention_id}. Last exception: {type(last_exc).__name__}: {last_exc}")
            return
        except Exception as e:
            m.status = "failed"
            session.commit()
            print(f"LLM failed for mention {mention_id}: {type(e).__name__}: {e}")
            return
    finally:
        session.close()


# ---------------- Twitter (Tweepy) helpers ----------------
def make_twitter_client() -> tweepy.Client:
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


def get_own_user_id(client: tweepy.Client) -> str:
    if TWITTER_USERNAME:
        me = client.get_user(username=TWITTER_USERNAME)
    else:
        me = client.get_me()
    return str(me.data.id)


def fetch_mentions(client: tweepy.Client, user_id: str, since_id: Optional[str]) -> List[Dict]:
    mentions = []
    try:
        if since_id:
            resp = client.get_users_mentions(id=user_id, since_id=since_id, max_results=50, tweet_fields=["author_id", "text", "created_at", "id"])
        else:
            resp = client.get_users_mentions(id=user_id, max_results=50, tweet_fields=["author_id", "text", "created_at", "id"])
    except TypeError:
        resp = client.get_users_mentions(user_id, max_results=50, tweet_fields=["author_id", "text", "created_at", "id"])
    if resp and getattr(resp, "data", None):
        mentions = [dict(id=str(t.id), text=t.text, author_id=str(t.author_id), created_at=str(t.created_at)) for t in resp.data]
        mentions = list(reversed(mentions))
    return mentions


def post_reply(client: tweepy.Client, text: str, in_reply_to_tweet_id: str) -> Optional[str]:
    resp = client.create_tweet(text=text, in_reply_to_tweet_id=in_reply_to_tweet_id)
    data = getattr(resp, "data", None)
    if isinstance(data, dict) and "id" in data:
        return str(data["id"])
    try:
        return str(data.id)
    except Exception:
        return None


# ---------------- Simulation / Posting ----------------
def simulate_posting(mention_id: str):
    session = SessionLocal()
    try:
        m = session.get(Mention, mention_id)
        if not m:
            print("Not found")
            return
        if m.status != "approved":
            print("Only approved mentions can be posted/simulated.")
            return
        m.status = "posted"
        m.replied_at = datetime.now(timezone.utc)
        session.commit()
        print(f"[SIMULATED POST] Reply to {mention_id}:\n{m.reply}")
    finally:
        session.close()


def publish_reply_to_twitter(client: tweepy.Client, mention_id: str):
    session = SessionLocal()
    try:
        m = session.get(Mention, mention_id)
        if not m or m.status != "approved":
            print("Not approved or not found.")
            return
        if PREVIEW_ONLY:
            print("PREVIEW_ONLY set — not posting. Reply would be:\n", m.reply)
            return
        tweet_id = post_reply(client, text=m.reply, in_reply_to_tweet_id=mention_id)
        if tweet_id:
            m.status = "posted"
            m.replied_at = datetime.now(timezone.utc)
            session.commit()
            print(f"Posted reply id: {tweet_id}")
        else:
            m.status = "failed"
            session.commit()
            print("Post failed (no tweet id returned).")
    finally:
        session.close()


# ---------------- Admin CLI ----------------
def list_pending():
    session = SessionLocal()
    try:
        rows = session.query(Mention).filter(Mention.status == "pending").all()
        if not rows:
            print("No pending mentions.")
            return
        for r in rows:
            print("---")
            print(f"mention_id: {r.mention_id}")
            print(f"author_id: {r.author_id}")
            print(f"text: {r.text}")
            print(f"generated reply: {r.reply}")
            print(f"created_at: {r.created_at}")
    finally:
        session.close()


def approve(mention_id: str):
    session = SessionLocal()
    try:
        m = session.get(Mention, mention_id)
        if not m:
            print("Not found")
            return
        m.status = "approved"
        session.commit()
        print(f"Mention {mention_id} approved.")
    finally:
        session.close()


def reject(mention_id: str):
    session = SessionLocal()
    try:
        m = session.get(Mention, mention_id)
        if not m:
            print("Not found")
            return
        m.status = "rejected"
        session.commit()
        print(f"Mention {mention_id} rejected.")
    finally:
        session.close()


# ---------------- Main agent flow ----------------
STATE_FILE = os.getenv("STATE_FILE", "agent_state.json")


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"last_mention_id": None}
    import json
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: dict):
    import json
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def run_agent_once_live(client: tweepy.Client, state: dict):
    user_id = get_own_user_id(client)
    since_id = state.get("last_mention_id")
    mentions = fetch_mentions(client, user_id, since_id)
    if not mentions:
        print("No new mentions.")
        return
    for m in mentions:
        mention_id = str(m["id"])
        mention_text = m["text"]
        print(f"Processing mention {mention_id}: {mention_text}")
        saved = save_new_mention(mention_id, m.get("author_id", ""), mention_text)
        if not saved:
            print("Already processed; skipping.")
            state["last_mention_id"] = mention_id
            save_state(state)
            continue
        process_mention_db(mention_id)
        # leave posting for admin approve/publish (or add auto-approve logic here)
        state["last_mention_id"] = mention_id
        save_state(state)


def run_agent_once_mock(state: dict):
    demo_mention = {"id": "999999", "text": "@you How do I reset my password?", "author_id": "1111"}
    saved = save_new_mention(demo_mention["id"], demo_mention["author_id"], demo_mention["text"])
    if saved:
        process_mention_db(demo_mention["id"])
    else:
        print("No new mentions.")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--admin":
        print("Admin CLI: list | approve <id> | reject <id> | publish <id> | simulate <id> | exit")
        while True:
            try:
                cmd = input("cmd> ").strip().split()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            op = cmd[0].lower()
            if op in ("exit", "quit"):
                break
            if op == "list":
                list_pending(); continue
            if op == "approve" and len(cmd) > 1:
                approve(cmd[1]); continue
            if op == "reject" and len(cmd) > 1:
                reject(cmd[1]); continue
            if op == "simulate" and len(cmd) > 1:
                simulate_posting(cmd[1]); continue
            if op == "publish" and len(cmd) > 1:
                if MOCK_MODE:
                    print("MOCK_MODE=true — simulate with 'simulate' instead.")
                else:
                    try:
                        client = make_twitter_client()
                        publish_reply_to_twitter(client, cmd[1])
                    except Exception as e:
                        print("Publish failed:", e)
                continue
            print("Unknown command")
    else:
        print(f"Starting Twitter Gemini LangChain agent. MOCK_MODE={MOCK_MODE} PREVIEW_ONLY={PREVIEW_ONLY}")
        state = load_state()
        try:
            while True:
                try:
                    if MOCK_MODE:
                        run_agent_once_mock(state)
                    else:
                        client = make_twitter_client()
                        run_agent_once_live(client, state)
                except Exception as e:
                    print("Agent run error:", type(e).__name__, e)
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("Agent stopped by user.")
