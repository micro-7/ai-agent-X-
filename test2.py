# twitter_agent_gemini.py
"""
Converted from OpenAI to Google Gemini (direct google.generativeai client).
All other logic, DB schema, moderation, admin CLI, and mock-mode behavior preserved.
Install (inside your activated venv):
pip install google-generativeai python-dotenv sqlalchemy tenacity
.env should contain:
GEMINI_API_KEY=your_gemini_key_here
MOCK_MODE=true
PREVIEW_ONLY=true
POLL_INTERVAL=20
DATABASE_URL=sqlite:///agent.db  # optional
"""

import os
import time
import sys
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timezone

# google generative AI client
import google.generativeai as genai

# SQLAlchemy (2.0 style)
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# retries
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError

load_dotenv()

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("1", "true", "yes")
PREVIEW_ONLY = os.getenv("PREVIEW_ONLY", "true").lower() in ("1", "true", "yes")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "20"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///agent.db")

# Basic immediate blacklist
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

# ---------------- LLM (Gemini) SETUP ----------------
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in environment for LLM generation")

# configure client
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- Prompt (string template replacing PromptTemplate) ----------------
PROMPT_TEMPLATE = """You are a helpful, concise, human-sounding assistant that replies to Twitter mentions.
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


def build_prompt(mention_text: str, context: Optional[str] = "") -> str:
    return PROMPT_TEMPLATE.format(mention_text=mention_text, context=context or "")


def contains_blacklisted(text: str) -> bool:
    txt = (text or "").lower()
    return any(k in txt for k in BLACKLIST_KEYWORDS)


# ---------------- Gemini call WITH RETRY ----------------
@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(4), retry=retry_if_exception_type(Exception))
def call_gemini(prompt_text: str) -> str:
    """
    Try common google.generativeai call shapes for compatibility across client versions.
    Normalize and return text trimmed to 270 chars.
    """
    resp = None

    # 1) genai.generate_text (some versions)
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model="gemini-1.5-flash", text=prompt_text)
            out = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
            out = str(out).strip()
            if out:
                return out if len(out) <= 270 else out[:267] + "..."
    except Exception:
        resp = None

    # 2) genai.respond (older examples)
    try:
        if hasattr(genai, "respond"):
            resp = genai.respond(model="gemini-1.5-flash", input=prompt_text)
            out = getattr(resp, "output_text", None)
            if not out:
                try:
                    out = resp.candidates[0].output[0].content
                except Exception:
                    out = None
            if out is None:
                out = str(resp)
            out = str(out).strip()
            return out if len(out) <= 270 else out[:267] + "..."
    except Exception:
        resp = None

    # 3) chat style (some SDKs)
    try:
        if hasattr(genai, "chat"):
            resp = genai.chat.completions.create(model="gemini-1.5-flash", messages=[{"role": "user", "content": prompt_text}])
            try:
                out = resp.choices[0].message.content
            except Exception:
                out = None
            if out is None:
                out = str(resp)
            out = str(out).strip()
            return out if len(out) <= 270 else out[:267] + "..."
    except Exception:
        resp = None

    # fallback stringify
    if resp is not None:
        out = str(resp).strip()
        return out if len(out) <= 270 else out[:267] + "..."

    # nothing worked; raise to trigger retry
    raise RuntimeError("No compatible google.generativeai response shape detected")


# ---------------- Moderation stub ----------------
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
        if not m:
            return
        if m.status not in ("new",):
            return

        # moderation
        mod = moderate_text(m.text)
        m.moderated_at = datetime.now(timezone.utc)
        if mod == "unsafe":
            m.status = "blocked"
            session.commit()
            print(f"Mention {mention_id} blocked by moderation.")
            return

        prompt_text = build_prompt(m.text, context="")
        try:
            gen = call_gemini(prompt_text)
            m.reply = gen
            # keep human-in-the-loop even for safe items
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


# ---------------- Admin CLI helpers ----------------
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
        print(f"Mention {mention_id} approved. Use 'simulate {mention_id}' to simulate posting.")
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


# ---------------- Main loop (mock-friendly) ----------------
def run_agent_once():
    if MOCK_MODE:
        demo_mention = {"id": "999999", "text": "@you How do I reset my password?", "author_id": "1111"}
        saved = save_new_mention(demo_mention["id"], demo_mention["author_id"], demo_mention["text"])
        if saved:
            process_mention_db(demo_mention["id"])
        else:
            print("No new mentions.")
    else:
        print("Non-mock fetching not implemented in this demo run.")


if __name__ == "__main__":
    print("GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY")))
    if len(sys.argv) > 1 and sys.argv[1] == "--admin":
        print("Admin CLI: list | approve <id> | reject <id> | simulate <id> | exit")
        while True:
            try:
                cmd = input("cmd> ").strip().split()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            if cmd[0] in ("exit", "quit"):
                break
            if cmd[0] == "list":
                list_pending()
                continue
            if cmd[0] == "approve" and len(cmd) > 1:
                approve(cmd[1])
                continue
            if cmd[0] == "reject" and len(cmd) > 1:
                reject(cmd[1])
                continue
            if cmd[0] == "simulate" and len(cmd) > 1:
                simulate_posting(cmd[1])
                continue
            print("Unknown command")
    else:
        print(f"Starting agent loop. MOCK_MODE={MOCK_MODE} PREVIEW_ONLY={PREVIEW_ONLY}")
        try:
            while True:
                try:
                    run_agent_once()
                except Exception as e:
                    print("Agent run error:", type(e).__name__, e)
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("Agent stopped by user.")
