          ┌──────────────────────┐
          │    Twitter API       │
          │ (mentions & replies) │
          └──────────┬───────────┘
                     │ fetch_mentions()
                     ▼
         ┌────────────────────────────┐
         │   Agent Main Loop          │
         │  (mock/live mode)          │
         └───────┬────────────────────┘
                 │ save_new_mention()
                 ▼
       ┌───────────────────────┐
       │  SQLite Database       │
       │  (SQLAlchemy ORM)      │
       └──────┬────────────────┘
              │ process_mention_db()
              ▼
      ┌───────────────────────────┐
      │  Moderation Filter         │
      │ (keyword blacklist, rules) │
      └──────────┬────────────────┘
                 │ safe?
                 ▼
     ┌─────────────────────────────┐
     │  Gemini LLM via LangChain   │
     │  (ChatGoogleGenerativeAI)   │
     └──────────┬──────────────────┘
                │ call_llm()
                ▼
       ┌───────────────────────────┐
       │   Reply stored as         │
       │   status = "pending"      │
       └──────────┬────────────────┘
                  │ Admin reviews
                  ▼
   ┌──────────────────────────────────────┐
   │       Admin CLI (--admin mode)       │
   │ list | approve | reject | simulate   │
   └──────────┬───────────────────────────┘
              │ publish_reply_to_twitter()
              ▼
         ┌──────────────────────┐
         │     Twitter API      │
         │ (post reply tweet)   │
         └──────────────────────┘



1. Fetch Twitter mention  (or create fake one in mock mode)
2. Save it in database as "new"
3. Check if message is safe
4. Ask Gemini to generate reply → mark as "pending"
5. Admin reviews:
       approve → post reply
       reject  → discard
       simulate → test without posting
6. If approved → send reply via Twitter API


In mock mode, it doesn’t touch Twitter — it just simulates everything for testing.
In live mode, it uses actual Twitter API keys to fetch mentions and post replies.

Main tech used:

LangChain + Gemini → generate the reply

Tweepy → talk to Twitter (fetch mentions + post replies)

SQLAlchemy + SQLite → store mentions, replies, statuses

Tenacity → retry LLM failures

dotenv → load API keys

Main functions (simple meaning):

save_new_mention() → store mention in DB

process_mention_db() → moderate + generate reply

call_llm() → call Gemini to create the reply

list_pending() → show unapproved replies

approve() / reject() → review workflow

simulate_posting() → pretend to post (mock mode)

publish_reply_to_twitter() → actually post the reply

run_agent_once_mock() → fake test mention

run_agent_once_live() → fetch real mentions


Twitter → Agent → Database → Admin → Twitter


