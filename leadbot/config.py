import os
from dataclasses import dataclass
from pathlib import Path


def _ids(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.replace(" ", "").split(",") if x)


@dataclass(frozen=True)
class Config:
    bot_token: str
    allowed_users: frozenset[int]
    # Chat that receives the daily leads: a group with the team, or one person. Defaults to the first allowed user.
    leads_chat_id: int
    gemini_api_key: str
    # Tried in order: the first one that answers wins.
    gemini_models: tuple[str, ...]
    # Apify: Google Maps search (and optionally Instagram profiles when Instagram blocks direct requests).
    apify_token: str
    # Budget for the free Apify plan ($5 a month): places scraped per day, per search query,
    # and the monthly spend at which searching stops.
    places_per_day: int
    places_per_query: int
    apify_monthly_budget: float
    # Read Instagram profiles through Apify when Instagram blocks us. Costs extra, so off by default.
    apify_instagram: bool
    # Instagram Graph API (Business Discovery): a long-lived token and the id of your Instagram business account.
    meta_token: str
    meta_ig_id: str
    meta_api_version: str
    daily_leads: int
    # "HH:MM" in `timezone`; empty disables the daily run.
    daily_time: str
    # ISO weekdays (1 = Monday) when the daily run happens.
    daily_weekdays: frozenset[int]
    timezone: str
    # Optional lead filters, 0 = off: an account must have posted within `active_days` days, have at most
    # `max_posts` posts, and the business at most `max_reviews` Google reviews.
    active_days: int
    max_posts: int
    max_reviews: int
    # Minimal fit score (1-10) from the model; 0 keeps every business the model accepts.
    min_score: int
    db_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        allowed = _ids(os.environ.get("ALLOWED_USERS", ""))
        chat = os.environ.get("LEADS_CHAT_ID", "").strip()
        return cls(
            bot_token=os.environ["BOT_TOKEN"],
            allowed_users=frozenset(allowed),
            leads_chat_id=int(chat) if chat else (allowed[0] if allowed else 0),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_models=tuple(
                m.strip()
                for m in os.environ.get("GEMINI_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite").split(",")
                if m.strip()
            ),
            apify_token=os.environ.get("APIFY_TOKEN", ""),
            places_per_day=int(os.environ.get("APIFY_PLACES_PER_DAY", "40")),
            places_per_query=int(os.environ.get("APIFY_PLACES_PER_QUERY", "5")),
            apify_monthly_budget=float(os.environ.get("APIFY_MONTHLY_BUDGET", "4.5")),
            apify_instagram=os.environ.get("APIFY_INSTAGRAM", "").strip().lower() in ("1", "true", "yes"),
            meta_token=os.environ.get("META_ACCESS_TOKEN", "").strip(),
            meta_ig_id=os.environ.get("META_IG_USER_ID", "").strip(),
            meta_api_version=os.environ.get("META_API_VERSION", "v23.0").strip(),
            daily_leads=int(os.environ.get("DAILY_LEADS", "40")),
            daily_time=os.environ.get("DAILY_TIME", "10:00").strip(),
            daily_weekdays=frozenset(_ids(os.environ.get("DAILY_WEEKDAYS", "1,2,3,4,5"))),
            timezone=os.environ.get("TIMEZONE", "Asia/Yerevan"),
            active_days=int(os.environ.get("ACTIVE_DAYS", "0")),
            max_posts=int(os.environ.get("MAX_POSTS", "0")),
            max_reviews=int(os.environ.get("MAX_REVIEWS", "0")),
            min_score=int(os.environ.get("MIN_SCORE", "0")),
            db_path=Path(os.environ.get("DB_PATH", "/data/leads.sqlite3")),
        )
