"""Finding a business's Instagram account and reading its public profile."""

import asyncio
import html
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp

log = logging.getLogger(__name__)

HANDLE_RE = re.compile(r"(?:https?:)?(?://)?(?:www\.|m\.)?instagram\.com/([A-Za-z0-9_.]{1,30})", re.I)
# Paths on instagram.com that are not accounts.
RESERVED = {
    "p", "reel", "reels", "tv", "explore", "stories", "accounts", "about", "developer", "legal",
    "direct", "share", "web", "privacy", "terms", "help", "static", "embed.js", "instagram",
}
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
DESCRIPTION_RE = re.compile(
    r"<meta[^>]+(?:name|property)=[\"'](?:og:)?description[\"'][^>]*content=[\"']([^\"']*)", re.I
)
SCRIPT_RE = re.compile(r"<(script|style|noscript|svg)\b.*?</\1>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
WEB_APP_SIGNALS = [
    ("mobile_app", re.compile(r"apps\.apple\.com|play\.google\.com/store/apps", re.I)),
    ("login_or_account", re.compile(
        r"log ?in\b|sign ?in\b|my account|личный кабинет|войти|մուտք գործել|անձնական (էջ|գրասենյակ)", re.I)),
    ("online_store", re.compile(r"add to cart|checkout|/cart\b|корзин|в корзину|զամբյուղ", re.I)),
    ("online_booking", re.compile(
        r"book now|online booking|онлайн[- ]запись|записаться онлайн|առցանց ամրագր|"
        r"calendly\.com|booksy\.com|fresha\.com|altegio|yclients", re.I)),
]
PROFILE_URL = "https://i.instagram.com/api/v1/users/web_profile_info/"
# The public app id of instagram.com; the endpoint refuses requests without it.
IG_HEADERS = {
    "x-ig-app-id": "936619743392459",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
APIFY_URL = "https://api.apify.com/v2/acts/apify~instagram-profile-scraper/run-sync-get-dataset-items"
MAX_PAGE = 2_000_000


class Blocked(Exception):
    """Instagram refuses anonymous requests from this IP for now."""


@dataclass
class Profile:
    username: str
    full_name: str = ""
    bio: str = ""
    category: str = ""
    followers: int = 0
    posts: int = 0
    is_private: bool = False
    # Times of the most recent posts (pinned ones included), newest first.
    post_times: list[datetime] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)

    @property
    def last_post(self) -> datetime | None:
        return max(self.post_times, default=None)

    @property
    def first_post(self) -> datetime | None:
        """Known only when every post of the account is among the fetched ones."""
        if self.post_times and self.posts <= len(self.post_times):
            return min(self.post_times)
        return None


def extract_handle(text: str) -> str | None:
    for m in HANDLE_RE.finditer(text):
        handle = m.group(1).rstrip(".").lower()
        if handle and handle not in RESERVED:
            return handle
    return None


def site_summary(page: str) -> str:
    parts = []
    if m := TITLE_RE.search(page):
        parts.append(m.group(1))
    if m := DESCRIPTION_RE.search(page):
        parts.append(m.group(1))
    text = " — ".join(" ".join(html.unescape(p).split()) for p in parts if p.strip())
    return text[:400]


def visible_text(page: str, limit: int = 1500) -> str:
    page = SCRIPT_RE.sub(" ", page)
    return " ".join(html.unescape(TAG_RE.sub(" ", page)).split())[:limit]


def web_app_signals(page: str) -> list[str]:
    """Hints that the business already runs its own web or mobile application."""
    return [name for name, pattern in WEB_APP_SIGNALS if pattern.search(page)]


def analyze_site(page: str) -> dict:
    return {"about": site_summary(page), "text": visible_text(page), "web_app_signals": web_app_signals(page)}


async def find_handle(session: aiohttp.ClientSession, website: str) -> tuple[str | None, dict]:
    """Instagram handle and what the site says about the business (see analyze_site); {} without a site."""
    if not website:
        return None, {}
    if "instagram.com" in website.lower():
        return extract_handle(website), {}
    try:
        async with session.get(
            website,
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": IG_HEADERS["User-Agent"]},
            # Small business sites often have broken certificates; we only read public pages.
            ssl=False,
        ) as r:
            if r.status >= 400:
                return None, {}
            raw = await r.content.read(MAX_PAGE)
    except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError, ValueError) as e:
        log.info("site %s: %s", website, e)
        return None, {}
    page = raw.decode("utf-8", errors="replace")
    return extract_handle(page), analyze_site(page)


def _ts(value) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def parse_web_profile(data: dict) -> Profile | None:
    user = (data.get("data") or {}).get("user")
    if not user:
        return None
    media = user.get("edge_owner_to_timeline_media") or {}
    times, captions = [], []
    for edge in media.get("edges", []):
        node = edge.get("node") or {}
        if t := _ts(node.get("taken_at_timestamp")):
            times.append(t)
        cap_edges = (node.get("edge_media_to_caption") or {}).get("edges") or []
        if cap_edges and (text := (cap_edges[0].get("node") or {}).get("text")):
            captions.append(text)
    return Profile(
        username=user.get("username", ""),
        full_name=user.get("full_name") or "",
        bio=user.get("biography") or "",
        category=user.get("category_name") or "",
        followers=int((user.get("edge_followed_by") or {}).get("count") or 0),
        posts=int(media.get("count") or 0),
        is_private=bool(user.get("is_private")),
        post_times=sorted(times, reverse=True),
        captions=captions,
    )


def parse_apify_profile(item: dict) -> Profile | None:
    if not item.get("username") or item.get("error"):
        return None
    times, captions = [], []
    for post in item.get("latestPosts") or []:
        if t := _ts(post.get("timestamp")):
            times.append(t)
        if post.get("caption"):
            captions.append(post["caption"])
    return Profile(
        username=item["username"],
        full_name=item.get("fullName") or "",
        bio=item.get("biography") or "",
        category=item.get("businessCategoryName") or "",
        followers=int(item.get("followersCount") or 0),
        posts=int(item.get("postsCount") or 0),
        is_private=bool(item.get("private")),
        post_times=sorted(times, reverse=True),
        captions=captions,
    )


class InstagramClient:
    """Reads profiles directly from instagram.com; switches to Apify (if configured) once Instagram blocks us."""

    def __init__(self, session: aiohttp.ClientSession, apify_token: str = ""):
        self.session = session
        self.apify_token = apify_token
        self.blocked = False
        self._last_request = 0.0

    async def profile(self, username: str) -> Profile | None:
        """None if the account does not exist. Raises Blocked when no way to read it is left."""
        if not self.blocked:
            try:
                return await self._direct(username)
            except Blocked:
                self.blocked = True
                log.warning("Instagram blocks direct requests")
        if self.apify_token:
            return await self._apify(username)
        raise Blocked

    async def _direct(self, username: str) -> Profile | None:
        # A human-like pace keeps the IP from being blocked.
        loop = asyncio.get_running_loop()
        wait = self._last_request + random.uniform(4, 9) - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = loop.time()
        async with self.session.get(
            PROFILE_URL, params={"username": username}, headers=IG_HEADERS,
            timeout=aiohttp.ClientTimeout(total=30), allow_redirects=False,
        ) as r:
            if r.status == 404:
                return None
            if r.status in (401, 403, 429) or 300 <= r.status < 400:
                raise Blocked
            if r.status != 200:
                raise RuntimeError(f"Instagram {r.status}")
            try:
                data = await r.json(content_type=None)
            except ValueError:
                # A login page instead of JSON.
                raise Blocked from None
        return parse_web_profile(data)

    async def _apify(self, username: str) -> Profile | None:
        async with self.session.post(
            APIFY_URL, params={"token": self.apify_token}, json={"usernames": [username]},
            timeout=aiohttp.ClientTimeout(total=180),
        ) as r:
            if r.status >= 300:
                raise RuntimeError(f"Apify {r.status}: {(await r.text())[:200]}")
            items = await r.json(content_type=None)
        return parse_apify_profile(items[0]) if items else None
