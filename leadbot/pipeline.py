"""Search → filter → evaluate: produces ready-to-send leads one by one."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from . import instagram, llm, places
from .config import Config
from .db import DB

log = logging.getLogger(__name__)

# Safety limits for one run, so a bad day never burns the Google quota.
MAX_PAGES_PER_RUN = 150
CHECKS_PER_LEAD = 25
MAX_LLM_ERRORS = 3


class Cancelled(Exception):
    pass


@dataclass
class Lead:
    id: int


def reject_reason(profile: instagram.Profile, now: datetime, cfg: Config) -> str | None:
    """Why an Instagram account does not look like a new and active business, or None if it does."""
    if profile.is_private:
        return "ig_private"
    last = profile.last_post
    if last is None or (now - last).days > cfg.active_days:
        return "ig_inactive"
    if profile.posts > cfg.max_posts:
        return "ig_old"
    return None


def build_context(place: places.Place, handle: str, site: str, profile: instagram.Profile | None) -> dict:
    ctx = {
        "name": place.name,
        "category": place.category,
        "address": place.address,
        "phone": place.phone,
        "website": place.website,
        "maps_url": place.maps_url,
        "rating": place.rating,
        "google_reviews": place.reviews,
        "site": site,
        "instagram": handle,
        "ig_checked": profile is not None,
    }
    if profile:
        first = profile.first_post
        ctx |= {
            "ig_name": profile.full_name,
            "ig_bio": profile.bio,
            "ig_category": profile.category,
            "followers": profile.followers,
            "posts": profile.posts,
            "last_post": profile.last_post.date().isoformat() if profile.last_post else None,
            "first_post": first.date().isoformat() if first else None,
            "recent_captions": [" ".join(c.split())[:300] for c in profile.captions[:6]],
        }
    return ctx


async def find_leads(
    cfg: Config,
    db: DB,
    session: aiohttp.ClientSession,
    want: int,
    is_cancelled: Callable[[], bool],
    notify: Callable[[str], Awaitable[None]],
) -> AsyncIterator[Lead]:
    ig = instagram.InstagramClient(session, cfg.apify_token)
    found = checked = pages = llm_errors = restarts = 0
    warned_blocked = False

    while found < want:
        if pages >= MAX_PAGES_PER_RUN or checked >= want * CHECKS_PER_LEAD:
            await notify(f"⚠️ Лимит одного запуска: проверено {checked} компаний. Остальное — в следующий раз.")
            return
        nxt = db.next_query()
        if nxt is None:
            restarts += 1
            if restarts > 1:
                await notify("⚠️ Все поисковые запросы пройдены, новых компаний пока нет.")
                return
            db.restart_queries()
            continue
        query, token = nxt
        batch, next_token = await places.search(session, cfg.google_api_key, query, token)
        pages += 1
        db.advance_query(query, next_token)

        for place in batch:
            if found >= want:
                return
            if is_cancelled():
                raise Cancelled
            if db.place_checked(place.id):
                continue
            checked += 1

            if place.reviews > cfg.max_reviews:
                db.mark_place(place.id, "old_reviews")
                continue
            handle, site = await instagram.find_handle(session, place.website)
            if not handle:
                db.mark_place(place.id, "no_instagram")
                continue
            if db.instagram_taken(handle):
                db.mark_place(place.id, "duplicate")
                continue

            try:
                profile = await ig.profile(handle)
            except instagram.Blocked:
                profile = None
                if not warned_blocked:
                    warned_blocked = True
                    await notify(
                        "⚠️ Instagram временно блокирует проверку профилей. Лиды идут дальше, но активность "
                        "аккаунтов не проверена. Решение: подождать пару часов или добавить APIFY_TOKEN."
                    )
            except Exception as e:
                # Not the place's fault: leave it unchecked so a later run retries it.
                log.warning("instagram %s: %s", handle, e)
                continue
            else:
                if profile is None:
                    db.mark_place(place.id, "ig_not_found")
                    continue
                if reason := reject_reason(profile, datetime.now(timezone.utc), cfg):
                    db.mark_place(place.id, reason)
                    continue

            context = build_context(place, handle, site, profile)
            try:
                verdict = await llm.evaluate(
                    session, cfg.gemini_api_key, cfg.gemini_models, context, datetime.now().date().isoformat()
                )
            except llm.LLMError as e:
                log.warning("gemini for %s: %s", place.name, e)
                llm_errors += 1
                if llm_errors >= MAX_LLM_ERRORS:
                    raise
                continue
            llm_errors = 0
            if not verdict.fit or verdict.score < cfg.min_score:
                db.mark_place(place.id, "not_fit")
                continue

            lead_id = db.add_lead(place.id, handle, context, verdict.score, verdict.reason, verdict.idea,
                                  verdict.message)
            found += 1
            yield Lead(lead_id)
