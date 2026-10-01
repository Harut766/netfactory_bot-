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
    today: str,
) -> AsyncIterator[Lead]:
    """`today` is the local date: the Apify budget is counted per day."""
    ig = instagram.InstagramClient(session, cfg.apify_token if cfg.apify_instagram else "")
    found = llm_errors = restarts = 0
    warned_blocked = False

    used = await places.monthly_usage(session, cfg.apify_token)
    if used is not None and used >= cfg.apify_monthly_budget:
        await notify(f"⚠️ Бесплатный лимит Apify на этот месяц почти исчерпан (${used:.2f}). "
                     "Новые компании появятся после обновления лимита в начале следующего месяца.")
        return

    while found < want:
        budget = cfg.places_per_day - db.places_bought(today)
        if budget <= 0:
            await notify(f"ℹ️ Дневной лимит Apify исчерпан ({cfg.places_per_day} компаний). "
                         "Завтра продолжу с того же места.")
            return
        per_query = min(cfg.places_per_query, budget)
        queries = db.next_queries(max(1, budget // per_query))
        if not queries:
            restarts += 1
            if restarts > 1:
                await notify("⚠️ Все поисковые запросы пройдены, новых компаний пока нет.")
                return
            db.restart_queries()
            continue
        # Counted before the run: even a failed run may have been charged.
        db.add_places_bought(today, per_query * len(queries))
        # ~$5 per 1000 places on the free plan, plus a margin for the start fee.
        batch = await places.search(session, cfg.apify_token, queries, per_query,
                                    max_charge_usd=0.05 + per_query * len(queries) * 0.008)
        db.finish_queries(queries)

        for place in batch:
            if found >= want:
                return
            if is_cancelled():
                raise Cancelled
            if db.place_checked(place.id):
                continue

            if place.reviews > cfg.max_reviews:
                db.mark_place(place.id, "old_reviews")
                continue
            if place.instagram:
                handle, site = instagram.extract_handle(place.instagram), ""
            else:
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
                        "аккаунтов не проверена. Обычно блок снимается через пару часов."
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
                verdict = await llm.evaluate(session, cfg.gemini_api_key, cfg.gemini_models, context, today)
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
