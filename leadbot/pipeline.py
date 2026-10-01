"""Search → filter → evaluate: produces ready-to-send leads one by one."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
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
    """Optional activity/age filters (0 in the config turns a filter off). None means the account passes."""
    last = profile.last_post
    if cfg.active_days and (last is None or (now - last).days > cfg.active_days):
        return "ig_inactive"
    if cfg.max_posts and profile.posts > cfg.max_posts:
        return "ig_old"
    return None


def build_context(place: places.Place, handle: str | None, site: dict, profile: instagram.Profile | None) -> dict:
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
    """Yields up to `want` lead cards. Every place bought on Apify becomes a card (or waits in the pending
    queue for the next run); `today` is the local date the Apify budget is counted for."""
    ig = instagram.InstagramClient(session, cfg.apify_token if cfg.apify_instagram else "", cfg.meta_token,
                                   cfg.meta_ig_id, cfg.meta_api_version)
    found = llm_errors = restarts = 0
    warned_blocked = warned_meta = month_checked = False

    async def warn_meta() -> None:
        nonlocal warned_meta
        if ig.meta_error and not warned_meta:
            warned_meta = True
            await notify(f"⚠️ Meta API не отвечает: {ig.meta_error}. Проверь META_ACCESS_TOKEN в .env.")

    async def buy(n: int) -> bool:
        """Buys about `n` new places from Apify into the pending queue. False when nothing more can be bought."""
        nonlocal month_checked, restarts
        if not month_checked:
            month_checked = True
            used = await places.monthly_usage(session, cfg.apify_token)
            if used is not None and used >= cfg.apify_monthly_budget:
                await notify(f"⚠️ Бесплатный лимит Apify на этот месяц почти исчерпан (${used:.2f}). "
                             "Новые компании появятся после обновления лимита в начале следующего месяца.")
                return False
        budget = cfg.places_per_day - db.places_bought(today)
        if budget <= 0:
            await notify(f"ℹ️ Дневной лимит Apify исчерпан ({cfg.places_per_day} компаний). "
                         "Завтра продолжу с того же места.")
            return False
        per_query = min(cfg.places_per_query, budget)
        queries = db.next_queries(max(1, min(-(-n // per_query), budget // per_query)))
        if not queries:
            restarts += 1
            if restarts > 1:
                await notify("⚠️ Все поисковые запросы пройдены, новых компаний пока нет.")
                return False
            db.restart_queries()
            return True
        try:
            # ~$5 per 1000 places on the free plan, plus a margin for the start fee.
            batch = await places.search(session, cfg.apify_token, queries, per_query,
                                        max_charge_usd=0.05 + per_query * len(queries) * 0.008)
        except places.StartError:
            raise
        except Exception:
            # The run started, so it may have been charged even though it failed.
            db.add_places_bought(today, per_query * len(queries))
            raise
        db.add_places_bought(today, per_query * len(queries))
        db.add_pending([(p.id, asdict(p)) for p in batch if not db.place_checked(p.id)])
        db.finish_queries(queries)
        return True

    while found < want:
        if is_cancelled():
            raise Cancelled
        raw = db.pop_pending()
        if raw is None:
            if not await buy(want - found):
                return
            continue
        place = places.Place(**raw)
        if db.place_checked(place.id):
            continue

        if cfg.max_reviews and place.reviews > cfg.max_reviews:
            db.mark_place(place.id, "old_reviews")
            continue
        if place.instagram:
            handle, site = instagram.extract_handle(place.instagram), {}
            if place.website and "instagram.com" not in place.website.lower():
                site = (await instagram.find_handle(session, place.website))[1]
        else:
            handle, site = await instagram.find_handle(session, place.website)
        if db.instagram_taken(handle):
            # The same business (e.g. another branch) already got a card.
            db.mark_place(place.id, "duplicate")
            continue

        profile = None
        if handle:
            try:
                profile = await ig.profile(handle)
            except instagram.Blocked:
                await warn_meta()
                if not warned_blocked:
                    warned_blocked = True
                    await notify(
                        "⚠️ Instagram временно не отдаёт данные профилей. Карточки идут дальше, но без "
                        "статистики Instagram."
                    )
            except Exception as e:
                # A network hiccup on one profile: the card just goes without Instagram stats.
                log.warning("instagram %s: %s", handle, e)
            else:
                await warn_meta()
                if profile is None:
                    # No such account: the link on the site is stale.
                    handle = None
                elif reason := reject_reason(profile, datetime.now(timezone.utc), cfg):
                    db.mark_place(place.id, reason)
                    continue

        context = build_context(place, handle, site, profile)
        try:
            verdict = await llm.evaluate(session, cfg.gemini_api_key, cfg.gemini_models, context, today)
        except llm.LLMError as e:
            log.warning("gemini for %s: %s", place.name, e)
            db.add_pending([(place.id, raw)])
            llm_errors += 1
            if llm_errors >= MAX_LLM_ERRORS:
                raise
            continue
        llm_errors = 0
        if cfg.min_score and verdict.score < cfg.min_score:
            db.mark_place(place.id, "low_score")
            continue

        context |= {"web_presence": verdict.web_presence, "has_web_app": verdict.has_web_app, "fit": verdict.fit}
        lead_id = db.add_lead(place.id, handle, context, verdict.score, verdict.reason, verdict.idea,
                              verdict.message, verdict.summary)
        found += 1
        yield Lead(lead_id)
