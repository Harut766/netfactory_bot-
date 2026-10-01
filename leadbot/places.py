"""Google Maps places through Apify's Google Maps Scraper (compass/crawler-google-places)."""

import asyncio
from dataclasses import dataclass

import aiohttp

API = "https://api.apify.com/v2"
ACTOR = "compass~crawler-google-places"
RUN_TIMEOUT = 900
# Apify refuses a per-run charge cap below this.
MIN_CHARGE_USD = 0.5


class StartError(RuntimeError):
    """Apify refused to start the run, so nothing was charged."""


@dataclass
class Place:
    id: str
    name: str
    address: str
    website: str
    phone: str
    rating: float | None
    reviews: int
    category: str
    maps_url: str
    # Some scraper versions already return social links.
    instagram: str = ""


def parse_place(raw: dict) -> Place | None:
    if not raw.get("placeId") or raw.get("permanentlyClosed") or raw.get("temporarilyClosed"):
        return None
    instagrams = raw.get("instagrams") or []
    return Place(
        id=raw["placeId"],
        name=raw.get("title") or "",
        address=raw.get("address") or "",
        website=raw.get("website") or "",
        phone=raw.get("phone") or raw.get("phoneUnformatted") or "",
        rating=raw.get("totalScore"),
        reviews=int(raw.get("reviewsCount") or 0),
        category=raw.get("categoryName") or "",
        maps_url=raw.get("url") or "",
        instagram=instagrams[0] if instagrams and isinstance(instagrams[0], str) else "",
    )


async def _json(r: aiohttp.ClientResponse) -> dict:
    data = await r.json(content_type=None)
    if r.status >= 300:
        err = (data.get("error") or {}).get("message") if isinstance(data, dict) else data
        raise RuntimeError(f"Apify {r.status}: {err}")
    return data


async def search(session: aiohttp.ClientSession, token: str, queries: list[str], per_query: int,
                 max_charge_usd: float) -> list[Place]:
    """Runs the scraper once for several queries (one run = one start fee) and returns the places."""
    params = {"token": token, "maxTotalChargeUsd": f"{max(max_charge_usd, MIN_CHARGE_USD):.2f}"}
    body = {
        "searchStringsArray": queries,
        "maxCrawledPlacesPerSearch": per_query,
        "language": "en",
        "skipClosedPlaces": True,
        # Everything below costs extra per place; the listing already has what we need.
        "scrapePlaceDetailPage": False,
        "maxImages": 0,
        "maxReviews": 0,
        "scrapeContacts": False,
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with session.post(f"{API}/acts/{ACTOR}/runs", params=params, json=body, timeout=timeout) as r:
        try:
            run = (await _json(r))["data"]
        except RuntimeError as e:
            raise StartError(str(e)) from None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + RUN_TIMEOUT
    while run["status"] in ("READY", "RUNNING"):
        if loop.time() > deadline:
            raise RuntimeError("Apify: поиск на картах идёт слишком долго")
        await asyncio.sleep(10)
        async with session.get(f"{API}/actor-runs/{run['id']}", params={"token": token}, timeout=timeout) as r:
            run = (await _json(r))["data"]
    # TIMED-OUT / ABORTED runs (e.g. the charge cap was hit) still keep what they scraped.
    if run["status"] == "FAILED":
        raise RuntimeError(f"Apify: поиск на картах завершился ошибкой ({run.get('statusMessage') or ''})")
    async with session.get(
        f"{API}/datasets/{run['defaultDatasetId']}/items", params={"token": token, "clean": "true"},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as r:
        items = await r.json(content_type=None)
        if r.status >= 300:
            raise RuntimeError(f"Apify {r.status}: {items}")
    return [p for p in map(parse_place, items) if p]


async def monthly_usage(session: aiohttp.ClientSession, token: str) -> float | None:
    """Money spent on Apify this billing month, in USD; None if unknown."""
    try:
        async with session.get(f"{API}/users/me/limits", params={"token": token},
                               timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = (await _json(r))["data"]
        return float(data["current"]["monthlyUsageUsd"])
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, KeyError, TypeError, ValueError):
        return None
