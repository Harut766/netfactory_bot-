"""Google Places API (New): Text Search."""

from dataclasses import dataclass

import aiohttp

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_FIELDS = (
    "id",
    "displayName",
    "formattedAddress",
    "websiteUri",
    "internationalPhoneNumber",
    "rating",
    "userRatingCount",
    "primaryTypeDisplayName",
    "businessStatus",
    "googleMapsUri",
)
FIELD_MASK = ",".join(f"places.{f}" for f in _FIELDS) + ",nextPageToken"
# Armenia's bounding box, so "Kentron" or "Armavir" never resolve to a place abroad.
ARMENIA = {"rectangle": {"low": {"latitude": 38.8, "longitude": 43.4}, "high": {"latitude": 41.35, "longitude": 46.7}}}


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


def parse_place(raw: dict) -> Place | None:
    if raw.get("businessStatus", "OPERATIONAL") != "OPERATIONAL":
        return None
    return Place(
        id=raw["id"],
        name=(raw.get("displayName") or {}).get("text", ""),
        address=raw.get("formattedAddress", ""),
        website=raw.get("websiteUri", ""),
        phone=raw.get("internationalPhoneNumber", ""),
        rating=raw.get("rating"),
        reviews=int(raw.get("userRatingCount") or 0),
        category=(raw.get("primaryTypeDisplayName") or {}).get("text", ""),
        maps_url=raw.get("googleMapsUri", ""),
    )


async def search(
    session: aiohttp.ClientSession, api_key: str, query: str, page_token: str | None = None
) -> tuple[list[Place], str | None]:
    """One page (up to 20 places) and the token of the next page, if any."""
    body = {"textQuery": query, "languageCode": "hy", "regionCode": "AM", "pageSize": 20,
            "locationRestriction": ARMENIA}
    if page_token:
        body["pageToken"] = page_token
    headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": FIELD_MASK}
    async with session.post(SEARCH_URL, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as r:
        data = await r.json(content_type=None)
        if r.status != 200:
            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
            raise RuntimeError(f"Google Places {r.status}: {msg or data}")
    places = [p for p in map(parse_place, data.get("places", [])) if p]
    return places, data.get("nextPageToken")
