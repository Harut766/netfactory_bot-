import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("BOT_TOKEN", "123:test")
os.environ.setdefault("ALLOWED_USERS", "1")
os.environ.setdefault("DB_PATH", ":memory:")

from leadbot import instagram, llm, pipeline, places  # noqa: E402
from leadbot.config import Config  # noqa: E402
from leadbot.db import DB  # noqa: E402
from leadbot.queries import all_queries  # noqa: E402

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def cfg() -> Config:
    return Config.from_env()


# ---------- instagram ----------

@pytest.mark.parametrize("text, handle", [
    ("https://www.instagram.com/Beauty.Salon_am/", "beauty.salon_am"),
    ("https://instagram.com/shop.am?igsh=abc", "shop.am"),
    ('<a href="//instagram.com/p/XYZ">post</a> <a href="https://instagram.com/realone">', "realone"),
    ("instagram.com/reel/abc", None),
    ("no links here", None),
])
def test_extract_handle(text, handle):
    assert instagram.extract_handle(text) == handle


def test_site_summary():
    page = '<html><title> Ծաղիկներ &amp; Co </title><meta name="description" content="Flowers in Yerevan"></html>'
    assert instagram.site_summary(page) == "Ծաղիկներ & Co — Flowers in Yerevan"


def _web_profile(posts: int, times: list[int], private=False) -> dict:
    return {"data": {"user": {
        "username": "shop", "full_name": "Shop", "biography": "bio", "category_name": "Shopping",
        "is_private": private,
        "edge_followed_by": {"count": 1200},
        "edge_owner_to_timeline_media": {"count": posts, "edges": [
            {"node": {"taken_at_timestamp": t, "edge_media_to_caption": {"edges": [{"node": {"text": f"c{t}"}}]}}}
            for t in times
        ]},
    }}}


def test_parse_web_profile():
    t1, t2 = int((NOW - timedelta(days=3)).timestamp()), int((NOW - timedelta(days=40)).timestamp())
    p = instagram.parse_web_profile(_web_profile(2, [t2, t1]))
    assert (p.followers, p.posts, p.bio) == (1200, 2, "bio")
    assert p.last_post == datetime.fromtimestamp(t1, timezone.utc)
    # All posts are visible, so the first one is known.
    assert p.first_post == datetime.fromtimestamp(t2, timezone.utc)
    p = instagram.parse_web_profile(_web_profile(50, [t1, t2]))
    assert p.first_post is None
    assert instagram.parse_web_profile({"data": {"user": None}}) is None


def test_parse_apify_profile():
    p = instagram.parse_apify_profile({
        "username": "shop", "followersCount": 10, "postsCount": 5,
        "latestPosts": [{"timestamp": "2026-09-28T10:00:00.000Z", "caption": "hi"}],
    })
    assert p.last_post == datetime(2026, 9, 28, 10, tzinfo=timezone.utc)
    assert p.captions == ["hi"]
    assert instagram.parse_apify_profile({"error": "not_found"}) is None


# ---------- filters ----------

def _profile(days_ago: int | None, posts=50, private=False) -> instagram.Profile:
    times = [NOW - timedelta(days=days_ago)] if days_ago is not None else []
    return instagram.Profile(username="x", posts=posts, is_private=private, post_times=times)


def test_reject_reason():
    c = cfg()
    assert pipeline.reject_reason(_profile(5), NOW, c) is None
    assert pipeline.reject_reason(_profile(c.active_days + 1), NOW, c) == "ig_inactive"
    assert pipeline.reject_reason(_profile(None), NOW, c) == "ig_inactive"
    assert pipeline.reject_reason(_profile(5, posts=c.max_posts + 1), NOW, c) == "ig_old"
    assert pipeline.reject_reason(_profile(5, private=True), NOW, c) == "ig_private"


def test_parse_place_skips_closed():
    raw = {"id": "a", "displayName": {"text": "Cafe"}, "userRatingCount": 12, "businessStatus": "OPERATIONAL"}
    assert places.parse_place(raw).reviews == 12
    assert places.parse_place(raw | {"businessStatus": "CLOSED_PERMANENTLY"}) is None


# ---------- llm ----------

def test_parse_verdict():
    v = llm.parse_verdict('```json\n{"fit": true, "score": "8", "reason": "r", "idea": "i", "message": "Բարև"}\n```')
    assert (v.fit, v.score, v.message) == (True, 8, "Բարև")
    assert llm.parse_verdict('{"fit": false, "score": 2}').fit is False
    with pytest.raises(llm.LLMError):
        llm.parse_verdict('{"fit": true, "score": 9, "message": ""}')
    with pytest.raises(llm.LLMError):
        llm.parse_verdict("not json")


# ---------- db ----------

def test_queries_progress():
    db = DB(":memory:")
    db.sync_queries(["a", "b"])
    assert db.next_query() == ("a", None)
    db.advance_query("a", "tok")
    assert db.next_query() == ("a", "tok")
    db.advance_query("a", None)
    db.advance_query("b", None)
    assert db.next_query() is None
    db.restart_queries()
    assert db.next_query() == ("a", None)
    # A new query inserted before existing ones keeps the order and the progress of the rest.
    db.advance_query("a", "tok2")
    db.sync_queries(["new", "a"])
    assert db.next_query() == ("new", None)
    db.advance_query("new", None)
    assert db.next_query() == ("a", "tok2")


def test_leads_flow():
    db = DB(":memory:")
    lead_id = db.add_lead("p1", "shop", {"name": "Shop"}, 8, "r", "i", "msg")
    assert db.place_checked("p1") and db.instagram_taken("shop")
    db.set_message(lead_id, "msg2", "i2")
    db.set_status(lead_id, "sent", "@me")
    lead = db.lead(lead_id)
    assert (lead["message"], lead["status"], lead["context"]["name"]) == ("msg2", "sent", "Shop")
    s = db.stats("2000-01-01")
    assert s["leads"] == {"sent": 1} and s["sent_today"] == 1


def test_all_queries_unique():
    q = all_queries()
    assert len(q) == len(set(q)) > 500


# ---------- telegram card ----------

def test_format_card_escapes():
    from leadbot.main import card_keyboard, format_card

    lead = {
        "id": 1, "instagram": "shop", "score": 8, "reason": "a < b", "idea": "x", "message": "Բարև <Ձեզ>",
        "status": "new", "handled_by": None,
        "context": {"name": "A&B", "category": "Cafe", "instagram": "shop", "ig_checked": True,
                    "followers": 10, "posts": 5, "last_post": NOW.date().isoformat(), "first_post": None,
                    "website": "https://ab.am", "rating": 4.8, "google_reviews": 20},
    }
    text = format_card(lead)
    assert "A&amp;B" in text and "Բարև &lt;Ձեզ&gt;" in text and "a &lt; b" in text
    assert len(card_keyboard(lead).inline_keyboard) == 2
    lead["status"], lead["handled_by"] = "sent", "@me"
    assert "Отправлено — @me" in format_card(lead)
    assert len(card_keyboard(lead).inline_keyboard) == 1


# ---------- pipeline (network replaced with fakes) ----------

def test_find_leads(monkeypatch):
    import asyncio

    def place(i, reviews=10, site=""):
        return places.Place(id=f"p{i}", name=f"N{i}", address="", website=site, phone="", rating=None,
                            reviews=reviews, category="", maps_url="")

    batch = [
        place(1, reviews=999, site="https://instagram.com/old"),  # too many reviews
        place(2),  # no website, no instagram
        place(3, site="https://instagram.com/inactive"),
        place(4, site="https://instagram.com/good"),
        place(5, site="https://instagram.com/notfit"),
        place(6, site="https://instagram.com/good"),  # same account as p4
    ]

    async def fake_search(session, key, query, token):
        return batch, None

    async def fake_profile(self, handle):
        return _profile(90 if handle == "inactive" else 2)

    async def fake_evaluate(session, key, models, ctx, today):
        fit = ctx["instagram"] != "notfit"
        return llm.Verdict(fit=fit, score=8 if fit else 2, reason="r", idea="i", message="Բարև" if fit else "")

    monkeypatch.setattr(places, "search", fake_search)
    monkeypatch.setattr(instagram.InstagramClient, "profile", fake_profile)
    monkeypatch.setattr(llm, "evaluate", fake_evaluate)

    db = DB(":memory:")
    db.sync_queries(["q1"])
    notes = []

    async def notify(text):
        notes.append(text)

    async def collect():
        return [lead async for lead in pipeline.find_leads(cfg(), db, None, 5, lambda: False, notify)]

    leads = asyncio.run(collect())
    assert [db.lead(lead.id)["instagram"] for lead in leads] == ["good"]
    s = db.stats("2000-01-01")["places"]
    assert s == {"old_reviews": 1, "no_instagram": 1, "ig_inactive": 1, "lead": 1, "not_fit": 1, "duplicate": 1}
    # The only query ran out twice in a row: the run stops and says so.
    assert any("новых компаний пока нет" in n for n in notes)
