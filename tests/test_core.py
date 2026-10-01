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


def cfg(**overrides) -> Config:
    from dataclasses import replace

    return replace(Config.from_env(), **overrides)


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
    # Filters are off by default: every account passes.
    assert pipeline.reject_reason(_profile(None, posts=10_000), NOW, cfg()) is None
    c = cfg(active_days=30, max_posts=400)
    assert pipeline.reject_reason(_profile(5), NOW, c) is None
    assert pipeline.reject_reason(_profile(31), NOW, c) == "ig_inactive"
    assert pipeline.reject_reason(_profile(None), NOW, c) == "ig_inactive"
    assert pipeline.reject_reason(_profile(5, posts=401), NOW, c) == "ig_old"


def test_analyze_site():
    page = ('<html><title>Shop</title><script>var login = 1</script><body>'
            '<a href="https://apps.apple.com/x">App</a> Մուտք գործել <b>Add to cart</b></body></html>')
    site = instagram.analyze_site(page)
    assert site["web_app_signals"] == ["mobile_app", "login_or_account", "online_store"]
    assert "var login" not in site["text"] and "Add to cart" in site["text"]
    assert instagram.analyze_site("<title>Flowers</title><p>Call us on WhatsApp</p>")["web_app_signals"] == []


def test_parse_place():
    raw = {"placeId": "a", "title": "Cafe", "reviewsCount": 12, "totalScore": 4.5,
           "instagrams": ["https://www.instagram.com/cafe.am/"]}
    p = places.parse_place(raw)
    assert (p.name, p.reviews, p.instagram) == ("Cafe", 12, "https://www.instagram.com/cafe.am/")
    assert places.parse_place(raw | {"permanentlyClosed": True}) is None
    assert places.parse_place({"title": "no id"}) is None


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
    db.sync_queries(["a", "b", "c"])
    assert db.next_queries(2) == ["a", "b"]
    db.finish_queries(["a", "b"])
    assert db.next_queries(2) == ["c"]
    db.finish_queries(["c"])
    assert db.next_queries(2) == []
    db.restart_queries()
    assert db.next_queries(1) == ["a"]
    # A new query inserted before existing ones keeps the order and the progress of the rest.
    db.finish_queries(["a"])
    db.sync_queries(["new", "a", "b"])
    assert db.next_queries(5) == ["new", "b"]


def test_apify_budget():
    db = DB(":memory:")
    assert db.places_bought("2026-10-01") == 0
    db.add_places_bought("2026-10-01", 30)
    db.add_places_bought("2026-10-01", 5)
    assert db.places_bought("2026-10-01") == 35
    assert db.places_bought("2026-10-02") == 0


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
                    "website": "https://ab.am", "rating": 4.8, "google_reviews": 20,
                    "web_presence": "только Instagram, своего приложения нет"},
    }
    text = format_card(lead)
    assert "Веб-приложение:</b> только Instagram" in text
    assert "A&amp;B" in text and "Բարև &lt;Ձեզ&gt;" in text and "a &lt; b" in text
    assert len(card_keyboard(lead).inline_keyboard) == 2
    lead["status"], lead["handled_by"] = "sent", "@me"
    assert "Отправлено — @me" in format_card(lead)
    assert len(card_keyboard(lead).inline_keyboard) == 1


# ---------- pipeline (network replaced with fakes) ----------

def _fake_place(i, reviews=10, site="", ig=""):
    return places.Place(id=f"p{i}", name=f"N{i}", address="", website=site, phone="", rating=None,
                        reviews=reviews, category="", maps_url="", instagram=ig)


def _run_find_leads(monkeypatch, batch, want=5, usage=0.0, config=None):
    import asyncio

    calls = []

    async def fake_search(session, token, queries, per_query, max_charge_usd):
        calls.append((list(queries), per_query))
        return batch if len(calls) == 1 else []

    async def fake_usage(session, token):
        return usage

    async def fake_profile(self, handle):
        return _profile(90 if handle == "inactive" else 2)

    async def fake_evaluate(session, key, models, ctx, today):
        has_app = ctx["instagram"] == "hasapp"
        fit = ctx["instagram"] not in ("notfit", "hasapp")
        return llm.Verdict(fit=fit, score=8 if fit else 2, reason="r", idea="i", message="Բարև" if fit else "",
                           summary="s", has_web_app=has_app, web_presence="w")

    monkeypatch.setattr(places, "search", fake_search)
    monkeypatch.setattr(places, "monthly_usage", fake_usage)
    monkeypatch.setattr(instagram.InstagramClient, "profile", fake_profile)
    monkeypatch.setattr(llm, "evaluate", fake_evaluate)

    db = DB(":memory:")
    db.sync_queries([f"q{i}" for i in range(10)])
    notes = []

    async def notify(text):
        notes.append(text)

    async def collect():
        return [lead async for lead in pipeline.find_leads(config or cfg(), db, None, want, lambda: False, notify,
                                                           "2026-10-01")]

    leads = asyncio.run(collect())
    return db, leads, calls, notes


def test_find_leads(monkeypatch):
    batch = [
        _fake_place(1, reviews=999, site="https://instagram.com/old"),
        _fake_place(2),  # no website, no instagram: nobody to message
        _fake_place(3, site="https://instagram.com/inactive"),
        _fake_place(4, ig="https://www.instagram.com/good/"),
        _fake_place(5, site="https://instagram.com/notfit"),
        _fake_place(6, site="https://instagram.com/good"),  # same account as p4
        _fake_place(7, site="https://instagram.com/hasapp"),
    ]
    db, leads, calls, notes = _run_find_leads(monkeypatch, batch)
    # Without filters, only the businesses with their own web app (and non-businesses) are dropped.
    assert [db.lead(lead.id)["instagram"] for lead in leads] == ["old", "inactive", "good"]
    assert db.lead(leads[0].id)["summary"] == "s"
    assert db.lead(leads[0].id)["context"]["web_presence"] == "w"
    s = db.stats("2000-01-01")["places"]
    assert s == {"lead": 3, "no_instagram": 1, "not_fit": 1, "duplicate": 1, "has_web_app": 1}
    # The daily budget (40 places) goes into one run of 4 queries × 10.
    assert calls == [(["q0", "q1", "q2", "q3"], 10)]
    assert db.places_bought("2026-10-01") == 40
    assert any("Дневной лимит" in n for n in notes)


def test_find_leads_with_filters(monkeypatch):
    batch = [
        _fake_place(1, reviews=999, site="https://instagram.com/old"),
        _fake_place(3, site="https://instagram.com/inactive"),
        _fake_place(4, ig="https://www.instagram.com/good/"),
    ]
    config = cfg(active_days=30, max_reviews=200, places_per_day=35)
    db, leads, calls, notes = _run_find_leads(monkeypatch, batch, config=config)
    assert [db.lead(lead.id)["instagram"] for lead in leads] == ["good"]
    assert db.stats("2000-01-01")["places"] == {"old_reviews": 1, "ig_inactive": 1, "lead": 1}
    # 35 places: 3 queries × 10, then 1 query × 5.
    assert calls == [(["q0", "q1", "q2"], 10), (["q3"], 5)]


def test_find_leads_stops_when_month_budget_spent(monkeypatch):
    db, leads, calls, notes = _run_find_leads(monkeypatch, [], usage=4.9)
    assert leads == [] and calls == []
    assert any("лимит Apify на этот месяц" in n for n in notes)


def test_refused_apify_start_does_not_spend_budget(monkeypatch):
    import asyncio

    async def refused(*args, **kwargs):
        raise places.StartError("Apify 400")

    async def fake_usage(session, token):
        return 0.0

    monkeypatch.setattr(places, "search", refused)
    monkeypatch.setattr(places, "monthly_usage", fake_usage)
    db = DB(":memory:")
    db.sync_queries(["q0"])

    async def notify(text):
        pass

    async def collect():
        return [x async for x in pipeline.find_leads(cfg(), db, None, 3, lambda: False, notify, "2026-10-01")]

    with pytest.raises(places.StartError):
        asyncio.run(collect())
    assert db.places_bought("2026-10-01") == 0
    assert db.next_queries(1) == ["q0"]


def test_parse_profile_page():
    page = ('<meta property="og:title" content="TUZ BARBERSHOP (@tuzbarbershop) &#x2022; Instagram photos and videos">'
            '<meta property="og:description" content="12.5K Followers, 310 Following, 1,204 Posts - '
            'TUZ BARBERSHOP (@tuzbarbershop) on Instagram: &quot;Barbershop in Yerevan&quot;">')
    p = instagram.parse_profile_page("tuzbarbershop", page)
    assert (p.full_name, p.followers, p.posts, p.bio) == ("TUZ BARBERSHOP", 12_500, 1204, "Barbershop in Yerevan")
    assert p.last_post is None
    page = ('<meta property="og:description" content="87 Followers, 5 Following, 9 Posts - '
            'See Instagram photos and videos from Flowers (@flowers.am)">')
    p = instagram.parse_profile_page("flowers.am", page)
    assert (p.followers, p.posts, p.bio) == (87, 9, "")
    assert instagram.parse_profile_page("x", "<html>Login • Instagram</html>") is None


def test_parse_business_discovery():
    p = instagram.parse_business_discovery({"business_discovery": {
        "username": "shop", "name": "Shop", "biography": "bio", "followers_count": 900, "media_count": 40,
        "media": {"data": [{"timestamp": "2026-09-28T10:00:00+0000", "caption": "new"},
                           {"timestamp": "2026-09-01T10:00:00+0000"}]},
    }, "id": "1"})
    assert (p.full_name, p.followers, p.posts, p.captions) == ("Shop", 900, 40, ["new"])
    assert p.last_post == datetime(2026, 9, 28, 10, tzinfo=timezone.utc)
    assert instagram.parse_business_discovery({"id": "1"}) is None


class _FakeResponse:
    def __init__(self, data):
        self.data = data

    async def json(self, content_type=None):
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, data):
        self.data = data

    def get(self, url, **kwargs):
        return _FakeResponse(self.data)


def test_meta_client(monkeypatch):
    import asyncio

    ok = {"business_discovery": {"username": "shop", "followers_count": 5, "media_count": 1}}
    client = instagram.InstagramClient(_FakeSession(ok), meta_token="t", meta_ig_id="1")
    assert asyncio.run(client.profile("shop")).followers == 5

    # A personal account is invisible to Business Discovery: the client falls back to the other ways.
    async def page(self, username):
        return instagram.Profile(username=username, followers=7)

    monkeypatch.setattr(instagram.InstagramClient, "_page", page)
    monkeypatch.setattr(instagram.InstagramClient, "_direct", lambda self, u: (_ for _ in ()).throw(instagram.Blocked))
    personal = {"error": {"code": 110, "message": "Invalid user id"}}
    client = instagram.InstagramClient(_FakeSession(personal), meta_token="t", meta_ig_id="1")
    assert asyncio.run(client.profile("me")).followers == 7
    assert not client.meta_blocked

    # An expired token stops Meta for the run and is reported.
    expired = {"error": {"code": 190, "message": "Session has expired"}}
    client = instagram.InstagramClient(_FakeSession(expired), meta_token="t", meta_ig_id="1")
    assert asyncio.run(client.profile("me")).followers == 7
    assert client.meta_blocked and client.meta_error == "Session has expired"
