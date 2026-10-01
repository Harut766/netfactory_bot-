import asyncio
import html
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, User

from . import llm, pipeline, places
from .config import Config
from .db import DB
from .queries import all_queries

log = logging.getLogger(__name__)

cfg = Config.from_env()
db = DB(cfg.db_path)
router = Router()
run_lock = asyncio.Lock()
# Keeps references to background searches started by /leads (asyncio only holds weak ones).
tasks: set[asyncio.Task] = set()
cancel_requested = False

PLACE_STATUSES = {
    "lead": "лиды",
    "has_web_app": "уже есть своё веб-приложение",
    "not_fit": "не бизнес или не в Армении",
    "no_instagram": "без Instagram",
    "ig_inactive": "неактивный Instagram",
    "ig_old": "давно существуют (много постов)",
    "old_reviews": "давно существуют (много отзывов)",
    "ig_not_found": "Instagram не найден",
    "duplicate": "дубли",
}


def _who(user: User) -> str:
    return f"@{user.username}" if user.username else user.full_name


def _days_ago(iso_date: str | None) -> str:
    if not iso_date:
        return "—"
    days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(iso_date).date()).days
    return "сегодня" if days <= 0 else f"{days} дн. назад"


def format_card(lead: dict) -> str:
    c, e = lead["context"], html.escape
    lines = [f"🏢 <b>{e(c['name'])}</b>" + (f" · {e(c['category'])}" if c.get("category") else "")]
    if c.get("address"):
        lines.append(f"📍 {e(c['address'])}")
    ig = f"📸 <a href=\"https://instagram.com/{e(c['instagram'])}\">@{e(c['instagram'])}</a>"
    if c.get("ig_checked"):
        ig += f" · {c['followers']} подписчиков · {c['posts']} постов · последний пост {_days_ago(c['last_post'])}"
        if c.get("first_post"):
            ig += f" · первый пост {e(c['first_post'])}"
    else:
        ig += " · ⚠️ активность не проверена"
    lines.append(ig)
    extra = []
    if c.get("website") and "instagram.com" not in c["website"]:
        extra.append(f"🌐 {e(c['website'])}")
    if c.get("rating"):
        extra.append(f"⭐ {c['rating']} ({c['google_reviews']})")
    if c.get("phone"):
        extra.append(f"📞 {e(c['phone'])}")
    if extra:
        lines.append(" · ".join(extra))
    if c.get("maps_url"):
        lines.append(f"🗺 <a href=\"{e(c['maps_url'])}\">Google Maps</a>")
    lines.append("")
    if lead.get("summary"):
        lines.append(f"🔍 <b>О компании:</b> {e(lead['summary'])}")
    lines += [
        f"🎯 <b>Оценка {lead['score']}/10.</b> {e(lead['reason'])}",
        f"🤖 <b>Что предложить:</b> {e(lead['idea'])}",
        "",
        "✉️ <b>Сообщение</b> (нажми, чтобы скопировать):",
        f"<code>{e(lead['message'])}</code>",
    ]
    if lead["status"] == "sent":
        lines += ["", f"✅ Отправлено — {e(lead['handled_by'] or '')}"]
    elif lead["status"] == "rejected":
        lines += ["", f"❌ Не подходит — {e(lead['handled_by'] or '')}"]
    return "\n".join(lines)


def card_keyboard(lead: dict) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text="✉️ Открыть Direct", url=f"https://ig.me/m/{lead['instagram']}")]]
    if lead["status"] == "new":
        rows.append([
            InlineKeyboardButton(text="✅ Отправил", callback_data=f"sent:{lead['id']}"),
            InlineKeyboardButton(text="🔄 Другой текст", callback_data=f"regen:{lead['id']}"),
            InlineKeyboardButton(text="❌ Не подходит", callback_data=f"skip:{lead['id']}"),
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send_card(bot: Bot, chat_id: int, lead_id: int) -> None:
    lead = db.lead(lead_id)
    await bot.send_message(
        chat_id, format_card(lead), reply_markup=card_keyboard(lead), disable_web_page_preview=True
    )


async def refresh_card(call: CallbackQuery, lead_id: int) -> None:
    lead = db.lead(lead_id)
    try:
        await call.message.edit_text(
            format_card(lead), reply_markup=card_keyboard(lead), disable_web_page_preview=True
        )
    except TelegramBadRequest:
        pass


# ---------- searching ----------

async def run(bot: Bot, chat_id: int, want: int) -> None:
    global cancel_requested
    if run_lock.locked():
        await bot.send_message(chat_id, "⏳ Поиск уже идёт. /stop — остановить.")
        return
    async with run_lock:
        cancel_requested = False
        status = await bot.send_message(chat_id, f"🔎 Ищу до {want} лидов… Это займёт 5–20 минут.")
        found = 0

        async def notify(text: str) -> None:
            await bot.send_message(chat_id, text, parse_mode=None)

        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                today = datetime.now(ZoneInfo(cfg.timezone)).date().isoformat()
                leads = pipeline.find_leads(cfg, db, session, want, lambda: cancel_requested, notify, today)
                async for lead in leads:
                    await send_card(bot, chat_id, lead.id)
                    found += 1
            result = f"✅ Готово: {found} лидов."
        except pipeline.Cancelled:
            result = f"⛔ Остановлено. Найдено лидов: {found}."
        except Exception as e:
            log.exception("search failed")
            result = f"❌ Ошибка после {found} лидов: {html.escape(str(e))[:500]}"
        try:
            await status.delete()
        except TelegramBadRequest:
            pass
        await bot.send_message(chat_id, result)


async def scheduler(bot: Bot) -> None:
    if not cfg.daily_time or not cfg.leads_chat_id:
        log.info("daily run disabled")
        return
    tz = ZoneInfo(cfg.timezone)
    hour, minute = map(int, cfg.daily_time.split(":"))
    while True:
        now = datetime.now(tz)
        nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        while nxt <= now or nxt.isoweekday() not in cfg.daily_weekdays:
            nxt += timedelta(days=1)
        log.info("next daily run at %s", nxt)
        await asyncio.sleep((nxt - now).total_seconds())
        try:
            await run(bot, cfg.leads_chat_id, cfg.daily_leads)
        except Exception:
            log.exception("daily run failed")


# ---------- access ----------

allowed = F.from_user.id.in_(cfg.allowed_users)
router.message.filter(allowed)
router.callback_query.filter(allowed)
denied = Router()


@denied.message(Command("start", "id"))
async def no_access(message: Message) -> None:
    await message.answer(
        f"⛔ Нет доступа. Ваш Telegram ID: <code>{message.from_user.id}</code>\n"
        f"ID этого чата: <code>{message.chat.id}</code>"
    )


# ---------- commands ----------

@router.message(CommandStart())
async def start(message: Message) -> None:
    schedule = f"по расписанию в {cfg.daily_time}" if cfg.daily_time else "по команде"
    await message.answer(
        "Я нахожу армянские бизнесы без своего веб-приложения, изучаю каждый и готовлю для него "
        "сообщение в Instagram от NetFactory.\n\n"
        f"Лиды приходят {schedule} (до {cfg.daily_leads} шт., в пределах бесплатного лимита Apify).\n"
        "Под каждым лидом: ✅ Отправил · 🔄 Другой текст · ❌ Не подходит.\n\n"
        "/leads — найти лиды сейчас (/leads 5 — пять штук)\n"
        "/stop — остановить поиск\n"
        "/stats — статистика\n"
        f"/id — ID этого чата (сейчас: <code>{message.chat.id}</code>)"
    )


@router.message(Command("id"))
async def chat_id(message: Message) -> None:
    await message.answer(f"ID этого чата: <code>{message.chat.id}</code>")


@router.message(Command("leads"))
async def leads_now(message: Message, command: CommandObject, bot: Bot) -> None:
    want = cfg.daily_leads
    if command.args:
        if not command.args.strip().isdigit() or not 1 <= int(command.args) <= 100:
            await message.answer("Укажи число от 1 до 100, например /leads 5")
            return
        want = int(command.args)
    task = asyncio.create_task(run(bot, message.chat.id, want))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


@router.message(Command("stop"))
async def stop(message: Message) -> None:
    global cancel_requested
    if not run_lock.locked():
        await message.answer("Сейчас ничего не ищу.")
        return
    cancel_requested = True
    await message.answer("⛔ Останавливаю после текущей компании…")


@router.message(Command("stats"))
async def stats(message: Message) -> None:
    midnight = datetime.now(ZoneInfo(cfg.timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
    s = db.stats(midnight.astimezone(timezone.utc).isoformat(timespec="seconds"))
    leads = s["leads"]
    async with aiohttp.ClientSession(trust_env=True) as session:
        used = await places.monthly_usage(session, cfg.apify_token)
    apify = f"Apify: сегодня {db.places_bought(midnight.date().isoformat())}/{cfg.places_per_day} компаний"
    if used is not None:
        apify += f", за месяц ${used:.2f} из ${cfg.apify_monthly_budget:.2f}"
    lines = [
        f"📊 <b>Сегодня:</b> найдено {s['today']}, отправлено {s['sent_today']}",
        apify,
        f"<b>Всего лидов:</b> {sum(leads.values())} — отправлено {leads.get('sent', 0)}, "
        f"ждут {leads.get('new', 0)}, отклонено {leads.get('rejected', 0)}",
        f"<b>Проверено компаний:</b> {sum(s['places'].values())}",
    ]
    for key, count in sorted(s["places"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  • {PLACE_STATUSES.get(key, key)}: {count}")
    await message.answer("\n".join(lines))


# ---------- card buttons ----------

def _lead_from(call: CallbackQuery) -> dict | None:
    return db.lead(int(call.data.split(":")[1]))


@router.callback_query(F.data.startswith("sent:") | F.data.startswith("skip:"))
async def on_status(call: CallbackQuery) -> None:
    lead = _lead_from(call)
    if not lead:
        await call.answer("Лид не найден", show_alert=True)
        return
    if lead["status"] != "new":
        await call.answer(f"Уже отмечено: {lead['handled_by']}", show_alert=True)
    else:
        db.set_status(lead["id"], "sent" if call.data.startswith("sent:") else "rejected", _who(call.from_user))
        await call.answer("Отмечено")
    await refresh_card(call, lead["id"])


@router.callback_query(F.data.startswith("regen:"))
async def on_regen(call: CallbackQuery) -> None:
    lead = _lead_from(call)
    if not lead or lead["status"] != "new":
        await call.answer("Этот лид уже обработан", show_alert=True)
        return
    await call.answer("Пишу другой вариант…")
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            verdict = await llm.rewrite(
                session, cfg.gemini_api_key, cfg.gemini_models, lead["context"],
                datetime.now().date().isoformat(), lead["message"],
            )
    except Exception as e:
        log.exception("rewrite failed")
        await call.message.answer(f"❌ Не получилось переписать: {html.escape(str(e))[:300]}")
        return
    db.set_message(lead["id"], verdict.message, verdict.idea or lead["idea"])
    await refresh_card(call, lead["id"])


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    missing = [k for k, v in (("APIFY_TOKEN", cfg.apify_token), ("GEMINI_API_KEY", cfg.gemini_api_key),
                              ("ALLOWED_USERS", cfg.allowed_users)) if not v]
    if missing:
        raise SystemExit(f"Заполните в .env: {', '.join(missing)}")
    db.sync_queries(all_queries())
    bot = Bot(cfg.bot_token, default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher()
    dp.include_routers(router, denied)
    await bot.set_my_commands([
        BotCommand(command="leads", description="Найти лиды сейчас"),
        BotCommand(command="stop", description="Остановить поиск"),
        BotCommand(command="stats", description="Статистика"),
        BotCommand(command="start", description="Как пользоваться"),
    ])
    scheduler_task = asyncio.create_task(scheduler(bot))
    try:
        await dp.start_polling(bot)
    finally:
        scheduler_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
