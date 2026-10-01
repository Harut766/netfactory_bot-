"""Gemini: decides whether a business fits NetFactory and writes the Instagram DM."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """\
Ты — менеджер по продажам компании NetFactory (Ереван, netfactory.tech).

ЧТО ДЕЛАЕТ NETFACTORY
Мы строим ИИ-агентов, которые делают повторяющуюся работу бизнеса, и подключаем их к инструментам,
которые у клиента уже есть. Типовые задачи:
- Ответ на заявки: каждая заявка (сайт, Instagram, WhatsApp, почта) получает ответ за минуту,
  клиент квалифицируется и записывается в календарь — ночью и в выходные тоже.
- Сортировка обращений поддержки: повторяющиеся вопросы закрываются автоматически, остальное
  уходит нужному человеку с кратким резюме.
- Счета и документы: счета, договоры, накладные читаются автоматически, данные попадают в учётную систему.
- Ответы по внутренним документам: сотрудники спрашивают обычным языком — ответ из договоров,
  прайсов и регламентов компании.
- Еженедельный отчёт: цифры из всех систем собираются в один отчёт автоматически.
- Записи и напоминания: запись, подтверждение, перенос и напоминания без звонков, меньше неявок.
- Порядок в CRM: карточки клиентов создаются и обновляются сами из почты, звонков и форм.
- Мультиязычный контент: описания товаров, карточки и ответы на армянском, русском и английском.
Также делаем веб- и мобильные приложения вокруг автоматизации (кабинеты клиентов, дашборды).
Первый агент запускается за 3–6 недель, цена фиксируется до начала работ.
Первый шаг для клиента — бесплатный 30-минутный звонок, где мы честно говорим, что стоит автоматизировать.

КТО НАШ КЛИЕНТ
Почти любой армянский бизнес: у всех есть рутина, которую можно отдать агенту. Ниша любая.
НЕ подходит только бизнес, у которого УЖЕ есть своё веб-приложение: личный кабинет клиента, собственная
онлайн-запись или интернет-магазин с корзиной на своём сайте, своё мобильное приложение. Такие компании уже
цифровые. Признаки — в поле site.web_app_signals и в тексте сайта, но проверь сам: ссылка «Вход» на
шаблонном сайте или чужой виджет — ещё не своё приложение. Простой сайт-визитка, Instagram, Google-форма,
запись через WhatsApp или Direct — это НЕ веб-приложение, такие компании подходят.
Также fit=false, если это не бизнес (личный блог) или бизнес не в Армении.

ИССЛЕДОВАНИЕ
Перед сообщением разберись в компании по всем данным: чем занимается, для кого, как к ним приходят клиенты
(запись, заказы, доставка, звонки, Direct), какая рутина у них наверняка съедает время.

СООБЩЕНИЕ В DIRECT
- Язык: армянский (восточноармянский), обращение на «Դուք». Грамотно и естественно, не машинный перевод.
- Тон: деловой, спокойный, конкретный — как сайт NetFactory. Без восклицательных знаков подряд, без
  эмодзи, без «уникальных решений» и прочих штампов, без давления.
- Длина: 350–650 символов, 3–5 коротких предложений.
- Структура: короткое приветствие → одна конкретная деталь об ИХ бизнесе (из данных ниже) → одна
  конкретная рутина, которую у них, скорее всего, можно отдать агенту, и что это даст → мягкое
  предложение бесплатного 30-минутного звонка → подпись «NetFactory» (можно с netfactory.tech).
- Не выдумывай факты, которых нет в данных. Не называй цены. Не пиши, что следил за ними.

ФОРМАТ ОТВЕТА — строго JSON:
{"fit": true|false, "has_web_app": true|false, "score": 1-10 (насколько горячий клиент),
 "summary": "исследование: чем занимается компания, для кого, как работают с клиентами — 2-3 предложения по-русски",
 "web_presence": "что у компании есть онлайн: нет сайта / сайт-визитка / онлайн-запись через чужой сервис /
   свой интернет-магазин или кабинет и т.п., и есть ли своё веб-приложение — 1 короткое предложение по-русски",
 "reason": "почему подходит/не подходит, 1 предложение по-русски",
 "idea": "что именно им автоматизировать в первую очередь, 1 предложение по-русски",
 "message": "текст DM на армянском"}
message пиши ВСЕГДА, даже если fit=false: команда сама решит, писать ли. Если instagram = null, сообщение
отправят в WhatsApp — текст тот же."""


class LLMError(Exception):
    pass


@dataclass
class Verdict:
    fit: bool
    score: int
    reason: str
    idea: str
    message: str
    summary: str = ""
    has_web_app: bool = False
    web_presence: str = ""


def parse_verdict(text: str) -> Verdict:
    text = text.strip()
    # Models sometimes wrap JSON in a code fence despite responseMimeType.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON: {text[:200]}") from e
    if not isinstance(data, dict):
        raise LLMError(f"model returned {type(data).__name__} instead of an object")
    try:
        score = int(data.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    verdict = Verdict(
        fit=bool(data.get("fit")),
        score=max(0, min(10, score)),
        reason=str(data.get("reason") or "").strip(),
        idea=str(data.get("idea") or "").strip(),
        message=str(data.get("message") or "").strip(),
        summary=str(data.get("summary") or "").strip(),
        has_web_app=bool(data.get("has_web_app")),
        web_presence=str(data.get("web_presence") or "").strip(),
    )
    if not verdict.message:
        raise LLMError("model wrote no message")
    return verdict


def _prompt(context: dict, today: str) -> str:
    return (
        f"Сегодня {today}. Данные о бизнесе (Google Maps, сайт, Instagram):\n"
        f"{json.dumps(context, ensure_ascii=False, indent=1)}\n\n"
        "Исследуй компанию, оцени, подходит ли она NetFactory, и напиши сообщение."
    )


async def _generate(session: aiohttp.ClientSession, api_key: str, models: tuple[str, ...], prompt: str,
                    temperature: float) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    errors = []
    for model in models:
        for attempt in range(3):
            try:
                async with session.post(
                    API_URL.format(model=model), json=body, headers={"x-goog-api-key": api_key},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as r:
                    data = await r.json(content_type=None)
                    status = r.status
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                errors.append(f"{model}: {e!r}")
                await asyncio.sleep(2 ** attempt)
                continue
            if status == 200:
                parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if text:
                    return text
                errors.append(f"{model}: empty answer")
                break
            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else data
            errors.append(f"{model} {status}: {msg}")
            if status in (429, 500, 502, 503, 504):
                await asyncio.sleep(5 * 2 ** attempt)
                continue
            # 400/404 etc.: retrying the same model won't help, try the next one.
            break
    raise LLMError("; ".join(errors[-3:]) or "no Gemini models configured")


async def evaluate(session: aiohttp.ClientSession, api_key: str, models: tuple[str, ...], context: dict,
                   today: str) -> Verdict:
    text = await _generate(session, api_key, models, _prompt(context, today), temperature=0.7)
    return parse_verdict(text)


async def rewrite(session: aiohttp.ClientSession, api_key: str, models: tuple[str, ...], context: dict,
                  today: str, previous: str) -> Verdict:
    prompt = (
        _prompt(context, today)
        + "\n\nПредыдущий вариант сообщения не понравился:\n"
        + previous
        + "\n\nНапиши заметно другой вариант: другой заход и другая идея автоматизации."
    )
    return parse_verdict(await _generate(session, api_key, models, prompt, temperature=1.0))
