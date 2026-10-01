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

ИДЕАЛЬНЫЙ КЛИЕНТ
Армянский бизнес, новый и активный: недавно открылся или растёт, активно ведёт Instagram, к нему идёт
поток клиентов и заявок, а значит, есть рутина, которую можно автоматизировать. Ниша любая.
Не подходят: крупные корпорации и госструктуры, IT/маркетинговые агентства-конкуренты, неактивные
и заброшенные аккаунты, личные блоги, аккаунты не из Армении.

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
{"fit": true|false, "score": 1-10, "reason": "почему подходит/не подходит, 1-2 предложения по-русски",
 "idea": "что именно им автоматизировать, 1 предложение по-русски", "message": "текст DM на армянском"}
Если fit=false, message можно оставить пустым."""


class LLMError(Exception):
    pass


@dataclass
class Verdict:
    fit: bool
    score: int
    reason: str
    idea: str
    message: str


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
    )
    if verdict.fit and not verdict.message:
        raise LLMError("model accepted the lead but wrote no message")
    return verdict


def _prompt(context: dict, today: str) -> str:
    return (
        f"Сегодня {today}. Данные о бизнесе (Google Maps, сайт, Instagram):\n"
        f"{json.dumps(context, ensure_ascii=False, indent=1)}\n\n"
        "Оцени, подходит ли он NetFactory, и напиши сообщение."
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
        + "\n\nБизнес уже признан подходящим (fit=true). Предыдущий вариант сообщения не понравился:\n"
        + previous
        + "\n\nНапиши заметно другой вариант: другой заход и другая идея автоматизации."
    )
    verdict = parse_verdict(await _generate(session, api_key, models, prompt, temperature=1.0))
    if not verdict.message:
        raise LLMError("model wrote no message")
    return verdict
