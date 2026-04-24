import httpx
import os
from axon.config.settings import settings

PINCHTAB_BASE = settings.pinchtab_url
PINCHTAB_TOKEN = settings.pinchtab_token or os.environ.get("PINCHTAB_TOKEN")


def get_headers() -> dict:
    if PINCHTAB_TOKEN:
        return {"Authorization": f"Bearer {PINCHTAB_TOKEN}"}
    return {}


async def check_pinchtab() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{PINCHTAB_BASE}/health", headers=get_headers(), timeout=2
            )
            return r.status_code == 200
    except Exception:
        return False


async def browse_text(url: str, max_tokens: int = 4000) -> str:
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{PINCHTAB_BASE}/navigate",
            json={"url": url},
            headers=get_headers(),
            timeout=30,
        )
        r = await client.get(f"{PINCHTAB_BASE}/text", headers=get_headers(), timeout=30)
        text = r.text
        return text[: max_tokens * 4]


async def browse_snapshot(url: str) -> str:
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{PINCHTAB_BASE}/navigate",
            json={"url": url},
            headers=get_headers(),
            timeout=30,
        )
        r = await client.get(
            f"{PINCHTAB_BASE}/snapshot?filter=interactive&format=compact",
            headers=get_headers(),
            timeout=30,
        )
        return r.text


async def browse_click(ref: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{PINCHTAB_BASE}/action",
            json={"kind": "click", "ref": ref},
            headers=get_headers(),
            timeout=15,
        )
        return f"✓ Clicked element {ref}"


async def browse_fill(ref: str, value: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{PINCHTAB_BASE}/action",
            json={"kind": "fill", "ref": ref, "value": value},
            headers=get_headers(),
            timeout=15,
        )
        return f"✓ Filled {ref}"


async def browse_screenshot() -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{PINCHTAB_BASE}/screenshot", headers=get_headers(), timeout=15
        )
        from datetime import datetime

        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, "wb") as f:
            f.write(r.content)
        return f"✓ Screenshot saved: {filepath}"
