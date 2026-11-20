"""Take real screenshots of the running llm-gateway dashboard.

Usage (gateway must be running, e.g. LLMGW_PORT=8005 ./run.sh):
    pip install playwright && playwright install chromium
    python scripts/screenshot.py [base-url]     # default http://127.0.0.1:8005/

Writes docs/screenshots/dashboard.png and docs/screenshots/dashboard-playground.png
into the repo. The playground shot clicks "Ask the model" for real, so
the answer in it came from the local Ollama model, not from a mock.
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8005/"


async def main() -> int:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 1000})
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(6000)  # let the 5s /stats refresh fill in
        total = await page.text_content("#total")
        print("dashboard total_requests shown:", total)
        await page.screenshot(path=str(ROOT / "docs" / "screenshots" / "dashboard.png"),
                              full_page=True)
        print("wrote docs/screenshots/dashboard.png")
        await page.click("#askBtn")
        try:
            await page.wait_for_function(
                "document.getElementById('answer').textContent.trim() !== '…' "
                "&& !document.getElementById('answer').textContent.startsWith('The answer will appear')",
                timeout=120000,
            )
        except Exception as exc:  # noqa: BLE001 - report, keep first shot
            print("playground answer did not arrive:", exc)
            await browser.close()
            return 1
        await page.wait_for_timeout(6000)  # stats refresh picks up the new row
        print("playground answer:",
              (await page.text_content("#answer"))[:120].replace("\n", " "))
        print("playground meta:", await page.text_content("#askMeta"))
        await page.screenshot(path=str(ROOT / "docs" / "screenshots" / "dashboard-playground.png"),
                              full_page=True)
        print("wrote docs/screenshots/dashboard-playground.png")
        await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
