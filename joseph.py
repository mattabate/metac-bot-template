import asyncio
import dotenv
import os

from browserbase import Browserbase
from playwright.async_api import async_playwright

# Load environment variables from .env file
dotenv.load_dotenv()
BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID")


async def call_research(question: str) -> str:
    bb = Browserbase(api_key=BROWSERBASE_API_KEY)
    session = bb.sessions.create(project_id=BROWSERBASE_PROJECT_ID)
    connect_url = session.connect_url  # Get the connection URL for Playwright

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(connect_url)
        context = (
            browser.contexts[0] if browser.contexts else await browser.new_context()
        )
        page = context.pages[0] if context.pages else await context.new_page()

        try:
            await page.goto("https://research.joseph.ma/")
            await asyncio.sleep(3)
            await page.wait_for_selector('form textarea[name="prompt"]', timeout=90000)
            await page.fill('form textarea[name="prompt"]', question)

            # Wait for button to be enabled with timeout
            button_wait_start = asyncio.get_event_loop().time()
            while True:
                button_disabled = await page.eval_on_selector(
                    'form button[aria-label="Start Research"]', "btn => btn.disabled"
                )
                if not button_disabled:
                    break

                if asyncio.get_event_loop().time() - button_wait_start > 60:
                    raise TimeoutError("Button never became enabled")
                await asyncio.sleep(0.5)

            await page.click('form button[aria-label="Start Research"]')

            status_selector = 'div[role="button"] p.text-sm.font-medium'
            research_start_time = asyncio.get_event_loop().time()
            last_status_update = research_start_time

            while True:
                try:
                    await page.wait_for_selector(status_selector, timeout=10000)
                    status_text = await page.eval_on_selector(
                        status_selector, "el => el.textContent"
                    )

                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_status_update > 120:
                        elapsed_minutes = int((current_time - research_start_time) / 60)
                        print(
                            f"Research in progress... Status: {status_text} (elapsed: {elapsed_minutes} minutes)"
                        )
                        last_status_update = current_time

                    if status_text and "Done!" in status_text:
                        break

                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(2)

            await asyncio.sleep(5)

            response_selector = (
                "body > div > div.flex.flex-col.w-full.md\\:w-1\\/2.space-y-4.flex-shrink-0 > "
                "div.flex.flex-col.border.border-gray-200.rounded-lg.shadow-md.min-h-0.flex-1.transition-all.duration-300.ease-in-out > "
                "div.flex-grow.overflow-auto.p-4.min-h-0"
            )
            await page.wait_for_selector(response_selector, timeout=90000)
            response_text = await page.eval_on_selector(
                response_selector, "el => el.innerText"
            )

            return response_text

        finally:
            await browser.close()
            print("Browser closed successfully")


async def async_call_research(question: str):
    question = question.strip().replace("\n", "  ")
    return await call_research(question)
