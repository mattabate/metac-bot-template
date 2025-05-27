import asyncio
from pyppeteer import launch


async def call_research(question: str) -> str:
    browser = await launch(headless=False)
    page = await browser.newPage()
    await page.goto("https://research.joseph.ma/")

    await asyncio.sleep(3)
    await page.waitForSelector('form textarea[name="prompt"]', timeout=10000)
    await page.type('form textarea[name="prompt"]', question, {"delay": 50})

    while True:
        button_disabled = await page.querySelectorEval(
            'form button[aria-label="Start Research"]', "(btn) => btn.disabled"
        )
        if not button_disabled:
            break
        await asyncio.sleep(0.5)

    await page.click('form button[aria-label="Start Research"]')

    print("Waiting for the response status to be 'Done!'...")
    status_selector = 'div[role="button"] p.text-sm.font-medium'

    while True:
        try:
            await page.waitForSelector(status_selector, timeout=5)
            status_text = await page.evaluate(
                f"""
                () => {{
                    const el = document.querySelector('{status_selector}');
                    return el ? el.textContent : null;
                }}
            """
            )
            if status_text:
                print(f"Current status: {status_text.strip()}")
                if "Done!" in status_text:
                    break
        except asyncio.TimeoutError:
            print("Status element not found, retrying...")
        await asyncio.sleep(1)

    print("Status is Done! Waiting 5 seconds for the final response...")
    await asyncio.sleep(5)

    response_selector = (
        "body > div > div.flex.flex-col.w-full.md\\:w-1\\/2.space-y-4.flex-shrink-0 > "
        "div.flex.flex-col.border.border-gray-200.rounded-lg.shadow-md.min-h-0.flex-1.transition-all.duration-300.ease-in-out > "
        "div.flex-grow.overflow-auto.p-4.min-h-0"
    )
    await page.waitForSelector(response_selector, timeout=10000)
    response_text = await page.querySelectorEval(
        response_selector, "(el) => el.innerText"
    )

    print("Extracted Response:\n")
    print("\033[93m" + response_text + "\033[0m")

    # Cleanly close the browser, catch any stray errors
    try:
        await browser.close()
    except Exception as e:
        print(f"Warning: Error during browser closure: {e}")

    return response_text


if __name__ == "__main__":
    question = """Before 2030, how many new AI labs will be leading labs within 2 years of their founding?
The options are: ['0 or 1', '2 or 3', '4 or 5', '6 or 7', '8 or 9', '10 or more']"""
    question = question.strip().replace("\n", "  ")
    result = asyncio.get_event_loop().run_until_complete(call_research(question))
    print("\nResearch Joseph's Final Response:\n", result)
