import asyncio
from pyppeteer import launch


async def call_research(question: str) -> str:
    browser = None
    try:
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.goto("https://research.joseph.ma/")

        await asyncio.sleep(3)
        await page.waitForSelector('form textarea[name="prompt"]', timeout=90000)
        await page.type('form textarea[name="prompt"]', question, {"delay": 50})

        # Wait for button to be enabled with timeout
        button_wait_start = asyncio.get_event_loop().time()
        while True:
            button_disabled = await page.querySelectorEval(
                'form button[aria-label="Start Research"]', "(btn) => btn.disabled"
            )
            if not button_disabled:
                break

            # Add timeout for button wait
            if asyncio.get_event_loop().time() - button_wait_start > 60:
                raise TimeoutError("Button never became enabled")

            await asyncio.sleep(0.5)

        await page.click('form button[aria-label="Start Research"]')

        status_selector = 'div[role="button"] p.text-sm.font-medium'

        # Wait for completion with periodic status updates
        research_start_time = asyncio.get_event_loop().time()
        last_status_update = research_start_time

        while True:
            try:
                await page.waitForSelector(status_selector, timeout=10)
                status_text = await page.evaluate(
                    f"""
                    () => {{
                        const el = document.querySelector('{status_selector}');
                        return el ? el.textContent : null;
                    }}
                """
                )

                current_time = asyncio.get_event_loop().time()

                # Print status updates every 2 minutes
                if current_time - last_status_update > 120:
                    elapsed_minutes = int((current_time - research_start_time) / 60)
                    print(
                        f"Research in progress... Status: {status_text} (elapsed: {elapsed_minutes} minutes)"
                    )
                    last_status_update = current_time

                if status_text and "Done!" in status_text:
                    break

            except asyncio.TimeoutError:
                # Continue if selector not found
                pass

            await asyncio.sleep(2)

        await asyncio.sleep(5)

        response_selector = (
            "body > div > div.flex.flex-col.w-full.md\\:w-1\\/2.space-y-4.flex-shrink-0 > "
            "div.flex.flex-col.border.border-gray-200.rounded-lg.shadow-md.min-h-0.flex-1.transition-all.duration-300.ease-in-out > "
            "div.flex-grow.overflow-auto.p-4.min-h-0"
        )
        await page.waitForSelector(response_selector, timeout=90000)
        response_text = await page.querySelectorEval(
            response_selector, "(el) => el.innerText"
        )

        return response_text

    except Exception as e:
        print(f"Error in call_research: {e}")
        raise
    finally:
        # Always try to close the browser
        if browser:
            try:
                await browser.close()
                print("Browser closed successfully")
            except Exception as e:
                print(f"Error closing browser: {e}")
                # Force kill any remaining browser processes
                try:
                    import psutil

                    for proc in psutil.process_iter(["pid", "name"]):
                        if (
                            "chrome" in proc.info["name"].lower()
                            or "chromium" in proc.info["name"].lower()
                        ):
                            try:
                                proc.kill()
                            except:
                                pass
                except ImportError:
                    print("psutil not available for process cleanup")
                except Exception as cleanup_error:
                    print(f"Error during process cleanup: {cleanup_error}")


async def async_call_research(question: str):
    question = question.strip().replace("\n", "  ")
    return await call_research(question)
