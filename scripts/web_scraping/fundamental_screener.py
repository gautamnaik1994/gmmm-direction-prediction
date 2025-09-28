
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync


def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    stealth_sync(context)
    page = context.new_page()
    page.goto("https://www.screener.in/")
    page.wait_for_timeout(5000)  # Wait for 5 seconds
    page.screenshot(path="screener.png")
    # context.close()
    # browser.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
