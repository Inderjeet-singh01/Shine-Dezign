import requests
from bs4 import BeautifulSoup
from app.core.logger import logger

''' Web scraper for single static web page '''

# def scrape_website(url: str) -> str:
#     """Scrape readable text from a single static web page."""
#     logger.info("Scraping website: %s", url)

#     try:
#         # response represents the HTML content of the page
#         response = requests.get(
#             url,
#             timeout=20,
#             # Use a user-agent header to mimic a real browser and avoid blocking
#             headers={"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"},
#         )
#         response.raise_for_status()

#         # Use BeautifulSoup to parse the HTML content
#         soup = BeautifulSoup(response.text, "html.parser")

#         # Remove script, style, nav, footer, header, noscript, and svg elements
#         for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
#             tag.decompose()

#         # Get the text content of the page
#         text = soup.get_text(separator=" ", strip=True)
#         cleaned_text = " ".join(text.split())

#         if not cleaned_text:
#             return "No readable text found on the website."

#         logger.info("Website scraped successfully. Characters: %s", len(cleaned_text))
#         return cleaned_text[:12000]

#     except requests.RequestException as e:
#         logger.error("Website request failed: %s", str(e))
#         return "Website scraping failed because the page could not be requested."
#     except Exception as e:
#         logger.error("Website scraping failed: %s", str(e))
#         return "Website scraping failed."



''' Updated scraper with Main page + internal sub-pages scrape '''

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from app.core.logger import logger


def is_internal_url(base_url: str, link: str) -> bool:
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link).netloc
    return base_domain == link_domain


def extract_links(base_url: str, soup: BeautifulSoup) -> list[str]:
    links = []

    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a_tag["href"])

        if is_internal_url(base_url, full_url):
            links.append(full_url)

    return list(set(links))


def clean_html(html: str) -> tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    cleaned_text = " ".join(text.split())

    return cleaned_text, soup


def scrape_single_page(url: str) -> tuple[str, BeautifulSoup | None]:
    try:
        logger.info("Scraping page: %s", url)

        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"},
        )
        response.raise_for_status()

        return clean_html(response.text)

    except requests.RequestException as e:
        logger.error("Page request failed: %s", str(e))
        return "", None


def scrape_website(url: str, max_pages: int = 5) -> str:
    """
    Scrape main page + internal sub-pages.
    max_pages controls how many pages will be scraped.
    """
    visited = set()
    pages_to_visit = [url]
    all_text = []

    while pages_to_visit and len(visited) < max_pages:
        current_url = pages_to_visit.pop(0)

        if current_url in visited:
            continue

        text, soup = scrape_single_page(current_url)
        visited.add(current_url)

        if text:
            all_text.append(f"\n\n--- Page: {current_url} ---\n{text}")

        if soup:
            links = extract_links(url, soup)

            for link in links:
                if link not in visited and link not in pages_to_visit:
                    pages_to_visit.append(link)

    final_text = " ".join(all_text)

    if not final_text:
        return "No readable text found on the website."

    logger.info("Multi-page scraping completed. Pages scraped: %s", len(visited))

    return final_text[:12000]