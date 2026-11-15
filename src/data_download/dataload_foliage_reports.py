import os
import re
import sys
import html
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

# Maine DACF fall foliage report archive, see data/README.md
# the individual weekly reports on www.maine.gov refuse scripted requests, so they are saved by hand (MANUAL_STEPS.md)
ARCHIVE_URL = "https://www1.maine.gov/dacf/mfs/projects/fall_foliage/report/past.shtml"
YEARS = [2024, 2025, 2026]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "maine_foliage_reports")
ARCHIVE_HTML = os.path.join(OUT_DIR, "past.shtml.html")
REPORTS_TSV = os.path.join(OUT_DIR, "report_urls.tsv")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)
    response = requests.get(ARCHIVE_URL, timeout=120)
    response.raise_for_status()
    with open(ARCHIVE_HTML, "wb") as f:
        f.write(response.content)
    logger.info("saved %s, %d bytes", ARCHIVE_HTML, len(response.content))

    text = response.text
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for year in YEARS:
        # each year has a section with id reportsYYYY that ends where the next section begins
        start = text.find(f'id="reports{year}"')
        assert start >= 0, f"no section for {year}"
        end = text.find('id="reports', start + 10)
        links = re.findall(r'href="(https://www\.maine\.gov/tools/whatsnew[^"]+)"[^>]*>([^<]+)<', text[start:end])
        for url, label in links:
            report_date = datetime.strptime(html.unescape(label).strip(), "%B %d, %Y").date()
            rows.append((report_date.isoformat(), html.unescape(url)))
        logger.info("%d: %d reports", year, len(links))

    with open(REPORTS_TSV, "w") as f:
        f.write("report_date\turl\tlocal_file\tlisted_utc\n")
        for report_date, url in sorted(rows):
            f.write(f"{report_date}\t{url}\t{report_date}.pdf\t{now}\n")
    logger.info("wrote %s with %d reports", REPORTS_TSV, len(rows))


if __name__ == "__main__":
    main()
