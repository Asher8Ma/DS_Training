"""Render an exported notebook HTML preview to PDF with Playwright."""

import argparse
import logging
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EXERCISE_DIRECTORY = PROJECT_DIRECTORY / "D - ML" / "regression" / "exercise"
sys.path[0:0] = [EXERCISE_DIRECTORY.as_posix()]

import regression_init


LOGGER = logging.getLogger(name=__name__)
PRINT_STYLES = """
@media print {
  .anchor-link {
    display: none !important;
  }

  h1,
  h2,
  h3,
  h4 {
    break-after: avoid-page;
    break-inside: avoid-page;
    page-break-after: avoid;
    page-break-inside: avoid;
  }

  .jp-MarkdownCell:has(
    h3[id="Bonus-solution:-quadratic-regression"]
  ) {
    break-before: page;
    page-break-before: always;
  }

  .jp-RenderedHTMLCommon p:has(+ p > mjx-container[display="true"]) {
    break-after: avoid-page;
    page-break-after: avoid;
  }

  .jp-RenderedHTMLCommon p:has(> mjx-container[display="true"]) {
    break-inside: avoid-page;
    page-break-inside: avoid;
  }

  #Working-on-a-real-life-data {
    break-after: avoid-page;
    page-break-after: avoid;
  }

  @supports not (break-after: avoid-page) {
    #Working-on-a-real-life-data {
      break-before: page;
      page-break-before: always;
    }
  }

  table.dataframe {
    break-inside: avoid;
    font-size: 9pt;
    max-width: 100%;
  }

  table.dataframe th,
  table.dataframe td {
    overflow-wrap: anywhere;
    white-space: normal !important;
  }

  .jp-OutputArea-output img {
    height: auto !important;
    max-width: 100% !important;
  }
}
"""

argument_parser = argparse.ArgumentParser(
    description="Render a Jupyter HTML preview into a print-quality PDF.",
)
argument_parser.add_argument(
    "--input-html",
    required=True,
    type=Path,
    help="Path to the exported notebook HTML.",
)
argument_parser.add_argument(
    "--output-pdf",
    required=True,
    type=Path,
    help="Path for the generated PDF.",
)
arguments = argument_parser.parse_args()

input_html_path = arguments.input_html.resolve()
output_pdf_path = arguments.output_pdf.resolve()
if not input_html_path.is_file():
    error_message = f"Input HTML does not exist: {input_html_path}"
    LOGGER.error(msg=error_message)
    raise FileNotFoundError(error_message)

output_pdf_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with sync_playwright() as playwright_instance:
    chromium_browser = playwright_instance.chromium.launch(
        headless=True,
        args=["--ignore-certificate-errors"],
    )
    browser_page = chromium_browser.new_page()
    browser_page.goto(
        url=input_html_path.as_uri(),
        wait_until="networkidle",
        timeout=120_000,
    )
    browser_page.add_style_tag(content=PRINT_STYLES)
    browser_page.emulate_media(media="print")
    browser_page.pdf(
        path=output_pdf_path.as_posix(),
        format="A4",
        print_background=True,
        margin={
            "top": "12mm",
            "right": "12mm",
            "bottom": "12mm",
            "left": "12mm",
        },
    )
    chromium_browser.close()

LOGGER.info(msg=f"Rendered notebook preview PDF: {output_pdf_path}")
