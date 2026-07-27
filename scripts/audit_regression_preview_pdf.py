"""Extract text and render page images from the regression preview PDF."""

import argparse
import logging
from pathlib import Path
import sys

import pdfplumber
import pymupdf


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EXERCISE_DIRECTORY = PROJECT_DIRECTORY / "D - ML" / "regression" / "exercise"
sys.path[0:0] = [EXERCISE_DIRECTORY.as_posix()]

import regression_init


LOGGER = logging.getLogger(name=__name__)

argument_parser = argparse.ArgumentParser(
    description="Extract and render a notebook preview PDF for auditing.",
)
argument_parser.add_argument(
    "--input-pdf",
    required=True,
    type=Path,
    help="PDF preview to inspect.",
)
argument_parser.add_argument(
    "--text-output",
    required=True,
    type=Path,
    help="Path for page-delimited extracted text.",
)
argument_parser.add_argument(
    "--page-output-directory",
    required=True,
    type=Path,
    help="Directory for separately rendered high-resolution page PNGs.",
)
arguments = argument_parser.parse_args()

input_pdf_path = arguments.input_pdf.resolve()
text_output_path = arguments.text_output.resolve()
page_output_directory = arguments.page_output_directory.resolve()
if not input_pdf_path.is_file():
    error_message = f"Input PDF does not exist: {input_pdf_path}"
    LOGGER.error(msg=error_message)
    raise FileNotFoundError(error_message)

text_output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)
page_output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

extracted_page_sections: list[str] = []
# pdfplumber defines the document input as positional-only.
with pdfplumber.open(input_pdf_path) as pdf_document:
    for page_number, pdf_page in enumerate(pdf_document.pages, start=1):
        extracted_page_text = pdf_page.extract_text(
            x_tolerance=2,
            y_tolerance=3,
        )
        extracted_page_sections += [
            f"===== PAGE {page_number} =====\n{extracted_page_text or ''}"
        ]

text_output_path.write_text(
    data="\n\n".join(extracted_page_sections),
    encoding="utf-8",
)

rendered_page_paths: list[Path] = []
pdf_render_document = pymupdf.open(filename=input_pdf_path)
render_matrix = pymupdf.Matrix(3.0, 3.0)
for page_index in range(pdf_render_document.page_count):
    rendered_page = pdf_render_document.load_page(page_id=page_index)
    page_pixmap = rendered_page.get_pixmap(
        matrix=render_matrix,
        alpha=False,
    )
    rendered_page_path = (
        page_output_directory / f"page_{page_index + 1:03d}.png"
    )
    page_pixmap.save(filename=rendered_page_path)
    rendered_page_paths += [rendered_page_path]
pdf_render_document.close()

LOGGER.info(
    msg=(
        f"Audited and separately rendered "
        f"{rendered_page_paths.__len__()} PDF pages; "
        f"text: {text_output_path}; pages: {page_output_directory}"
    )
)
