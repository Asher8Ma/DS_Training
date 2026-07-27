"""Structural and rendering-focused tests for the regression notebook."""

from pathlib import Path

import nbformat


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_DIRECTORY
    / "D - ML"
    / "regression"
    / "exercise"
    / "regression.ipynb"
)
PDF_RENDER_SCRIPT_PATH = (
    PROJECT_DIRECTORY / "scripts" / "render_notebook_preview_pdf.py"
)


def _load_original_cells() -> list[nbformat.NotebookNode]:
    """Return the original exercise cells in their stable source order."""
    with NOTEBOOK_PATH.open(mode="r", encoding="utf-8") as notebook_input:
        regression_notebook = nbformat.read(
            fp=notebook_input,
            as_version=4,
        )

    original_cells = [
        current_cell
        for current_cell in regression_notebook.cells
        if "exercise_cell_index" in current_cell.metadata
    ]
    original_cells.sort(
        key=lambda current_cell: current_cell.metadata["exercise_cell_index"],
    )
    return original_cells


def test_every_solution_has_three_approaches_and_a_direct_answer() -> None:
    """Each generated answer must document alternatives and answer explicitly."""
    with NOTEBOOK_PATH.open(mode="r", encoding="utf-8") as notebook_input:
        regression_notebook = nbformat.read(
            fp=notebook_input,
            as_version=4,
        )

    generated_solution_cells = [
        current_cell
        for current_cell in regression_notebook.cells
        if current_cell.metadata.get("generated_solution", False)
    ]

    assert generated_solution_cells.__len__() == 18
    for solution_cell in generated_solution_cells:
        solution_source = solution_cell.source
        assert "**Three approaches considered**" in solution_source
        assert "\n1. " in solution_source
        assert "\n2. " in solution_source
        assert "\n3. " in solution_source
        assert "**Choice:**" in solution_source
        assert "**Direct answer:**" in solution_source
        assert solution_source.count("$") % 2 == 0


def test_knn_question_and_wide_outputs_are_preview_safe() -> None:
    """The final question and wide result tables must survive PDF rendering."""
    original_cells = _load_original_cells()

    assert original_cells.__len__() == 45
    assert "KNeighborsRegressor" in original_cells[43].source
    assert "```" not in original_cells[43].source
    assert original_cells[14].source.rstrip().endswith(
        "logistic_sample_size_results.transpose()"
    )
    assert original_cells[25].source.rstrip().endswith(
        "overfit_diagnostics.transpose()"
    )
    assert ".transpose()" in original_cells[31].source
    assert original_cells[35].source.rstrip().endswith(
        "numeric_house_model_results.transpose()"
    )
    assert original_cells[44].source.rstrip().endswith(
        "final_house_model_comparison.transpose()"
    )


def test_pdf_preview_css_prevents_known_layout_defects() -> None:
    """Print CSS must hide anchors and keep the real-data heading with its section."""
    render_script_source = PDF_RENDER_SCRIPT_PATH.read_text(
        encoding="utf-8",
    )

    assert ".anchor-link" in render_script_source
    assert "break-after: avoid-page" in render_script_source
    assert "#Working-on-a-real-life-data" in render_script_source
    assert "break-before: page" in render_script_source


def test_executed_notebook_is_complete_and_error_free() -> None:
    """Stored outputs must prove that every code cell ran successfully in order."""
    with NOTEBOOK_PATH.open(mode="r", encoding="utf-8") as notebook_input:
        regression_notebook = nbformat.read(
            fp=notebook_input,
            as_version=4,
        )

    code_cells = [
        current_cell
        for current_cell in regression_notebook.cells
        if current_cell.cell_type == "code"
    ]
    execution_counts = [
        current_cell.execution_count for current_cell in code_cells
    ]
    error_outputs = [
        current_output
        for current_cell in code_cells
        for current_output in current_cell.outputs
        if current_output.output_type == "error"
    ]
    standard_error_outputs = [
        current_output
        for current_cell in code_cells
        for current_output in current_cell.outputs
        if (
            current_output.output_type == "stream"
            and current_output.name == "stderr"
        )
    ]
    final_output_data = [
        current_output.get("data", {})
        for current_output in code_cells[-1].outputs
        if current_output.output_type in {"display_data", "execute_result"}
    ]

    assert code_cells.__len__() == 21
    assert execution_counts == list(range(1, code_cells.__len__() + 1))
    assert error_outputs == []
    assert standard_error_outputs == []
    assert any(
        "text/html" in current_output_data
        for current_output_data in final_output_data
    )
