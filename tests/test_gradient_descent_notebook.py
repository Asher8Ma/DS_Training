"""Structural and execution tests for the completed recitation notebook."""

from pathlib import Path

import nbformat


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_DIRECTORY
    / "D - ML"
    / "regression"
    / "recitation"
    / "linear_regression_gradient_descent - recitation.ipynb"
)
PDF_RENDER_SCRIPT_PATH = (
    PROJECT_DIRECTORY / "scripts" / "render_notebook_preview_pdf.py"
)


def _load_notebook() -> nbformat.NotebookNode:
    """Load the recitation notebook as nbformat version 4."""
    with NOTEBOOK_PATH.open(mode="r", encoding="utf-8") as notebook_input:
        return nbformat.read(
            fp=notebook_input,
            as_version=4,
        )


def test_every_exercise_has_three_approaches_and_direct_answer() -> None:
    """Each of the four exercises must contain explicit documented reasoning."""
    recitation_notebook = _load_notebook()
    generated_solution_cells = [
        current_cell
        for current_cell in recitation_notebook.cells
        if current_cell.metadata.get("generated_solution", False)
    ]

    assert generated_solution_cells.__len__() == 4
    for solution_cell in generated_solution_cells:
        solution_source = solution_cell.source
        assert "**Three approaches considered**" in solution_source
        assert "\n1. " in solution_source
        assert "\n2. " in solution_source
        assert "\n3. " in solution_source
        assert "**Choice:**" in solution_source
        assert "**Direct answer:**" in solution_source
        assert solution_source.count("$") % 2 == 0


def test_original_cells_are_preserved_and_placeholders_are_filled() -> None:
    """Original cell order must remain traceable and all blanks must be solved."""
    recitation_notebook = _load_notebook()
    original_cells = [
        current_cell
        for current_cell in recitation_notebook.cells
        if "exercise_cell_index" in current_cell.metadata
    ]
    original_cells.sort(
        key=lambda current_cell: current_cell.metadata["exercise_cell_index"],
    )
    all_code_source = "\n".join(
        current_cell.source
        for current_cell in recitation_notebook.cells
        if current_cell.cell_type == "code"
    )

    assert original_cells.__len__() == 28
    assert [current_cell.metadata["exercise_cell_index"] for current_cell in original_cells] == list(
        range(28)
    )
    assert "#answer" not in all_code_source
    assert "##answer" not in all_code_source
    assert "random_state=RANDOM_SEED" in original_cells[5].source
    assert "def linear_regression(\n    *," in original_cells[10].source
    assert "def calc_cost(\n    *," in original_cells[18].source
    assert original_cells[20].source.rstrip().endswith(
        "learning_rate_summary"
    )


def test_preview_rules_and_generated_bonus_code_are_present() -> None:
    """The notebook and PDF renderer must contain the required preview safeguards."""
    recitation_notebook = _load_notebook()
    generated_bonus_code = [
        current_cell
        for current_cell in recitation_notebook.cells
        if current_cell.metadata.get("generated_solution_code", False)
    ]
    render_script_source = PDF_RENDER_SCRIPT_PATH.read_text(
        encoding="utf-8",
    )

    assert generated_bonus_code.__len__() == 2
    assert "def polynomial_regression(\n    *," in generated_bonus_code[0].source
    assert generated_bonus_code[-1].source.rstrip().endswith(
        "polynomial_comparison"
    )
    assert ".anchor-link" in render_script_source
    assert "table.dataframe" in render_script_source
    assert "max-width: 100%" in render_script_source


def test_executed_notebook_is_complete_and_error_free() -> None:
    """Stored outputs must show sequential execution without errors or stderr."""
    recitation_notebook = _load_notebook()
    code_cells = [
        current_cell
        for current_cell in recitation_notebook.cells
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

    assert code_cells.__len__() == 13
    assert execution_counts == list(range(1, 14))
    assert error_outputs == []
    assert standard_error_outputs == []
    assert any(
        "text/html" in current_output_data
        for current_output_data in final_output_data
    )
