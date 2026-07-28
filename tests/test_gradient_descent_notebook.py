"""Structural and execution tests for the completed recitation notebook."""

import hashlib
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
EXPECTED_CODE_SOURCE_DIGEST = (
    "f2745eb69396e12ec0b71e2c715cd967b5d089f051f9e011f653a0a41300d30d"
)


def _load_notebook() -> nbformat.NotebookNode:
    """Load the recitation notebook as nbformat version 4."""
    with NOTEBOOK_PATH.open(mode="r", encoding="utf-8") as notebook_input:
        return nbformat.read(
            fp=notebook_input,
            as_version=4,
        )


def _code_source_digest(*, recitation_notebook: nbformat.NotebookNode) -> str:
    """Return a stable digest of all code-cell sources in notebook order."""
    code_source = "\n\0\n".join(
        current_cell.source
        for current_cell in recitation_notebook.cells
        if current_cell.cell_type == "code"
    )
    return hashlib.sha256(code_source.encode(encoding="utf-8")).hexdigest()


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


def test_recitation_preserves_instructional_code_comments() -> None:
    """The modern notebook must retain the reference recitation's teaching cues."""
    recitation_notebook = _load_notebook()
    all_code_source = "\n".join(
        current_cell.source
        for current_cell in recitation_notebook.cells
        if current_cell.cell_type == "code"
    )

    expected_comment_fragments = [
        "# The examples used for every experiment are the same.",
        "# Evaluate the current model, calculate both gradients, and make one GD step.",
        "# Plot the observations and both fitted model lines.",
        "# Calculate the MSE at every (slope, intercept) coordinate.",
        "# Sort x-values so adjacent curve points follow the polynomial.",
    ]
    for expected_comment_fragment in expected_comment_fragments:
        assert expected_comment_fragment in all_code_source


def test_original_markdown_writings_are_preserved_without_code_changes() -> None:
    """Reference teaching text must be restored while code sources stay unchanged."""
    recitation_notebook = _load_notebook()
    all_markdown_source = "\n".join(
        current_cell.source
        for current_cell in recitation_notebook.cells
        if current_cell.cell_type == "markdown"
    )
    normalized_markdown_source = " ".join(
        all_markdown_source.replace(">", "").split()
    )
    expected_markdown_fragments = [
        "# Linear Regression - recitation",
        "In this short excersice, you will experience with implementing linear regression for 2D points.",
        "If you remember, the simple Linear Regression in 2D defined as follows:",
        "The way to do it is by derivating Loss(b,m):",
        "Now lets see how the gradient descent process works in our linear regression function",
        "This can give us a notion about how the choice of lr impact the convergence of the algorithm.",
        "Try to modify the function \"linear_regression\" so now it will return parameters a,b,c such that minimize the cost:",
    ]

    for expected_markdown_fragment in expected_markdown_fragments:
        assert expected_markdown_fragment in normalized_markdown_source
    assert _code_source_digest(
        recitation_notebook=recitation_notebook,
    ) == EXPECTED_CODE_SOURCE_DIGEST


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
