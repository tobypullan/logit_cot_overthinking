from pathlib import Path

import pandas as pd

from logit_cot_overthinking.visualization import plot_overview


def test_plot_overview_writes_png(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "question_id": "1",
                "decile": decile,
                "correct": decile > 0,
                "final_answer_commitment": 0.2 if decile == 0 else 0.9,
                "non_choice_probability": 0.8 if decile == 0 else 0.01,
                "prediction_flip": decile == 10,
            }
            for decile in range(0, 101, 10)
        ]
    )
    output = plot_overview(dataframe, tmp_path)
    assert output == tmp_path / "trajectory_overview.png"
    assert output.exists()
    assert output.stat().st_size > 0

