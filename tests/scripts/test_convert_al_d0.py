"""Tests for the AL_d0 initial-data conversion script."""

from pathlib import Path

from scripts.convert_al_d0 import convert_csv


def test_convert_csv_writes_one_observation_per_available_fidelity(
    tmp_path: Path,
) -> None:
    """The converter should reshape CXCALC and DOCK3 into fidelity rows."""

    input_path = tmp_path / "source.csv"
    output_path = tmp_path / "initial_data.csv"
    input_path.write_text(
        "INDX,SMILE,DOCK3,CXCALC\n7,CCO,-30.5,0.01\n8,CCN,-31.0,\n9,CCC,bad,0.02\n10\n",
        encoding="utf-8",
    )

    stats = convert_csv(input_path, output_path)

    assert stats.source_rows == 4
    assert stats.observations == 4
    assert stats.malformed_rows == 1
    assert stats.skipped_measurements["CXCALC"] == 1
    assert stats.skipped_measurements["DOCK3"] == 1
    assert output_path.read_text(encoding="utf-8") == (
        "SMILE,y,fidelity,INDX\n"
        "CCO,0.01,1,7\n"
        "CCO,-30.5,2,7\n"
        "CCN,-31.0,2,8\n"
        "CCC,0.02,1,9\n"
    )
