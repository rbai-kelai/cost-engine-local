import json
from pathlib import Path

from tcost_engine.cli import main

ROOT = Path(__file__).resolve().parents[1]
BLOTTER = ROOT / "examples" / "blotter.csv"


def test_cost_command_matches_the_readme_example(capsys) -> None:
    code = main(
        [
            "cost",
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1000",
            "--price",
            "50.02",
            "--bid",
            "50.00",
            "--ask",
            "50.02",
            "--per-share",
            "0.005",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "commish             5" in out
    assert "spread              10" in out
    assert "market impact       0" in out
    assert "residual            0" in out
    assert "total               15" in out


def test_blotter_command_on_the_example_file(capsys) -> None:
    code = main(
        [
            "blotter",
            str(BLOTTER),
            "--per-share",
            "0.005",
            "--min-commission",
            "1",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["commission"] == "7"
    assert payload["commish"] == "7"
    assert payload["spread"] == "30"
    assert payload["market_impact"] == "not_modeled"
    assert payload["residual"] == "not_modeled"
    assert payload["market_impact_cost"] == "0"
    assert payload["residual_cost"] == "0"
    assert payload["total"] == "37"
    assert payload["execution_notional"] == "184049"


def test_full_spread_bps_is_half_of_one_way_on_the_cli(capsys) -> None:
    full = main(
        [
            "cost",
            "--side",
            "buy",
            "--qty",
            "10",
            "--price",
            "100",
            "--full-spread-bps",
            "10",
            "--json",
        ]
    )
    full_payload = json.loads(capsys.readouterr().out)
    one_way = main(
        [
            "cost",
            "--side",
            "buy",
            "--qty",
            "10",
            "--price",
            "100",
            "--one-way-spread-bps",
            "10",
            "--json",
        ]
    )
    one_way_payload = json.loads(capsys.readouterr().out)
    assert full == 0 and one_way == 0
    assert full_payload["spread"] == "0.5"
    assert one_way_payload["spread"] == "1"


def test_cli_rejects_ambiguous_or_incomplete_input(capsys) -> None:
    ambiguous = main(
        [
            "cost",
            "--side",
            "buy",
            "--qty",
            "1",
            "--price",
            "10",
            "--bid",
            "10",
            "--ask",
            "10.01",
            "--full-spread-bps",
            "5",
        ]
    )
    missing_minimum = main(["cost", "--side", "buy", "--qty", "1", "--price", "10", "--bid", "10", "--ask", "10.01", "--min-commission", "1"])
    crossed = main(
        ["cost", "--side", "buy", "--qty", "1", "--price", "10", "--bid", "11", "--ask", "10"]
    )
    err = capsys.readouterr().err
    assert ambiguous == 2 and missing_minimum == 2 and crossed == 2
    assert "exactly one spread" in err
    assert "--min-commission requires --per-share" in err
    assert "crossed quote" in err


def test_cli_rejects_a_bad_csv_row(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "side,quantity,price,bid,ask\nbuy,10,100,99,100\nbuy,-1,100,99,100\n",
        encoding="utf-8",
    )
    code = main(["blotter", str(path)])
    assert code == 2
    assert "row 3" in capsys.readouterr().err


def test_cli_rejects_unknown_columns(tmp_path: Path, capsys) -> None:
    path = tmp_path / "extra.csv"
    path.write_text("side,quantity,price,impact\nbuy,1,10,9\n", encoding="utf-8")
    code = main(["blotter", str(path)])
    assert code == 2
    assert "unknown CSV column" in capsys.readouterr().err
