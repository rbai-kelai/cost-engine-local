import json
from pathlib import Path

from tcost_engine.cli import main

ROOT = Path(__file__).resolve().parents[1]
BLOTTER = ROOT / "examples" / "blotter.csv"


def test_cost_command_with_close_and_ten_mils(capsys) -> None:
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
            "--close",
            "50.00",
            "--bid",
            "50.00",
            "--ask",
            "50.02",
            "--mils",
            "10",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "commish             1" in out
    assert "spread              10" in out
    assert "market impact       0" in out
    assert "residual (slippage) 20" in out
    assert "total               31" in out


def test_blotter_command_on_the_example_file(capsys) -> None:
    code = main(["blotter", str(BLOTTER), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # 10 mils: AAPL 1 + MSFT 0.2 + SPY 0.1 = 1.3; spread 30; slippage 45
    assert payload["commission"] == "1.3"
    assert payload["commish"] == "1.3"
    assert payload["spread"] == "30"
    assert payload["market_impact"] == "not_modeled"
    assert payload["residual"] == "vwap_vs_close"
    assert payload["market_impact_cost"] == "0"
    assert payload["residual_cost"] == "45"
    assert payload["slippage"] == "45"
    assert payload["total"] == "76.3"
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
            "--mils",
            "0",
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
            "--mils",
            "0",
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
            "--mils",
            "0",
        ]
    )
    crossed = main(
        [
            "cost",
            "--side",
            "buy",
            "--qty",
            "1",
            "--price",
            "10",
            "--bid",
            "11",
            "--ask",
            "10",
            "--mils",
            "0",
        ]
    )
    err = capsys.readouterr().err
    assert ambiguous == 2 and crossed == 2
    assert "exactly one spread" in err
    assert "crossed quote" in err


def test_cli_rejects_a_bad_csv_row(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "side,quantity,price,bid,ask\nbuy,10,100,99,100\nbuy,-1,100,99,100\n",
        encoding="utf-8",
    )
    code = main(["blotter", str(path), "--mils", "0"])
    assert code == 2
    assert "row 3" in capsys.readouterr().err


def test_cli_rejects_unknown_columns(tmp_path: Path, capsys) -> None:
    path = tmp_path / "extra.csv"
    path.write_text("side,quantity,price,impact\nbuy,1,10,9\n", encoding="utf-8")
    code = main(["blotter", str(path), "--mils", "0"])
    assert code == 2
    assert "unknown CSV column" in capsys.readouterr().err
