import pandas as pd
import numpy as np
import sys

# ---------- helpers ----------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().dropna(how="all")

def ann_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std(ddof=1) * np.sqrt(252)

def compound(ser: pd.Series) -> float:
    return (1 + ser).prod() - 1

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df.rename(columns={"Exchange Date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()

def fmt_pct(x: float) -> str:
    return f"{x:+.2%}"

def to_bp(x: float) -> float:
    return x * 10000.0

def fmt_bp(x: float) -> str:
    return f"{x:+.0f} bp"

# Optional alert thresholds (tune as you like)
TOL_RET_BP  = 50   # flag if |Δ return| > 50 bp (0.50%)
TOL_RISK_BP = 25   # flag if |Δ risk|   > 25 bp (0.25%)

# ---------- main ----------
if __name__ == "__main__":
    prices = load_prices("data/raw/2025 FTSE 100 index stock price.xlsx")

    # WEIGHTS 
    w_train = pd.Series({
        "Tanco Holdings Bhd": 0.026,
        "Bursa Malaysia Bhd": 0.974,
    })

    # Expected from your generator
    expected_ret_sa   = 0.0839   # semi-annual return
    expected_risk_sa = 0.0792   # annualized risk (vol)

    # OOS blocks (add more if needed)
    blocks = [
        # ("2025-01-02", "2025-06-30"),
        ("2024-07-01", "2025-06-30"),
    ]

    # Prep returns
    r = daily_returns(prices)
    # test = r.loc["2025-01-02":"2025-06-30"].dropna(axis=1, how="all")
    test = r.loc["2024-07-01":"2025-06-30"].dropna(axis=1, how="all")

    # Align weights to available tickers
    w = w_train.reindex(test.columns).fillna(0.0)
    if w.sum() == 0:
        missing = sorted(set(w_train.index) - set(test.columns))
        print("ERROR: No overlap between weight tickers and price columns.", file=sys.stderr)
        if missing:
            print("These weighted tickers were not found in the price file:", file=sys.stderr)
            for t in missing:
                print(" -", t, file=sys.stderr)
        sys.exit(1)
    w = w / w.sum()

    # Compute realized & diffs
    rows = []
    for start, end in blocks:
        seg = (test @ w).loc[start:end].dropna()
        if len(seg) < 5:
            continue

        realized_ret_sa  = compound(seg)
        realized_risk_sa = seg.std(ddof=1) * np.sqrt(len(seg))

        d_ret   = realized_ret_sa   - expected_ret_sa
        d_risk  = realized_risk_sa - expected_risk_sa

        rows.append({
            "period": f"{start}→{end}",
            "realized_ret_sa": realized_ret_sa,
            "expected_ret_sa": expected_ret_sa,
            "d_ret": d_ret,
            "d_ret_bp": to_bp(d_ret),
            "realized_risk_sa": realized_risk_sa,
            "expected_risk_sa": expected_risk_sa,
            "d_risk": d_risk,
            "d_risk_bp": to_bp(d_risk),
            "ret_flag": "⚠" if abs(to_bp(d_ret))  > TOL_RET_BP  else "✓",
            "risk_flag":"⚠" if abs(to_bp(d_risk)) > TOL_RISK_BP else "✓",
        })

    diff = pd.DataFrame(rows)

    # Nicely formatted display
    if diff.empty:
        print("No results to display (check blocks and data).")
        sys.exit(0)

    disp = diff.copy()
    disp["realized_ret_sa"]   = disp["realized_ret_sa"].map(lambda x: f"{x:.2%}")
    disp["expected_ret_sa"]   = disp["expected_ret_sa"].map(lambda x: f"{x:.2%}")
    disp["d_ret_pct"]         = disp["d_ret"].map(fmt_pct)
    disp["d_ret_bp"]          = disp["d_ret_bp"].map(fmt_bp)

    disp["realized_risk_sa"] = disp["realized_risk_sa"].map(lambda x: f"{x:.2%}")
    disp["expected_risk_sa"] = disp["expected_risk_sa"].map(lambda x: f"{x:.2%}")
    disp["d_risk_pct"]        = disp["d_risk"].map(fmt_pct)
    disp["d_risk_bp"]         = disp["d_risk_bp"].map(fmt_bp)

    cols = [
        "period",
        "realized_ret_sa", "expected_ret_sa", "d_ret_pct", "d_ret_bp", "ret_flag",
        "realized_risk_sa", "expected_risk_sa", "d_risk_pct", "d_risk_bp", "risk_flag",
    ]
    print("\n=== Out-of-sample vs Expected (Semi-Annual) ===")
    print(disp[cols].to_string(index=False))

    # Quick one-liners (extra-straightforward)
    print("\n=== Differences (human-readable) ===")
    for _, r in diff.iterrows():
        print(f"[{r['period']}] Return Δ: {fmt_pct(r['d_ret'])} ({fmt_bp(r['d_ret_bp'])}) { '⚠' if r['ret_flag']=='⚠' else '✓' }")
        print(f"[{r['period']}] Risk   Δ: {fmt_pct(r['d_risk'])} ({fmt_bp(r['d_risk_bp'])}) { '⚠' if r['risk_flag']=='⚠' else '✓' }")
