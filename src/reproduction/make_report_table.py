"""Turns table2_reproduced_p1.csv into the Published-vs-Reproduced markdown
table used in REPRODUCTION_REPORT.md. Never hand-type these numbers."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "reproduction" / "table2_reproduced_p1.csv")

lines = ["| sigma_e | sigma_d | Strategy | g_th (pub) | g_th (repro) | ADR pub % | ADR repro % | ADR diff (pp) | FAR pub % | FAR repro % | FAR diff (pp) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    adr_diff = r["ADR_reproduced_%"] - r["ADR_published_%"]
    far_diff = r["FAR_reproduced_%"] - r["FAR_published_%"]
    lines.append(
        f"| {r['sigma_e']} | {r['sigma_d']} | {int(r['strategy'])} | "
        f"{r['g_th_published']} | {r['g_th_reproduced']} | "
        f"{r['ADR_published_%']} | {r['ADR_reproduced_%']} | {adr_diff:+.2f} | "
        f"{r['FAR_published_%']} | {r['FAR_reproduced_%']} | {far_diff:+.2f} |"
    )

table_md = "\n".join(lines)
print(table_md)

mae_adr = (df["ADR_reproduced_%"] - df["ADR_published_%"]).abs().mean()
mae_far = (df["FAR_reproduced_%"] - df["FAR_published_%"]).abs().mean()
mae_gth = (df["g_th_reproduced"] - df["g_th_published"]).abs().mean()
print(f"\nMAE ADR = {mae_adr:.2f} pp | MAE FAR = {mae_far:.2f} pp | MAE g_th = {mae_gth:.4f}")

with open(ROOT / "results" / "reproduction" / "table2_comparison.md", "w") as f:
    f.write(table_md)
    f.write(f"\n\nMAE ADR = {mae_adr:.2f} pp | MAE FAR = {mae_far:.2f} pp | MAE g_th = {mae_gth:.4f}\n")
