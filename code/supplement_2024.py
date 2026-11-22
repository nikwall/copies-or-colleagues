"""Supplement to analysis_h1_h4_2024.py: results by question type, like-with-like
information (humans saw the freeze values, so models WITH freeze values are the
fair comparison), the scratchpad model crowd, and the H1 ladder with scratchpad.

Usage: python -I supplement_2024.py <project folder>
"""
import sys
import numpy as np
import pandas as pd

ROOT = sys.argv[1]
m = pd.read_csv(ROOT + "/derived/matrix_2024.csv", dtype={"qid": str, "model": str, "org": str, "prompt": str},
                keep_default_na=False)
m["e"] = m.forecast - m.outcome
mod = m[(m.group == "model") & (m.imputed == 0) & ~m.model.str.contains("__2nd_news")].copy()
FRONTIER = ["claude_3p5_sonnet", "gpt_4o", "gemini_1p5_pro", "gpt_4_turbo_0409", "claude_3_opus",
            "llama_3_70b", "mistral_large", "qwen_1p5_110b"]


def comps(df):
    g = df.groupby("item").e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    return dict(B1=(df.e ** 2).groupby(df.item).mean()[ok].mean(), F=(mu ** 2 - v / n)[ok].mean(), s2=v[ok].mean())


def brier(df):
    return ((df.e ** 2).groupby(df.item).mean()).mean()


print("By question type (raw Brier, item mean within type)")
for qt in ("dataset", "market"):
    d = m[m.qtype == qt]
    md = mod[mod.qtype == qt]
    pub, sup = comps(d[d.group == "public"]), comps(d[d.group == "super"])
    nofrz = md[~md.prompt.str.contains("freeze")].groupby(["model", "prompt"]).apply(brier, include_groups=False)
    frz = md[md.prompt.str.contains("freeze")].groupby(["model", "prompt"]).apply(brier, include_groups=False)
    sp = md[(md.prompt == "scratchpad") & md.model.isin(FRONTIER)]
    spf = md[(md.prompt == "scratchpad_with_freeze_values") & md.model.isin(FRONTIER)]
    crowd = lambda x: (x.groupby("item").e.mean() ** 2).mean()
    print(f"\n  {qt}: {d.item.nunique()} items")
    print(f"   public  individual {pub['B1']:.4f}  floor {pub['F']:.4f}  idio {pub['s2']:.4f}")
    print(f"   supers  individual {sup['B1']:.4f}  floor {sup['F']:.4f}")
    print(f"   best model without freeze values {nofrz.min():.4f} {nofrz.idxmin()}")
    print(f"   best model with freeze values    {frz.min():.4f} {frz.idxmin()}")
    print(f"   frontier crowd, scratchpad {crowd(sp):.4f} ({sp.model.nunique()} models); with freeze values {crowd(spf):.4f} ({spf.model.nunique()} models)")
    for lab, b in (("no freeze", nofrz.min()), ("with freeze", frz.min())):
        ns = pub["s2"] / (b - pub["F"]) if b > pub["F"] else np.inf
        print(f"   N* public vs best model ({lab}): {ns:.1f}")

print("\nH1 ladder with the scratchpad prompt (and with freeze values)")


def a_of(df):
    c = comps(df)
    return c["F"] / (c["F"] + c["s2"])


for p in ("scratchpad", "scratchpad_with_freeze_values"):
    z = mod[mod.prompt == p]
    one = z[z.model.isin(z.groupby("org").model.first())]
    so = [a_of(d) for _, d in z.groupby("org") if d.model.nunique() >= 2]
    print(f"   {p:32s} same developer {np.mean(so):.2f} | one per developer {a_of(one):.2f} | all {a_of(z):.2f} | frontier {a_of(z[z.model.isin(FRONTIER)]):.2f}")
fz = mod[mod.prompt.str.contains("freeze")]
cp = [a_of(d) for _, d in fz.groupby("model") if d.prompt.nunique() >= 2]
print(f"   same model, different prompts, all with freeze values: {np.mean(cp):.2f} ({len(cp)} models)")
print(f"   public individuals (item mean): {a_of(m[m.group == 'public']):.2f}; supers: {a_of(m[m.group == 'super']):.2f}")
