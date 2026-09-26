"""Embed the trained MLP weights and test metrics into the dashboard page."""
import json
from litho_ai.page import full_page


def rnd(x, sig=6):
    if isinstance(x, list): return [rnd(v, sig) for v in x]
    return float(f"{x:.{sig}g}") if isinstance(x, float) else x


def load(path):
    m = json.load(open(path))
    m["weights"], m["biases"] = rnd(m["weights"]), rnd(m["biases"])
    return m


def build(out="docs/recipe-console.html"):
    data = {"cd": load("models/mlp_cd.json"), "print": load("models/mlp_print.json"),
            "ler": load("models/mlp_ler_log.json"), "metrics": json.load(open("models/metrics.json"))}
    html = open("dashboard/recipe_console_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open(out, "w").write(full_page(html))
    print(out, f"{len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
