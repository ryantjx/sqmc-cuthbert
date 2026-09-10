"""Canonical scalar-table rows shared by export and artifact validation."""


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


LABELS = {"alpha": r"$\alpha$", "beta": r"$\beta$", "kappa": r"$\kappa$",
          "friendly_scale": r"friendly\_scale"}


def table_body(rows):
    """Return complete ordered rows, including the header and table rules."""
    learned = [row for row in rows if row["parameter"] in LABELS]
    if [row["parameter"] for row in learned] != list(LABELS):
        raise ValueError("Scalar table requires the four learned parameters in order")
    lines = [r"\toprule", r"Parameter & Interpretation & EKF & RB-SQMC \\", r"\midrule"]
    for row in learned:
        fmt = ".3e" if row["parameter"] == "kappa" else ".4f"
        lines.append(" & ".join([
            LABELS[row["parameter"]], latex_escape(row["meaning"]),
            format(row["ekf"], fmt), format(row["sqmc"], fmt),
        ]) + r" \\")
    return lines + [r"\bottomrule"]
