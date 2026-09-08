"""Reusable table serialization for pipeline and analysis artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class ReportTable:
    name: str
    rows: Sequence[Mapping[str, Any]]
    description: str = ""
    compact: bool = True


@dataclass(frozen=True)
class ColumnDefinition:
    column_name: str
    display_label: str
    source_fields: tuple[str, ...]
    definition: str
    formula: str
    documentation_kind: str


_METRICS = {
    "balanced_accuracy": ("Macro-average recall across declared classes.", "mean_c TP_c/(TP_c+FN_c)"),
    "macro_f1": ("Unweighted mean of class F1 scores.", "mean_c 2 precision_c recall_c/(precision_c+recall_c)"),
    "precision": ("Positive predictive value.", "TP/(TP+FP)"),
    "recall": ("Sensitivity.", "TP/(TP+FN)"),
    "sensitivity": ("True-positive rate.", "TP/(TP+FN)"),
    "specificity": ("True-negative rate.", "TN/(TN+FP)"),
    "f1": ("Harmonic mean of precision and recall.", "2 precision recall/(precision+recall)"),
    "coverage_rate": ("Fraction of eligible observations retained.", "n_retained/n_total"),
    "expected_calibration_error": (
        "Top-label equal-width-bin calibration error.",
        "sum_b n_b/N |accuracy_b-confidence_b|",
    ),
    "multiclass_brier": ("Class-summed squared probability error.", "mean_i sum_c (p_ic-1[y_i=c])^2"),
    "multiclass_log_loss": ("Negative log probability of the true class.", "-mean_i log(p_i,y_i)"),
    "roc_auc": ("Area under the empirical ROC curve.", "integral TPR(FPR) dFPR"),
    "pr_auc": ("Average precision under the precision-recall curve.", "sum_n delta(recall_n) precision_n"),
}
_DIRECT = re.compile(r"(?:^|_)(id|name|path|case|class|config|fold|repeat|role|seed|source|status|version)(?:_|$)")


def _json(value: Any) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _metric_key(name: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_").replace("macrof1", "macro_f1")
    return next((key for key in sorted(_METRICS, key=len, reverse=True) if key in normalized), None)


def column_definition(name: str | tuple[str, str] | tuple[str, str, bool],
                      *,
                      display_label: str | None = None) -> ColumnDefinition:
    if isinstance(name, tuple):
        if len(name) not in {2, 3}:
            raise TypeError("mean/SD column needs (mean, SD[, percent])")
        mean, sd = name[:2]
        percent = True if len(name) == 2 else name[2]
        if not isinstance(mean, str) or not isinstance(sd, str) or not isinstance(percent, bool):
            raise TypeError("mean/SD column types are invalid")
        scale = "100" if percent else "1"
        return ColumnDefinition(
            f"{mean} + {sd}",
            display_label or f"{mean} mean +/- SD",
            (mean, sd),
            "Compact mean and sample-SD display.",
            f"display = {scale} * {mean} +/- {scale} * {sd}",
            "explicit_mean_sd_composite",
        )
    if not name.strip():
        raise ValueError("column name must be non-empty")
    metric = _metric_key(name)
    if metric:
        definition, formula = _METRICS[metric]
        kind = "metric"
    elif name.endswith(("_ci95", "_ci95_low", "_ci95_high")):
        definition, formula, kind = "95% interval endpoint.", "registered percentile or Student-t interval", "interval"
    elif name.endswith(("_p_value", "_adjusted_p_value")):
        definition, formula, kind = (
            "Registered statistical-test probability.",
            "see analysis_manifest statistical policy",
            "inference",
        )
    else:
        definition = "Persisted identifier, configuration, provenance, count, or status value."
        formula = "N/A — direct source value" if _DIRECT.search(name) else "N/A — source-defined value"
        kind = "direct"
    return ColumnDefinition(name, display_label or name, (name, ), definition, formula, kind)


def column_definition_rows(
    names: Sequence[str | tuple[str, str] | tuple[str, str, bool]],
    *,
    display_labels: Sequence[str] | None = None,
    table_name: str = "",
    table_description: str = "",
) -> list[dict[str, Any]]:
    labels: Sequence[str | None] = display_labels if display_labels is not None else [None] * len(names)
    if len(labels) != len(names):
        raise ValueError("display_labels must align with columns")
    output = []
    for index, (name, label) in enumerate(zip(names, labels, strict=True), 1):
        definition = column_definition(name, display_label=label)
        output.append({
            "table_name": table_name,
            "table_description": table_description,
            "ordinal_position": index,
            **vars(definition),
            "source_fields": "; ".join(definition.source_fields),
        })
    return output


def table_column_definition_rows(tables: Sequence[ReportTable]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for table in tables:
        names = tuple(dict.fromkeys(key for row in table.rows for key in row))
        output.extend(column_definition_rows(names, table_name=table.name, table_description=table.description))
    return output


def table_column_definition_rows_from_csv_directory(
    directory: str | Path, *, excluded_stems: Sequence[str] = ("table_column_definitions", )) -> list[dict[str, Any]]:
    output = []
    for path in sorted(Path(directory).glob("*.csv")):
        if path.stem in excluded_stems:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            names = csv.DictReader(stream).fieldnames or ()
        output.extend(
            column_definition_rows(names, table_name=path.stem, table_description="Persisted CSV report table"))
    return output


def _definitions_block(names: Sequence[str], labels: Sequence[str] | None, *, html: bool) -> str:
    rows = column_definition_rows(names, display_labels=labels)
    if html:
        body = "".join(
            f"<tr><td>{r['display_label']}</td><td>{r['definition']}</td><td><code>{r['formula']}</code></td></tr>"
            for r in rows)
        return f"<table><thead><tr><th>Column</th><th>Definition</th><th>Formula</th></tr></thead><tbody>{body}</tbody></table>"
    lines = ["| Column | Definition | Formula |", "|---|---|---|"]
    lines += [f"| {r['display_label']} | {r['definition']} | {r['formula']} |" for r in rows]
    return "\n".join(lines)


def markdown_column_definitions_block(names: Sequence[str], *, display_labels: Sequence[str] | None = None) -> str:
    return _definitions_block(names, display_labels, html=False)


def html_column_definitions_block(names: Sequence[str], *, display_labels: Sequence[str] | None = None) -> str:
    return _definitions_block(names, display_labels, html=True)


def write_table_column_definitions(
    output_directory: str | Path,
    *,
    tables: Sequence[ReportTable] | None = None,
    csv_directory: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    if (tables is None) == (csv_directory is None):
        raise ValueError("provide exactly one of tables or csv_directory")
    rows = table_column_definition_rows(tables or
                                        ()) if tables is not None else table_column_definition_rows_from_csv_directory(
                                            csv_directory)  # type: ignore[arg-type]
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(root / "table_column_definitions.csv", rows)
    json_path = root / "table_column_definitions.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    md_path = root / "TABLE_COLUMN_DEFINITIONS.md"
    md_path.write_text(
        "# Table column definitions\n\n" + _definitions_block(
            [str(row["column_name"]) for row in rows], [str(row["display_label"]) for row in rows], html=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, md_path


def format_mean_sd(mean: Any, sd: Any, *, percent: bool = False, decimals: int = 1) -> str:
    try:
        rendered = f"{(100.0 if percent else 1.0) * float(mean):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"
    try:
        spread = f"{(100.0 if percent else 1.0) * float(sd):.{decimals}f}"
    except (TypeError, ValueError):
        return rendered
    return f"{rendered} ± {spread}"


def format_interval(lower: Any, upper: Any, *, percent: bool = False, decimals: int = 1) -> str:
    try:
        scale = 100.0 if percent else 1.0
        return f"[{scale * float(lower):.{decimals}f}, {scale * float(upper):.{decimals}f}]"
    except (TypeError, ValueError):
        return "N/A"


def compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for row in output:
        for key in tuple(row):
            if key.endswith("_mean") and (sd_key := key[:-5] + "_sd") in row:
                row[key[:-5] + "_mean_sd"] = format_mean_sd(row[key], row[sd_key], percent=_metric_key(key) is not None)
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows({key: _json(value) for key, value in row.items()} for row in rows)
    return target


def write_excel_workbook(path: str | Path, tables: Sequence[ReportTable]) -> Path:
    """Write a small dependency-free XLSX using inline strings."""
    selected = list(tables) or [ReportTable("empty", ({"status": "N/A_no_tables"}, ))]
    used: set[str] = set()
    prepared = []
    for index, table in enumerate(selected, 1):
        base = re.sub(r"[\\/*?:\[\]]", "_", table.name)[:31] or f"table_{index}"
        name, suffix = base, 1
        while name.casefold() in used:
            suffix += 1
            name = f"{base[:27]}_{suffix}"
        used.add(name.casefold())
        rows = compact_rows(table.rows) if table.compact else [dict(row) for row in table.rows]
        prepared.append((name, rows or [{"status": "N/A_no_rows"}]))
    sheets = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                     for i, (name, _) in enumerate(prepared, 1))
    relations = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1,
                       len(prepared) + 1))
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1,
                       len(prepared) + 1))
    files = {
        "[Content_Types].xml":
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + overrides + "</Types>",
        "_rels/.rels":
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml":
        '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        + sheets + "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels":
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + relations + "</Relationships>",
    }
    for sheet_index, (_, rows) in enumerate(prepared, 1):
        fields = list(dict.fromkeys(str(key) for row in rows for key in row))
        matrix = [fields, *[[_json(row.get(key)) for key in fields] for row in rows]]
        xml_rows = []
        for row_index, values in enumerate(matrix, 1):
            cells = []
            for column_index, value in enumerate(values):
                n = column_index + 1
                column = ""
                while n:
                    n, remainder = divmod(n - 1, 26)
                    column = chr(65 + remainder) + column
                text = escape(str(value)[:32767])
                cells.append(
                    f'<c r="{column}{row_index}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>')
            xml_rows.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")
        files[f"xl/worksheets/sheet{sheet_index}.xml"] = (
            '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(xml_rows) + "</sheetData></worksheet>")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return target


def write_excel_workbook_from_csv_directory(path: str | Path, directory: str | Path) -> Path:
    tables = []
    old_limit = csv.field_size_limit()
    csv.field_size_limit(min(2**31 - 1, max(old_limit, 100_000_000)))
    try:
        for source in sorted(Path(directory).glob("*.csv")):
            with source.open(encoding="utf-8", newline="") as stream:
                tables.append(ReportTable(source.stem, tuple(csv.DictReader(stream)), compact=False))
    finally:
        csv.field_size_limit(old_limit)
    return write_excel_workbook(path, tables)


__all__ = [
    "ColumnDefinition",
    "ReportTable",
    "column_definition",
    "column_definition_rows",
    "compact_rows",
    "format_mean_sd",
    "format_interval",
    "html_column_definitions_block",
    "markdown_column_definitions_block",
    "table_column_definition_rows",
    "table_column_definition_rows_from_csv_directory",
    "write_csv",
    "write_excel_workbook",
    "write_excel_workbook_from_csv_directory",
    "write_table_column_definitions",
]
