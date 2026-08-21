"""Shared CSV, compact-display, workbook, and table/figure pairing helpers.

The JSON report artifacts remain the lossless numerical audit source.  CSV,
Markdown, HTML, and XLSX may use :func:`compact_rows` to collapse a reported
mean and its SD into one human-facing ``mean_sd`` field.
"""

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
    """One independently exported report table."""

    name: str
    rows: Sequence[Mapping[str, Any]]
    description: str = ""
    compact: bool = True


_SCORE_WORDS = (
    "accuracy",
    "auc",
    "coverage",
    "f1",
    "precision",
    "predictive_value",
    "recall",
    "sensitivity",
    "specificity",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_score_field(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _SCORE_WORDS) and not any(
        word in lowered for word in ("count", "rank", "runtime")
    )


def format_mean_sd(
    mean: Any,
    sd: Any,
    *,
    percent: bool = False,
    decimals: int = 1,
) -> str:
    """Format one mean/SD pair without manufacturing an SD when unavailable."""

    mean_value = _number(mean)
    sd_value = _number(sd)
    if mean_value is None:
        return "N/A"
    scale = 100.0 if percent else 1.0
    rendered_mean = f"{mean_value * scale:.{decimals}f}"
    if sd_value is None:
        return rendered_mean
    return f"{rendered_mean} ± {sd_value * scale:.{decimals}f}"


def _mean_sd_pairs(fields: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Discover conventional report mean/SD pairs without table-name checks."""

    available = set(fields)
    pairs: dict[str, tuple[str, str]] = {}
    if "mean" in available:
        for candidate in ("sample_sd", "population_sd", "sd", "std"):
            if candidate in available:
                pairs["mean"] = ("mean", candidate)
                break
    for mean_field in fields:
        candidates: list[str] = []
        candidates.append(f"{mean_field}_sd")
        if mean_field.endswith("_mean"):
            stem = mean_field[: -len("_mean")]
            candidates.extend((f"{stem}_sample_sd", f"{stem}_population_sd", f"{stem}_sd"))
        if mean_field.startswith("participant_mean_"):
            metric = mean_field[len("participant_mean_") :]
            candidates.extend(
                (
                    f"repeat_{metric}_sample_sd",
                    f"repeat_{metric}_population_sd",
                )
            )
        for candidate in candidates:
            if candidate in available:
                pairs[mean_field] = (mean_field, candidate)
                break
    return pairs


def compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a presentation projection with every available mean/SD collapsed.

    Raw JSON retains CI, extrema, and the individual mean/SD columns.  The
    compact projection removes only redundant distribution columns belonging
    to a successfully paired metric.
    """

    source = [dict(row) for row in rows]
    fields = list(dict.fromkeys(str(key) for row in source for key in row))
    pairs = _mean_sd_pairs(fields)
    if not pairs:
        return source
    paired_sd = {sd for _, sd in pairs.values()}
    removable: set[str] = set(paired_sd)
    for mean_field in pairs:
        if mean_field == "mean":
            removable.update(
                {
                    "population_sd",
                    "ci95_low",
                    "ci95_high",
                    "ci95_margin",
                    "minimum",
                    "maximum",
                }
            )
        elif mean_field.startswith("participant_mean_"):
            metric = mean_field[len("participant_mean_") :]
            removable.update(
                {
                    f"repeat_{metric}_population_sd",
                    f"repeat_{metric}_sample_sd",
                    f"repeat_{metric}_ci95_low",
                    f"repeat_{metric}_ci95_high",
                    f"repeat_{metric}_ci95_margin",
                    f"repeat_{metric}_minimum",
                    f"repeat_{metric}_maximum",
                }
            )
    output: list[dict[str, Any]] = []
    for row in source:
        projected: dict[str, Any] = {}
        for field in fields:
            if field in removable:
                continue
            if field in pairs:
                mean_field, sd_field = pairs[field]
                rendered_name = (
                    "mean_sd"
                    if field == "mean"
                    else (
                        "participant_"
                        + field[len("participant_mean_") :]
                        + "_mean_sd"
                    )
                    if field.startswith("participant_mean_")
                    else f"{field}_mean_sd"
                    if sd_field == f"{field}_sd"
                    else f"{field}_sd"
                )
                projected[rendered_name] = format_mean_sd(
                    row.get(mean_field),
                    row.get(sd_field),
                    percent=_is_score_field(
                        str(row.get("metric", mean_field))
                        if mean_field == "mean"
                        else mean_field
                    ),
                )
            else:
                projected[field] = row.get(field)
        output.append(projected)
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one table to one RFC-4180-style UTF-8 CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    if not fields:
        target.write_text("\n", encoding="utf-8")
        return target
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return target


def _sheet_name(raw: str, used: set[str]) -> str:
    base = re.sub(r"[\\/*?:\[\]]", "_", raw).strip(" '") or "table"
    base = base[:31]
    candidate = base
    index = 2
    while candidate.casefold() in used:
        suffix = f"_{index}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        result = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif value is None:
        result = ""
    else:
        result = str(value)
    return result[:32767]


def _column_name(index: int) -> str:
    value = index + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(65 + remainder) + output
    return output


def _worksheet_xml(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    data_rows: list[Sequence[Any]] = [fields]
    data_rows.extend([[row.get(field) for field in fields] for row in rows])
    if not fields:
        data_rows = [["status"], ["N/A_no_rows"]]
    xml_rows: list[str] = []
    for row_index, values in enumerate(data_rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(values):
            reference = f"{_column_name(column_index)}{row_index}"
            number = _number(value) if not isinstance(value, bool) else None
            if number is not None and not isinstance(value, str):
                cells.append(f'<c r="{reference}"><v>{number:.17g}</v></c>')
            else:
                text = escape(_cell_text(value))
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is>'
                    f'<t xml:space="preserve">{text}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def write_excel_workbook(path: str | Path, tables: Sequence[ReportTable]) -> Path:
    """Write a dependency-free XLSX workbook with exactly one sheet per table."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    prepared: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = []
    for table in tables:
        sheet = _sheet_name(table.name, used)
        rows = compact_rows(table.rows) if table.compact else [dict(row) for row in table.rows]
        prepared.append((table.name, sheet, rows))
    workbook_sheets = "".join(
        f'<sheet name="{escape(sheet)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (_name, sheet, _rows) in enumerate(prepared, start=1)
    )
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(prepared) + 1)
    )
    styles_id = len(prepared) + 1
    relationships += (
        '<Relationship '
        f'Id="rId{styles_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(prepared) + 1)
    )
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{overrides}</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{relationships}</Relationships>"
        ),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font/></fonts><fills count="1"><fill/></fills>'
            '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="1"><xf xfId="0"/></cellXfs></styleSheet>'
        ),
    }
    for index, (_name, _sheet, rows) in enumerate(prepared, start=1):
        files[f"xl/worksheets/sheet{index}.xml"] = _worksheet_xml(rows)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return target


__all__ = [
    "ReportTable",
    "compact_rows",
    "format_mean_sd",
    "write_csv",
    "write_excel_workbook",
]
