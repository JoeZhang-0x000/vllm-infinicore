"""Extract explicitly marked GSM8K final answers without using gold values."""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

METRIC = "numeric exact match from final #### or boxed answer"
NUMBER = r"[-+]?(?:[0-9][0-9,]*(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
MARKER = re.compile(r"####\s*(?:\\?\$[ \t]*)?(" + NUMBER + r")")
FRACTION = re.compile(
    r"(?:\\[dt]?frac\{("
    + NUMBER
    + r")\}\{("
    + NUMBER
    + r")\}|("
    + NUMBER
    + r")\s*/\s*("
    + NUMBER
    + r"))"
)


def numeric_value(text: str) -> Fraction | None:
    text = text.strip().replace(",", "")
    if re.fullmatch(NUMBER, text):
        try:
            value = Decimal(text)
            return Fraction(value) if value.is_finite() else None
        except (InvalidOperation, ValueError):
            return None
    match = FRACTION.fullmatch(text)
    if match:
        numerator, denominator = (
            match.groups()[:2] if match.group(1) is not None else match.groups()[2:]
        )
        a, b = numeric_value(numerator), numeric_value(denominator)
        return a / b if a is not None and b else None
    return None


def format_value(value: Fraction) -> str:
    # Preserve familiar decimal answers when they have a finite decimal form.
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return str(value)
    scale = max(twos, fives)
    scaled = value.numerator * 2 ** (scale - twos) * 5 ** (scale - fives)
    digits = str(abs(scaled)).zfill(scale + 1)
    if scale:
        digits = (digits[:-scale] + "." + digits[-scale:]).rstrip("0").rstrip(".")
    return ("-" if scaled < 0 else "") + digits


def boxed_answers(text: str):
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if not depth:
            yield match.start(), text[match.end() : end - 1].strip()


def extract_answer(text: str) -> str | None:
    text = text.rsplit("</think>", 1)[-1]
    candidates = []
    for match in MARKER.finditer(text):
        tail = text[match.end() :].lstrip()
        if text[: match.start()].rstrip().endswith("**") and tail.startswith("**"):
            tail = tail[2:].lstrip()
        if tail.startswith(("/", "+", "*", "=", ".", ",")):
            # A fraction may be a valid numeric answer; an expression is not.
            fraction = re.match(
                r"(" + NUMBER + r")\s*/\s*(" + NUMBER + r")",
                text[match.start() :].split("####", 1)[1].strip(),
            )
            value = numeric_value(fraction.group(0)) if fraction else None
        else:
            value = numeric_value(match.group(1))
        candidates.append((match.start(), value))
    for position, content in boxed_answers(text):
        candidates.append((position, numeric_value(content)))
    if not candidates:
        return None
    value = max(candidates, key=lambda pair: pair[0])[1]
    return None if value is None else format_value(value)
