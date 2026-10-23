"""Indian registration plate format checks and position-aware character fixes."""
import re

STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
BH_SERIES = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Characters an OCR engine commonly confuses, per expected character class.
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
TO_LETTER = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}.items()}


def clean(text):
    text = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    # "IND" strip printed on high-security plates
    if text.startswith("IND") and len(text) > 9:
        text = text[3:]
    return text


def is_valid(text):
    return bool(STANDARD.match(text) or BH_SERIES.match(text))


def repair(text):
    """Fix letter/digit confusions in positions where the class is unambiguous.

    Standard layout: SS DD [L{0,3}] NNNN - the first two are letters, the next
    1-2 are digits, and the last four are digits.
    """
    text = clean(text)
    if is_valid(text) or len(text) < 6:
        return text
    chars = list(text)
    for i in (0, 1):
        chars[i] = TO_LETTER.get(chars[i], chars[i])
    for i in (2, len(chars) - 4, len(chars) - 3, len(chars) - 2, len(chars) - 1):
        chars[i] = TO_DIGIT.get(chars[i], chars[i])
    fixed = "".join(chars)
    return fixed if is_valid(fixed) else text
