from pathlib import Path
import re

root = Path(__file__).parent

environments = r"(?:table|table\*|equation|equation\*)"

pattern = re.compile(
    rf"(\\begin\{{{environments}\}}.*?\\end\{{{environments}\}})",
    re.DOTALL
)

ac_pattern = re.compile(r"\\ac\{([^{}]+)\}")

for path in root.rglob("*.tex"):
    text = path.read_text(encoding="utf-8")
    new_text = pattern.sub(
        lambda match: ac_pattern.sub(r"\\acs{\1}", match.group(1)),
        text
    )

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Updated: {path}")