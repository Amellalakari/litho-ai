"""Wrap an artifact-style page fragment into a complete HTML document (for GitHub Pages)."""


def full_page(fragment: str) -> str:
    head, sep, body = fragment.partition('<div class="wrap">')
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'{head}</head>\n<body>\n{sep}{body}\n</body>\n</html>\n')
