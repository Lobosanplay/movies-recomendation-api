import ast
from typing import List


def convert(
    text: str,
) -> List[str]:
    if not isinstance(text, str):
        return []

    L = []

    try:
        for i in ast.literal_eval(text):
            L.append(i["name"])
    except (ValueError, SyntaxError, KeyError):
        return []
    return L


def convert3(text):
    L = []
    counter = 0

    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i["name"])
        counter += 1

    return L


def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])

    return L


def collapse(array: List[str]) -> List[str]:
    if not isinstance(array, list):
        return []
    L = []

    for i in array:
        if isinstance(i, str):
            L.append(i.replace(" ", ""))
        else:
            L.append(str(i).replace(" ", ""))
    return L
