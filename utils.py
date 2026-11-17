import csv
import glob
import json
import os

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
import yaml

# Repo root: this file lives at the repo root.
ROOT = os.path.dirname(os.path.abspath(__file__))

_CONFIG = None


def repo_path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(repo_path("config", "config.yaml"), encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_records(directory: str) -> list[dict]:
    return [read_json(p) for p in sorted(glob.glob(os.path.join(directory, "*.json")))]


def load_json_by_name(directory: str) -> dict[str, dict]:
    return {os.path.splitext(os.path.basename(p))[0]: read_json(p)
            for p in sorted(glob.glob(os.path.join(directory, "*.json")))}


def gold_records(set_name: str) -> dict[str, dict]:
    return {r["source_file_id"]: r
            for r in load_records(repo_path("data", "golden_set", "parsed_docs", set_name))}


def raw_docs(set_name: str) -> list[str]:
    return sorted(glob.glob(repo_path("data", "golden_set", "raw_docs", set_name, "*")))


def questions() -> dict[str, dict]:
    return {r["question_id"]: r for r in read_csv_rows(repo_path("data", "golden_set", "questions", "qa.csv"))}


def read_file_mapping(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            newp = (row.get("new_path") or "").strip()
            oldp = (row.get("old_path") or "").strip()
            if newp:
                mapping[newp] = oldp
    return mapping


_SCHEMA_CACHE: dict[str, dict] = {}
_REGISTRY: Registry | None = None


def load_schema(name: str) -> dict:
    if name not in _SCHEMA_CACHE:
        with open(repo_path("schemas", f"{name}.schema.json"), encoding="utf-8") as f:
            _SCHEMA_CACHE[name] = json.load(f)
    return _SCHEMA_CACHE[name]


def _schema_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        resources = []
        for fname in sorted(os.listdir(repo_path("schemas"))):
            if fname.endswith(".schema.json"):
                schema = load_schema(fname[: -len(".schema.json")])
                resources.append(
                    (schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012))
                )
        _REGISTRY = Registry().with_resources(resources)
    return _REGISTRY


def validate(obj, schema_name: str) -> tuple[bool, list[str]]:
    schema = load_schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema, registry=_schema_registry())
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    if not errors:
        return True, []
    return False, [e.message for e in errors]


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
