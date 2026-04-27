from .chunker import load_schema, build_dataframe


def build_chunks(json_path, mode=None, chunk_fields=None) -> list[dict]:
    schema = load_schema(json_path)
    df = build_dataframe(schema, chunk_fields=chunk_fields, mode=mode)
    return [{"id": i, **row} for i, row in enumerate(df.to_dict(orient="records"))]


__all__ = ["load_schema", "build_dataframe", "build_chunks"]
