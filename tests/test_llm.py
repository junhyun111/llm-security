from llm_security.llm import _decode_json_content


def test_decode_json_content_accepts_plain_and_fenced_json() -> None:
    assert _decode_json_content('{"findings": []}') == {"findings": []}
    assert _decode_json_content('```json\n{"findings": []}\n```') == {"findings": []}
