import json
import os
import tempfile
import unittest
from unittest.mock import patch

from limo4si.language_client import (
    LanguageTransportError,
    OpenAICompatibleStructuredClient,
    create_client,
)


class LanguageClientTests(unittest.TestCase):
    def test_strict_schema_request_and_json_response(self):
        captured = {}

        def transport(url, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body, timeout=timeout)
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "schema_version": "limo4si.language_realization.v1",
                            "semantic_gt_id": "case",
                            "answer_signature": "sha256:signed",
                            "question_template": "{{question_focus}}",
                            "option_template": "{{option_statement}}",
                            "explanation_template": "{{evidence_statement}}",
                        })
                    }
                }]
            }

        client = OpenAICompatibleStructuredClient(
            api_key="secret-not-logged",
            base_url="https://provider.example/v1/",
            model="language-model",
            transport=transport,
        )
        result = client.create_structured_output(
            system_prompt="wording only",
            payload={"semantic_gt_id": "case", "locked_clauses": {}},
            json_schema={"type": "object"},
        )
        self.assertEqual(captured["url"], "https://provider.example/v1/chat/completions")
        self.assertEqual(captured["body"]["response_format"]["type"], "json_schema")
        self.assertNotIn("correct_option_id", captured["body"]["messages"][1]["content"])
        self.assertEqual(result["semantic_gt_id"], "case")

    def test_factory_requires_environment_only_credentials_and_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LIMO4SI_LANGUAGE_API_KEY"):
                create_client()
        with patch.dict(os.environ, {"LIMO4SI_LANGUAGE_API_KEY": "secret"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LIMO4SI_LANGUAGE_MODEL"):
                create_client()

    def test_factory_can_read_a_restricted_temporary_secret_file(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as handle:
            handle.write("temporary-secret\n")
            handle.flush()
            environment = {
                "LIMO4SI_LANGUAGE_API_KEY_FILE": handle.name,
                "LIMO4SI_LANGUAGE_MODEL": "language-model",
            }
            with patch.dict(os.environ, environment, clear=True):
                client = create_client()
        self.assertEqual(client.api_key, "temporary-secret")

    def test_transient_transport_failure_is_retried(self):
        attempts = []

        def transport(url, headers, body, timeout):
            attempts.append(url)
            if len(attempts) < 3:
                raise LanguageTransportError("temporary TLS failure")
            return {
                "choices": [{"message": {"content": json.dumps({
                    "schema_version": "limo4si.language_realization.v1",
                    "semantic_gt_id": "case",
                    "answer_signature": "sha256:signed",
                    "question_template": "{{question_focus}}",
                    "option_template": "{{option_statement}}",
                    "explanation_template": "{{evidence_statement}}",
                })}}]
            }

        client = OpenAICompatibleStructuredClient(
            api_key="secret-not-logged", base_url="https://provider.example/v1",
            model="language-model", transport=transport, retry_delay_sec=0,
        )
        result = client.create_structured_output(
            system_prompt="wording only", payload={}, json_schema={"type": "object"},
        )
        self.assertEqual(result["semantic_gt_id"], "case")
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
