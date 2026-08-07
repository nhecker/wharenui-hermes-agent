# tests/agent flake classification

Three runs: `pytest tests/agent -n 4 --dist loadfile --tb=no -q`.

Classification is based on failure node IDs across all three runs.
- flaky: tests/agent/lsp/test_backend_gate.py::test_snapshot_baseline_called_for_local_env
- stable-fail: tests/agent/test_auxiliary_main_first.py::TestResolveAutoMainFirst::test_main_unavailable_falls_through_to_chain
- stable-fail: tests/agent/test_auxiliary_main_first.py::TestResolveAutoMainFirst::test_no_main_config_uses_chain_directly
- stable-fail: tests/agent/test_auxiliary_main_first.py::TestResolveAutoMainFirst::test_nous_main_uses_main_model_for_aux
- stable-fail: tests/agent/test_auxiliary_main_first.py::TestResolveAutoMainFirst::test_openrouter_main_uses_main_model_for_aux
- flaky: tests/agent/test_compress_focus.py::test_focus_topic_injected_into_summary_prompt
- flaky: tests/agent/test_compress_focus.py::test_no_focus_topic_no_injection
- flaky: tests/agent/test_compression_interrupt_protection.py::TestCompressionProtectsSummaryCall::test_compressor_call_site_uses_protection
- flaky: tests/agent/test_compression_logging_session_context.py::test_logging_session_context_follows_compression_rotation
- flaky: tests/agent/test_context_compressor_summary_continuity.py::test_existing_previous_summary_is_not_serialized_again_as_new_turn
- flaky: tests/agent/test_context_compressor_summary_continuity.py::test_resume_rehydrates_previous_summary_from_handoff_message
- stable-fail: tests/agent/test_skill_utils.py::test_skill_config_raw_cache_invalidates_on_config_edit
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_messaging_platform_is_case_insensitive
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[discord]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[email]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[matrix]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[signal]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[slack]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[sms]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_for_each_messaging_platform[whatsapp_cloud]
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_off_on_gateway_messaging_platform
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_auto_sentinel_resolves_to_surface_default
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_default_auto_off_on_messaging
- stable-fail: tests/agent/test_verification_stop.py::test_verify_on_stop_default_path_through_load_config

Stable-fail: 17
Flaky: 7
Stable-pass is the collected node set minus the failure union.

Recommendation: quarantine only the measured flaky IDs first, then fix shared module state, mock leakage, and ordering coupling in a separate work package. Do not pin ordering as the final fix.
