"""Provider-owned selection hints; not a Harness permission grant.

The service can still use its configured upstream when no authorized endpoint
matches. Only orchestration tools needing a generator advertise this metadata.
"""

GENERATOR_CONNECTIONS = {
    "exoharness": {
        "connection_bindings": {
            "version": 1,
            "headers": [{
                "name": "x-exoharness-video-generator-url",
                "required": False,
                "selection": "all",
                "selector": {
                    "input_fields": ["prompt", "duration_seconds"],
                    "description_groups": [
                        ["video", "clip", "视频"],
                        ["generat", "submit", "create", "生成", "提交"],
                    ],
                    "preferred_fields": [
                        "audio_enabled", "aspect_ratio", "resolution",
                        "first_frame_url", "prompt_mode",
                    ],
                    "deprioritize_terms": ["preview", "预览"],
                },
            }],
        },
    },
}
