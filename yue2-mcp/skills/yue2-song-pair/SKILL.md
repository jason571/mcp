---
name: yue2-song-pair
description: Create two original YuE2 songs from a user's brief, show and provide the lyrics, generate a separate cover for each song, and deliver playable audio plus downloads in an ExoHarness conversation. Use when the user asks for a new song, two song versions, or a complete music release with covers.
---

# Two YuE2 songs with covers

Turn one song brief into exactly two completed songs and two distinct cover images. Keep the user's language, theme, vocal and style requirements. If the user supplies lyrics, preserve their words unless they ask for changes. Otherwise write original, singable lyrics with clear `[Verse]` and `[Chorus]` sections. Show the complete lyrics in the conversation; use the same lyrics for both songs so the user can compare arrangements.

## Required capabilities

- `yue2` — song generation, progress, downloadable lyrics, MP3 preview and FLAC.
- `z-image` — one generated cover image for each song.

## Produce the pair

1. Make two meaningfully different arrangements from the brief, such as an intimate acoustic version and a fuller rhythmic version. Keep both faithful to explicit user constraints. Give each a short title and distinct image concept.
2. For each cover, call `z-image.generate_image` with a distinct prompt and `width=512`, `height=512`. Describe a square music cover matching that version's mood, colors and culture. Do not ask the image model to render text or a logo. Keep both returned image job IDs and links.
3. Call `yue2.generate_music` once per version with its own `style` and the complete lyrics. Keep the calls open for real MCP progress; generation is long. Do not omit `lyrics` while the service reports `lyrics_configured=false`. Do not submit a second copy when a call disconnects: recover the job ID from progress, then use `get_music_job` and `get_music_result`.
4. Treat each version as finished only when its music result and its own cover result both report `succeeded`. If a call returns only a queued/running job, query the corresponding `get_music_job` or `get_image_job` until terminal, then call `get_music_result` or `get_image_result` to publish the final resource links. Do not repeat a successful generation. Report a failed part and its job ID rather than claiming a full pair.

## Deliver in the conversation

- Show the full lyrics as readable text. The YuE2 music result also publishes a `lyrics.txt` file card for download; if needed, call `get_music_result` for the first successful song to expose it.
- Present Version 1 and Version 2 separately. Each version must have its own generated cover image card and an MP3 preview player; the FLAC resource is the original-quality download. Use the returned Artifact cards rather than inventing or rewriting URLs.
- Mention the two arrangement differences and actual audio durations. Do not call a song complete until the corresponding result confirms success. If the chat does not render an inline player, include the returned MP3 preview link as a fallback.

The image and music services have separate GPU queues. Generate the covers and songs in sequence to avoid competing for GPU memory. A 512×512 cover is sufficient unless the user asks for a larger image.
