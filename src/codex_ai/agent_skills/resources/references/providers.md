# Direct provider use

These are fragments for an async application handler; `gemini_key` and `openai_key` come from application configuration. Install both optional extras to run both fragments.

```python
from pydantic import BaseModel
from codex_ai import GeminiProvider, OpenAIProvider, ImageInput

class Item(BaseModel):
    name: str
    power: int

gemini = GeminiProvider(api_key=gemini_key)
text = await gemini.generate_text("Write a short description.")
item = await gemini.generate_json("Create one item.", schema=Item)
image, content_type = await gemini.generate_image_bytes(
    "A simple map icon",
    response_mime_type="image/png",
    image_config={"aspect_ratio": "1:1"},
)

openai = OpenAIProvider(api_key=openai_key)
description = await openai.generate_text("Describe this item.")
structured = await openai.generate_json("Create one item.", schema=Item)
async for chunk in openai.stream_text("Tell a short story."):
    print(chunk, end="")
```

For a Gemini image edit, pass `input_images=[ImageInput(data=source_bytes, mime_type="image/png")]` to `generate_image_bytes()`. Its `response_mime_type` is a preferred or fallback output type, not the text config MIME field. Pass image controls in `image_config`. A rejected `4K` image-size request is retried once at `2K` by the provider.

For Imagen models, call `generate_imagen_bytes()` instead. It uses the separate `generate_images` SDK path, with `response_mime_type` mapped to Imagen's output MIME setting. Do not assume that image generation is supported by `OpenAIProvider` in this package.

Use `model=` and SDK-specific keyword arguments only after checking the installed package signatures. Defaults may change between releases.

OpenAI requests use the Responses API. It rejects `max_tokens`; pass `max_output_tokens` instead. `OpenAIProvider` defaults to `store=False`, so responses are not stored unless explicitly enabled.
