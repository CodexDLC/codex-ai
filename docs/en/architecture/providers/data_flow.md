# Providers — Data Flow

## Gemini Text

```
PromptResult | str
    -> GeminiProvider.generate_text()
    -> google.genai.models.generate_content()
    -> response.text
    -> str
```

`GeminiProvider.answer()` delegates to `generate_text()` for compatibility with `LLMDispatcher.process()`.

## Gemini JSON

```
PromptResult | str
    -> GeminiProvider.generate_json(schema=...)
    -> GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    -> response.text
    -> json.loads()
    -> optional Pydantic validation
```

## Gemini Images

```
prompt: str
    -> GeminiProvider.generate_image_bytes()
    -> GenerateContentConfig(response_modalities=[IMAGE], image_config=...)
    -> first inline_data image part
    -> (bytes, actual_mime_type)
```

With `input_images`, the provider sends multimodal content:

```
prompt: str + input_images: list[ImageInput]
    -> text part + inline_data image parts
    -> GeminiProvider.generate_image_bytes()
    -> first returned inline_data image part
    -> (bytes, actual_mime_type)
```

`response_mime_type` is not passed to `GenerateContentConfig.response_mime_type` on this path; it is only a fallback content type when Gemini omits `inline_data.mime_type`.
When `image_config.image_size` is `4K` and Gemini rejects the request, the provider retries once with `image_size` changed to `2K`.

## Imagen Images

```
prompt: str
    -> GeminiProvider.generate_imagen_bytes()
    -> GenerateImagesConfig(output_mime_type=requested_mime)
    -> first generated_images image
    -> (bytes, actual_mime_type)
```

## OpenAI Text

```
PromptResult | str
    -> OpenAIProvider.generate_text()
    -> _OpenAIRequestMapper
    -> responses.create(store=False)
    -> response.output_text
    -> str
```

## OpenAI Structured JSON

```
PromptResult | str + Pydantic schema
    -> OpenAIProvider.generate_json()
    -> responses.parse(text_format=schema)
    -> response.output_parsed
    -> BaseModel
```

Without a schema, the provider requests JSON mode and decodes `response.output_text`.

## OpenAI Streaming

```
PromptResult | str
    -> OpenAIProvider.stream_text()
    -> responses.create(stream=True)
    -> response.output_text.delta events
    -> AsyncIterator[str]
```
