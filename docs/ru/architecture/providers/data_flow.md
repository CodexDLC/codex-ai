# Провайдеры — поток данных

## Gemini Text

```
PromptResult | str
    -> GeminiProvider.generate_text()
    -> google.genai.models.generate_content()
    -> response.text
    -> str
```

`GeminiProvider.answer()` делегирует в `generate_text()` для совместимости с `LLMDispatcher.process()`.

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

`response_mime_type` в этом пути не передается в `GenerateContentConfig.response_mime_type`; он используется только как fallback content type, если Gemini не вернул `inline_data.mime_type`.
Если `image_config.image_size` равен `4K` и Gemini отклоняет запрос, провайдер один раз повторяет его с `image_size="2K"`.

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
    -> chat.completions.create()
    -> choices[0].message.content
    -> str
```
