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
    -> GenerateContentConfig(response_modalities=[IMAGE])
    -> first inline_data image part
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
