# Providers

WorldForge ships with 11 provider adapters. Providers are auto-detected from
environment variables at startup.

## Provider Overview

| Provider   | Env Var(s)                            | Capabilities                                  | Access            |
|------------|---------------------------------------|-----------------------------------------------|-------------------|
| **Cosmos** | `NVIDIA_API_KEY`                      | predict, generate, reason, transfer, embed, plan | NIM API / self-hosted |
| **Runway** | `RUNWAY_API_SECRET`                   | predict, generate, transfer, plan             | REST API          |
| **JEPA**   | `JEPA_MODEL_PATH`                     | predict, reason, embed, plan (gradient)       | Local weights     |
| **Sora 2** | `OPENAI_API_KEY`                      | predict, generate                             | OpenAI API        |
| **Veo 3**  | `GOOGLE_API_KEY`                      | predict, generate                             | GenAI API         |
| **PAN**    | `PAN_API_KEY`                         | predict, generate, plan (stateful rounds)     | MBZUAI API        |
| **KLING**  | `KLING_API_KEY` + `KLING_API_SECRET`  | predict, generate                             | JWT REST API      |
| **MiniMax**| `MINIMAX_API_KEY`                     | predict, generate                             | REST API          |
| **Genie**  | `GENIE_API_KEY`                       | predict, generate, reason, transfer, plan     | Local surrogate   |
| **Marble** | *(always on)*                         | predict, generate, reason, transfer, embed, plan | Local surrogate   |
| **Mock**   | *(always on)*                         | all                                           | Deterministic testing |

## Capabilities Explained

- **predict**: Given state + action, predict the next world state.
- **generate**: Generate video frames from text or image prompts.
- **reason**: Extract semantic understanding from world state.
- **transfer**: Transfer world state between representations.
- **embed**: Produce embedding vectors for world states.
- **plan**: Multi-step action planning with the provider's native planner.

## Adding a Provider

See the [Contributing](../contributing.md) guide for step-by-step instructions
on implementing a new provider adapter.
