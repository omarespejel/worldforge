# REST API Reference

WorldForge exposes a REST API via `worldforge serve` on port 8080 by default.
All endpoints are prefixed with `/v1/`.

## Starting the Server

```bash
worldforge serve                    # default: 127.0.0.1:8080
worldforge serve --port 9090        # custom port
worldforge serve --host 0.0.0.0     # bind all interfaces
```

## Endpoints

### Providers

| Method | Endpoint                       | Description              |
|--------|--------------------------------|--------------------------|
| GET    | `/v1/providers`                | List registered providers |
| GET    | `/v1/providers/{name}`         | Get provider details     |
| GET    | `/v1/providers/{name}/health`  | Provider health check    |

### Worlds

| Method | Endpoint                          | Description              |
|--------|-----------------------------------|--------------------------|
| POST   | `/v1/worlds`                      | Create a world           |
| GET    | `/v1/worlds`                      | List all worlds          |
| GET    | `/v1/worlds/{id}`                 | Get world state          |
| DELETE | `/v1/worlds/{id}`                 | Delete a world           |
| POST   | `/v1/worlds/{id}/predict`         | Run a prediction         |
| POST   | `/v1/worlds/{id}/plan`            | Plan action sequence     |
| POST   | `/v1/worlds/{id}/evaluate`        | Run eval suite on world  |
| POST   | `/v1/worlds/{id}/verify`          | Generate ZK proof        |
| GET    | `/v1/worlds/{id}/scene-graph`     | Get scene graph          |

### Cross-Provider

| Method | Endpoint              | Description                    |
|--------|-----------------------|--------------------------------|
| POST   | `/v1/compare`         | Cross-provider comparison      |
| POST   | `/v1/evals/run`       | Run evaluation suite           |
| GET    | `/v1/evals/suites`    | List available eval suites     |

## Example Requests

### Create a World

```bash
curl -X POST http://localhost:8080/v1/worlds \
  -H "Content-Type: application/json" \
  -d '{"name": "kitchen", "provider": "cosmos"}'
```

Response:
```json
{
  "id": "wld_abc123",
  "name": "kitchen",
  "provider": "cosmos",
  "created_at": "2025-01-15T10:30:00Z"
}
```

### Run a Prediction

```bash
curl -X POST http://localhost:8080/v1/worlds/wld_abc123/predict \
  -H "Content-Type: application/json" \
  -d '{
    "action": {"move_to": [0.5, 0.8, 0.0]},
    "steps": 10
  }'
```

Response:
```json
{
  "prediction_id": "pred_xyz789",
  "physics_score": 0.87,
  "frames": 10,
  "world_state": { "...": "..." }
}
```

### Cross-Provider Comparison

```bash
curl -X POST http://localhost:8080/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "world_name": "kitchen",
    "action": {"move_to": [0.5, 0.8, 0.0]},
    "providers": ["cosmos", "runway", "sora"],
    "steps": 10
  }'
```

## Authentication

The REST API does not enforce authentication by default. Use a reverse proxy
(e.g., nginx, Envoy) to add auth in production deployments.

## OpenAPI Spec

The server generates an OpenAPI 3.1 specification at `/v1/openapi.json`.
