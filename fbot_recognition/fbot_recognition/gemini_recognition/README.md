### How to use

The gemini_recognition node is a drop-in replacement for the moondream_recognition / yolov8_recognition nodes, but instead of running a local model it queries a Gemini model through the [OpenRouter](https://openrouter.ai) API, which is faster and more powerful and does not require Google credits. It publishes the exact same `fbot_vision_msgs/msg/Detection3DArray` output as the other recognition nodes.

At startup, the node begins with no detection classes set. To start detecting objects, publish a `std_msgs/msg/String` message to `/fbot_vision/fr/object_prompt`. The message may contain either a single class, or **several classes separated by commas** (e.g. `coke, apple, cup`) — all requested classes are detected in a single API request per frame, and each returned detection carries its own label. You can change the classes by publishing another message, and set the message to `""` (empty string) to stop detection.

### Requirements

- Python package: `openai` (used as a generic OpenAI-compatible client pointed at OpenRouter).
- An OpenRouter API key, provided either through the `model.api_key` ROS parameter (see `config/gemini_object_recognition.yaml`) or through the `OPENROUTER_API_KEY` environment variable.

### Configuration

- `model.name` — OpenRouter model slug (default: `google/gemini-2.5-flash`). Any vision-capable model on OpenRouter works.
- `model.base_url` — API endpoint (default: `https://openrouter.ai/api/v1`). Point this at any OpenAI-compatible endpoint if desired.
- `model.api_key` — OpenRouter API key; falls back to the `OPENROUTER_API_KEY` env var when empty.
