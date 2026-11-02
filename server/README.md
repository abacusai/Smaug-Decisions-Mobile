# Server

`decisions_server.py` serves a decision model, or the cascade, at `POST /v1/decisions`, with two demo pages. The guide is
[docs/server.md](../docs/server.md); the tests are in [`tests/test_server.py`](../tests/test_server.py).

```bash
pip install -e ".[server]"
python server/decisions_server.py --model abacusai/Smaug-Decisions --fast abacusai/Smaug-Audio-Decisions-0.7B --port 8000
```
