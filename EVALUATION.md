# Evaluation Plan

1. **Smoke tests**: Run `python main.py --mode eval`. This verifies the multi-agent flow runs end-to-end (with mocked LLMs when no OPENAI_API_KEY present).

2. **Automated scoring**:
   - Use references (gold summaries) and compute simple similarity (BLEU, ROUGE) using `nltk`.
   - Provide human-in-the-loop review: rubric for factuality, clarity, and usefulness.

3. **Observability checks**:
   - Confirm `agent.log` contains relevant info.
   - Confirm `metrics.json` increments `tasks_started` / `tasks_completed`.

# Notes on real deployments
- Replace `WebSearchTool.search` with a real web search API; ensure you follow API quotas.
- Use a robust LLM client with streaming for better UX.
- Do NOT execute untrusted code in production.
