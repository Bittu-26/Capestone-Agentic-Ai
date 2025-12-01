"""
# AutoConcierge — Capstone Agent (Concierge Agents Track)

**Project pitch**
Writing detailed, well-researched articles or briefing notes is time-consuming. AutoConcierge is an automated, multi-agent concierge that: 
- accepts a user prompt for a topic (e.g., "Write a 1,200-word explainer on Vascular Dementia for caregivers"),
- uses parallel ResearchAgents to gather and synthesize facts from tools,
- composes a high-quality output via a WriterAgent,
- keeps session state and long-term memory to improve successive outputs,
- supports pause/resume for long-running research tasks,
- logs behavior for observability and provides a simple evaluation harness.

**Key concepts demonstrated (>=3)**
1. **Multi-agent system**: Coordinator agent orchestrates multiple ResearchAgents (parallel) and a WriterAgent (sequential composition).
2. **Tools**: WebSearchTool (pluggable), CodeExecutionTool (runs Python snippets), and a simple MCP-style tool interface.
3. **Sessions & Memory**: InMemorySessionService for session state and a MemoryBank persisted to `memory.json` for long term memory.
4. **Long-running operations**: pause/resume via checkpoints saved to disk.
5. **Observability**: structured logging and lightweight metrics collected to `metrics.json`.
6. **Agent evaluation**: EVALUATION.md contains simple automated tests and scoring heuristics.

**Deliverables**
- `main.py`: runnable agent system
- `requirements.txt`
- `README.md` (this text)
- `EVALUATION.md`

**Deadline note**: Submit by **December 1, 2025 11:59 AM PT** (as required).

"""
