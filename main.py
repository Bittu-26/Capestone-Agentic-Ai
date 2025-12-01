
import os
import json
import time
import uuid
import asyncio
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Optional: import openai if available
try:
    import openai
except Exception:
    openai = None

# -------------------- Observability --------------------
LOGFILE = "agent.log"
METRICS_FILE = "metrics.json"
logging.basicConfig(level=logging.INFO, filename=LOGFILE, filemode="a",
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('AutoConcierge')

_metrics = {"tasks_started": 0, "tasks_completed": 0, "tokens_consumed": 0}

def inc_metric(k, v=1):
    _metrics[k] = _metrics.get(k, 0) + v
    with open(METRICS_FILE, 'w') as f:
        json.dump(_metrics, f, indent=2)

# -------------------- Memory Bank --------------------
MEMORY_FILE = 'memory.json'

class MemoryBank:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.mem = json.load(f)
        else:
            self.mem = {"notes": []}
            self._persist()

    def add(self, entry: Dict[str, Any]):
        self.mem['notes'].append(entry)
        self._persist()
        logger.info('MemoryBank: added entry')

    def query(self, keyword: str):
        return [n for n in self.mem['notes'] if keyword.lower() in json.dumps(n).lower()]

    def _persist(self):
        with open(self.path, 'w') as f:
            json.dump(self.mem, f, indent=2)

# -------------------- Session Service --------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"user_id": user_id, "history": [], "created_at": time.time()}
        logger.info(f"Session started {session_id} for {user_id}")
        return session_id

    def append(self, session_id: str, record: Dict[str, Any]):
        self.sessions[session_id]['history'].append(record)

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]


# -------------------- Tool Interfaces --------------------
@dataclass
class ToolResult:
    name: str
    success: bool
    output: Any

class WebSearchTool:
    """A pluggable web search tool. By default this is a **mock** that returns canned results.
    Replace `search()` to integrate a real search API (SerpAPI, Bing, Google) and add keys.
    """
    def __init__(self, real_api_key: Optional[str]=None):
        self.api_key = real_api_key

    async def search(self, query: str, limit=3) -> ToolResult:
        logger.info(f"WebSearchTool.search: {query}")
        # Mocked results for offline usage
        results = [
            {"title": f"{query} - overview (mock)", "snippet": f"Mocked snippet about {query}.", "url": "https://example.com/mock1"},
            {"title": f"{query} - study (mock)", "snippet": f"Mocked academic snippet on {query}.", "url": "https://example.com/mock2"}
        ][:limit]
        await asyncio.sleep(0.2)
        return ToolResult(name='web_search', success=True, output=results)

class CodeExecutionTool:
    def run(self, code: str) -> ToolResult:
        logger.info('CodeExecution: running snippet')
        try:
            # Danger note: executing untrusted code is unsafe. This demo assumes trusted snippets.
            local_vars = {}
            exec(code, {"__name__": "__main__"}, local_vars)
            return ToolResult(name='code_exec', success=True, output=local_vars)
        except Exception as e:
            return ToolResult(name='code_exec', success=False, output=str(e))

# -------------------- LLM Wrapper (pluggable) --------------------
class LLM:
    def __init__(self, model='gpt-4o-mini', temperature=0.2):
        self.model = model
        self.temperature = temperature
        self.api_key = os.environ.get('OPENAI_API_KEY')

    async def generate(self, prompt: str, max_tokens=512) -> str:
        # Minimal wrapper; if `openai` isn't installed, return a mock.
        logger.info('LLM.generate called')
        inc_metric('tokens_consumed', 10)
        if openai is None or self.api_key is None:
            # Return a predictable mock reply so users can run offline.
            return f"[MOCK LLM RESPONSE] Summary for: {prompt[:120]}..."
        else:
            # Async-friendly call to OpenAI (if available)
            loop = asyncio.get_event_loop()
            def call_openai():
                openai.api_key = self.api_key
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role":"user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )
                return resp
            resp = await loop.run_in_executor(None, call_openai)
            text = resp['choices'][0]['message']['content']
            return text

# -------------------- Agents --------------------
class ResearchAgent:
    def __init__(self, id: int, web_tool: WebSearchTool, llm: LLM):
        self.id = id
        self.web = web_tool
        self.llm = llm

    async def run(self, query: str) -> Dict[str, Any]:
        logger.info(f"ResearchAgent[{self.id}] running query: {query}")
        res = await self.web.search(query, limit=2)
        summary = await self.llm.generate("Summarize these results: " + json.dumps(res.output), max_tokens=150)
        return {"agent_id": self.id, "query": query, "results": res.output, "summary": summary}

class WriterAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    async def write(self, topic: str, research_summaries: List[str], length_words: int=800) -> str:
        prompt = (
            f"You are an expert writer. Write a {length_words}-word, well-structured article on '{topic}'.\n"
            f"Here are research summaries:\n" + "\n---\n".join(research_summaries) +
            "\n\nWrite the article with headings, bullet points where helpful, and a short TL;DR at the top."
        )
        logger.info('WriterAgent: generating article')
        article = await self.llm.generate(prompt, max_tokens=1024)
        return article

# -------------------- Coordinator (Multi-agent orchestration) --------------------
class Coordinator:
    def __init__(self, session_service: InMemorySessionService, memory: MemoryBank):
        self.session_service = session_service
        self.memory = memory
        self.web_tool = WebSearchTool()
        self.code_tool = CodeExecutionTool()
        self.llm = LLM()

    async def handle_request(self, user_id: str, topic: str, n_research_agents: int = 3, length_words: int = 800, pause_after_research: bool=False) -> Dict[str, Any]:
        inc_metric('tasks_started', 1)
        session_id = self.session_service.start_session(user_id)
        # 1) Launch research agents in parallel
        agents = [ResearchAgent(i, self.web_tool, self.llm) for i in range(n_research_agents)]
        research_queries = [f"{topic} background and recent developments",
                            f"Key statistics and trusted sources for {topic}",
                            f"Common misconceptions and caregiver advice about {topic}"]
        logger.info('Coordinator: starting parallel research')
        research_tasks = [agent.run(research_queries[i % len(research_queries)]) for i, agent in enumerate(agents)]
        research_results = await asyncio.gather(*research_tasks)

        # persist research to memory bank
        for r in research_results:
            self.memory.add({"topic": topic, "summary": r['summary'], "timestamp": time.time()})
            self.session_service.append(session_id, {"type": "research", "payload": r})

        if pause_after_research:
            # Save checkpoint and simulate long-running pause
            checkpoint = {"session_id": session_id, "status": "paused_after_research", "research": research_results}
            cpfile = f"checkpoint_{session_id}.json"
            with open(cpfile, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Coordinator: paused (checkpoint saved to {cpfile})")
            return {"status": "paused", "checkpoint": cpfile, "session_id": session_id}

        # 2) Sequential composition: WriterAgent composes final article
        research_summaries = [r['summary'] for r in research_results]
        writer = WriterAgent(self.llm)
        article = await writer.write(topic, research_summaries, length_words=length_words)

        # store final
        self.session_service.append(session_id, {"type": "final_article", "payload": article})
        inc_metric('tasks_completed', 1)
        logger.info('Coordinator: completed request')
        return {"status": "completed", "session_id": session_id, "article": article}

# -------------------- Simple CLI / Interactive wrapper --------------------
async def interactive_mode():
    print("AutoConcierge interactive mode — type 'quit' to exit")
    memory = MemoryBank()
    session_svc = InMemorySessionService()
    coord = Coordinator(session_svc, memory)
    while True:
        user = input("User id (type blank for 'local'): ") or 'local'
        topic = input("What do you want the concierge to write about? ")
        if topic.strip().lower() in ('quit', 'exit'):
            print('bye')
            break
        resp = await coord.handle_request(user, topic, n_research_agents=3, length_words=600)
        if resp['status'] == 'completed':
            print('\n==== ARTICLE (first 800 chars) ====')
            print(resp['article'][:2000])
            print('\n==== saved in memory and session ====')
        else:
            print('Paused — see checkpoint at', resp['checkpoint'])

# -------------------- Evaluation harness --------------------
def run_evaluation():
    # Very small smoke tests that check pieces run without crashing
    memory = MemoryBank()
    session_svc = InMemorySessionService()
    coord = Coordinator(session_svc, memory)
    print('Running synchronous smoke test (this uses mocks if no OpenAI key)')
    loop = asyncio.get_event_loop()
    start = time.time()
    result = loop.run_until_complete(coord.handle_request('test-user', 'Photosynthesis explained for 10-year-olds', n_research_agents=2, length_words=300))
    elapsed = time.time() - start
    print('Elapsed', elapsed)
    assert result['status'] == 'completed'
    print('Smoke test passed — article length', len(result['article']))

# -------------------- CLI entry --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['interactive', 'eval', 'demo'], default='demo')
    args = parser.parse_args()
    if args.mode == 'interactive':
        asyncio.run(interactive_mode())
    elif args.mode == 'eval':
        run_evaluation()
    else:
        # demo run
        asyncio.run(Coordinator(InMemorySessionService(), MemoryBank()).handle_request('demo-user', 'Climate change impacts on coastal cities', n_research_agents=3, length_words=500))
