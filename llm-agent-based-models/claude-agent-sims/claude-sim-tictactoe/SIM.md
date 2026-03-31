# Claude-vs-Claude Tic-Tac-Toe Simulation

## How to Run

From a plain terminal (not inside a Claude session):

```bash
cd ~/Desktop/Professional/personal_coding_projects/memory/agent_based_models/claude-sim-trial
bash play.sh
```

The game plays itself to completion and prints the result.

---

## How the Agents Interact

There are no persistent Claude sessions involved. Instead, `play.sh` is a shell script — a plain text file of terminal commands — that acts as an **orchestrator**. It runs `claude --print` once per turn, which spins up a fresh Claude process, gets a response, and exits. The two "agents" are just the same CLI called twice per round with different role instructions.

### Shared state via file

The only thing connecting the two agents is `board.md`. After each turn, the active agent overwrites the file with the updated board. On the next turn, the other agent reads the file fresh and responds.

```
play.sh (orchestrator)
   │
   ├── Turn 1: claude --print [Player A prompt]
   │              reads board.md via cat
   │              writes updated board.md
   │              outputs: MOVE_DONE
   │
   ├── Turn 2: claude --print [Player B prompt]
   │              reads board.md via cat
   │              writes updated board.md
   │              outputs: MOVE_DONE
   │
   └── ... repeat until GAME_OVER:* signal
```

### Why `cat` instead of the Read tool

Claude Code's `Read` tool caches file contents within a session. If an agent used `Read` to check `board.md`, it might get a stale version from earlier in the same call. `cat` via `Bash` always fetches from disk, so each agent always sees the current board state.

### Why sequential, not concurrent

The obvious alternative — two Claude processes running simultaneously, each polling `board.md` for changes — creates two problems:

1. **Race conditions**: both agents could try to write the file at the same time, corrupting it.
2. **Polling loops**: a long-running bash loop inside an agent session is fragile and can time out or be interrupted.

The sequential orchestrator avoids both. The script controls turn order; each agent simply reads, moves, and exits.

### How the orchestrator detects game over

Each agent is instructed to output a specific signal as the last line of its response — either `MOVE_DONE` or `GAME_OVER:A_WINS` / `GAME_OVER:B_WINS` / `GAME_OVER:DRAW`. The orchestrator parses this with `grep` and exits the loop when it sees a `GAME_OVER` signal.

---

## Permissions and Sandboxing

### What tools the agents can use

Each `claude --print` call is launched with `--allowedTools "Bash(cat *),Write"`. This means:

- **`Bash(cat *)`** — agents can only run `cat` commands via the shell. They cannot run scripts, delete files, make network requests, or execute anything else.
- **`Write`** — agents can write files. The prompt instructs them to only write `board.md`, but this is not enforced at the filesystem level.
- **No `WebFetch` or `WebSearch`** — no internet access.
- **`--max-turns 6`** — each agent's internal reasoning loop is capped at 6 steps, preventing runaway behavior.

### What's not restricted

`Bash(cat *)` is a tool-type filter, not a path filter. An agent could technically `cat` any file on the machine it knows the path to. Similarly, `Write` has no directory restriction — an agent following its instructions will only write `board.md`, but there is no hard OS-level enforcement of this.

The practical risk is low for a cooperative game like this. The agents have no instruction or incentive to access anything else, and the restricted `Bash` pattern prevents any destructive shell commands.

### Why Docker matters for more sophisticated sims

For simulations beyond simple games — agents that browse the web, write and execute code, manage files autonomously, or interact with external services — prompt-based restrictions are not sufficient. A misbehaving or confused agent could:

- Write files outside the intended directory
- Read sensitive files elsewhere on the machine
- Make unexpected network calls if given broader tool access

Docker solves this with **OS-level isolation**:

- The agent runs inside a container with its own filesystem. It physically cannot see or touch files on the host machine unless you explicitly mount them.
- Network access can be disabled or limited to specific endpoints at the container level, not just by omitting tools from a prompt.
- The container is disposable — if something goes wrong, you throw it away with no impact on the host.

For research simulations where agents have broad autonomy, Docker (or similar sandboxing) is the right default. For this tic-tac-toe sim, it's overkill.
