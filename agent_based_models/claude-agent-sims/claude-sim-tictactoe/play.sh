#!/bin/bash
# Orchestrates a Tic-Tac-Toe game between two Claude Code agents.
# Run with: bash play.sh
# Requires: claude CLI in PATH

set -euo pipefail

BOARD_FILE="$(dirname "$0")/board.md"

# ── Initialize board ──────────────────────────────────────────────────────────
cat > "$BOARD_FILE" << 'EOF'
BOARD (positions 1-9):
 1 | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9

CURRENT STATE:
   |   |
-----------
   |   |
-----------
   |   |

STATUS: IN_PROGRESS
EOF

echo "╔══════════════════════════════╗"
echo "║  Tic-Tac-Toe: Claude vs Claude ║"
echo "╚══════════════════════════════╝"
echo ""

# ── Per-player prompt template ────────────────────────────────────────────────
# $1 = player letter (A or B), $2 = symbol (X or O), $3 = opponent symbol
make_prompt() {
  local player="$1" symbol="$2" opponent="$3"
  cat << EOF
You are Player $player in a Tic-Tac-Toe game. Your symbol is $symbol. Opponent is $opponent.

TIME CONSTRAINT: Complete your entire move within 10 seconds. Be decisive — read the board, pick your move, write the file, and output your status line immediately. Do not over-think.

IMPORTANT RULES:
1. Read the board with Bash: run the command "cat $BOARD_FILE"
   - Do NOT use the Read tool — it caches and may show stale state.
2. Parse the CURRENT STATE section of the file.
   - Rows are separated by --- lines.
   - Each row has 3 cells separated by |.
   - Empty cells contain spaces, occupied cells contain X or O.
3. Choose any empty cell. Play optimally (win if possible, block if needed, else best strategic move).
4. Write the full updated board back to $BOARD_FILE using the Write tool.
   - Preserve the exact format (position reference, separator lines, CURRENT STATE section).
   - Update the STATUS line to one of: IN_PROGRESS | A_WINS | B_WINS | DRAW
5. Print the updated board to terminal.
6. As the VERY LAST line of your response, output exactly one of:
   MOVE_DONE
   GAME_OVER:A_WINS
   GAME_OVER:B_WINS
   GAME_OVER:DRAW

Win condition: 3 in a row horizontally, vertically, or diagonally.
Draw condition: all 9 cells filled with no winner.
EOF
}

# ── Timeout wrapper (macOS-compatible) ───────────────────────────────────────
# Usage: run_with_timeout <seconds> <output_var> cmd [args...]
# Sets output_var to the command output; returns non-zero on timeout/failure.
run_with_timeout() {
  local secs="$1" outvar="$2"; shift 2
  local tmpfile; tmpfile=$(mktemp)
  "$@" > "$tmpfile" 2>&1 &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $elapsed -ge "$secs" ]; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      rm -f "$tmpfile"
      return 1
    fi
  done
  wait "$pid"
  local rc=$?
  printf -v "$outvar" '%s' "$(cat "$tmpfile")"
  rm -f "$tmpfile"
  return $rc
}

# ── Game loop ─────────────────────────────────────────────────────────────────
MAX_MOVES=9
move=0
game_over=false

while [ $move -lt $MAX_MOVES ] && [ "$game_over" = false ]; do

  # ── Player A's turn (moves 0, 2, 4, 6, 8) ──
  echo "━━━ Player A (X) — Move $((move + 1)) ━━━"
  PROMPT_A=$(make_prompt "A" "X" "O")
  run_with_timeout 60 output_a claude --print "$PROMPT_A" \
    --allowedTools "Bash(cat *),Write" \
    --output-format text \
    --max-turns 6 || { echo "[Player A timed out — skipping turn]"; move=$((move + 1)); continue; }
  echo "$output_a"
  last_line_a=$(echo "$output_a" | grep -E '^(MOVE_DONE|GAME_OVER:.*)$' | tail -1 | tr -d '[:space:]')
  if [[ "$last_line_a" == GAME_OVER* ]]; then
    echo ""
    echo "🏆  $last_line_a"
    game_over=true
    break
  fi
  move=$((move + 1))

  [ $move -ge $MAX_MOVES ] && { echo "🤝  GAME_OVER:DRAW"; break; }

  # ── Player B's turn (moves 1, 3, 5, 7) ──
  echo ""
  echo "━━━ Player B (O) — Move $((move + 1)) ━━━"
  PROMPT_B=$(make_prompt "B" "O" "X")
  run_with_timeout 60 output_b claude --print "$PROMPT_B" \
    --allowedTools "Bash(cat *),Write" \
    --output-format text \
    --max-turns 6 || { echo "[Player B timed out — skipping turn]"; move=$((move + 1)); continue; }
  echo "$output_b"
  last_line_b=$(echo "$output_b" | grep -E '^(MOVE_DONE|GAME_OVER:.*)$' | tail -1 | tr -d '[:space:]')
  if [[ "$last_line_b" == GAME_OVER* ]]; then
    echo ""
    echo "🏆  $last_line_b"
    game_over=true
    break
  fi
  move=$((move + 1))

done

echo ""
echo "━━━ Final Board ━━━"
cat "$BOARD_FILE"
