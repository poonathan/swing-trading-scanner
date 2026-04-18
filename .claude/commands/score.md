# Score a ticker

Run a full Deep Dive analysis on a single ticker from the CLI.

```bash
python main.py score $ARGUMENTS
```

**Usage:** `/score NVDA`

This scores the ticker across all 5 dimensions (Fundamental/Technical/Correction/News/Pattern) and prints the composite result with signal (BUY_DIP / WATCH / AVOID).

If no symbol is provided, ask the user which symbol to score.
