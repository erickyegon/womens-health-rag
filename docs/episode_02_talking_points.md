# Episode 2 — Talking Points & On-Camera Script

**Title:** Document Ingestion — Cleaning & Chunking  
**Hook:** *"Your chunking strategy determines 80% of your RAG quality. Most people never tune it."*  
**Target runtime:** 20–24 minutes  
**GitHub branch:** `episode/02`

---

## Pre-roll (0:00 – 0:30) — no speaking

Show the terminal running:
```bash
python scripts/ingest.py --metadata data/metadata.json
```
Let the progress bars fill. Show the final output:
```
✅ Ingestion complete in 47.3s
   8,234 chunks indexed | 8,234 total in database
```
Cut to black. Title card: *"Here's every decision that went into those 8,234 chunks."*

---

## Hook (0:30 – 1:30)

"Last episode we loaded six DHS reports — 1,466 pages of dense public health data.
Today we turn those pages into something a vector database can actually search.

This is the most underrated episode in any RAG course. Everybody wants to talk
about agents and rerankers. Nobody talks about chunking. But I've seen RAG systems
fail completely because the chunking was wrong — and I've seen mediocre systems
become excellent just by fixing it.

Chunking determines the unit of retrieval. When a user asks a question,
the system finds the most relevant chunks and sends them to the LLM.
If your chunks are too big — too much noise, the answer gets diluted.
Too small — not enough context, the LLM hallucinates the gaps.
Cut through a sentence — the LLM gets half an idea and fills in the rest.

We're going to fix all of that. By the end of this episode you will have:
an 8-step cleaning pipeline, four chunking strategies benchmarked against each other,
and a full ingestion script that takes raw PDFs and produces indexed chunks in one command.
Let's go."

---

## Section 1 — The raw artefacts (1:30 – 5:00)

**On screen:** Open the notebook. Run the cell that shows raw page text.

"Before we clean anything, let's look at what we're actually dealing with.
This is page 42 of PR157 — the Nigeria DHS report — straight out of PyMuPDF."

**Point to each artefact in turn:**

1. **Unicode ligatures** — "See that ﬁ? That's not f-i. It's a single Unicode character
   called a ligature — U+FB01. If you search for 'significant' in your vector index
   and the text has 'signiﬁcant' with a ligature, you get zero results. We replace
   every ligature with its ASCII equivalent in step one."

2. **Hyphenated line breaks** — "This is 'attend-\\nance'. The PDF renderer broke the word
   across a line. The text extractor faithfully reproduced that break. Our embeddings
   would treat 'attend-' and 'ance' as separate tokens. One regex fixes this for
   every word in 1,466 pages."

3. **Footnote superscripts** — "Rate12. That 12 is a footnote reference. In the PDF it's
   a tiny superscript. After extraction it's indistinguishable from the number 12.
   We remove digits that appear immediately after a word character — carefully, so
   we don't remove real numbers like years or percentages."

4. **Repeated headers** — "DEMOGRAPHIC AND HEALTH SURVEY 2021. This appears at the top
   of every page. If we don't remove it, it appears in hundreds of chunks. The
   embedder would associate the phrase 'maternal mortality' with this header,
   because they co-occur so often. That poisons retrieval."

5. **Lone page numbers** — "87. That's just a page number. One digit on its own line.
   In a chunk, it's noise. We remove it."

6. **Table artefacts** — "Pipes, dots, repeated N/A. When a table doesn't extract cleanly,
   you get these. We filter lines that are purely structural."

"Six artefact types. Eight cleaning steps. Let's build them."

---

## Section 2 — The cleaning pipeline (5:00 – 10:00)

**On screen:** Open `cleaner.py`. Walk through each function briefly.

"The cleaner is eight pure functions. Each one takes a string and returns a string.
Pure means no side effects, no state. That makes each step independently testable.
If your cleaning breaks something, you know exactly which step to look at."

### Step 1 — Unicode normalisation

"We start with unicode normalisation to NFC. This recomposes decomposed codepoints
so subsequent regexes work correctly. Then we replace a map of typographic characters.
The ligature map covers the eight most common PDF ligatures plus typographic quotes,
dashes, non-breaking spaces, and control characters."

**On screen:** Show the `replacements` dict in `_normalise_unicode`.

"Notice this line: zero-width space U+200B goes to empty string — complete removal.
Non-breaking space U+00A0 goes to a regular space — preserved but normalised.
These are different decisions and they matter."

### Step 2 — Hyphenated linebreaks

**On screen:** Show the regex: `re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)`

"Alphabetic character, hyphen, newline, alphabetic character. Rejoin. One line.
The key insight: I only match alphabetic characters on both sides. Not digits.
This preserves year ranges like '2020-\\n2022' which are intentional numeric hyphens."

### Step 3 — Footnote numbers

**On screen:** Show the regex: `re.sub(r"(?<=\w)(\d{1,2})(,\d{1,2})*(?=[,.\s\n\t]|$)", "", text)`

"Lookbehind: word character. Then one or two digits. Then optional comma-separated
additional refs like ,2,3. Then lookahead: punctuation or whitespace.

The guards prevent over-removal: 2021 has a space before it, not a word character.
95% — the 95 is preceded by a space. Decimal 3.5 — the period is before the digit,
not after it. We're precise about what we remove."

### Steps 4-8 — Brief walkthrough

"The header remover checks each line: all-caps with length 10 to 90 — likely a header.
Lone digits — page numbers. Pipe-at-end-of-line patterns — chapter footers.
Table artefact remover: lines of only pipes and spaces, four or more repeated identical
tokens, lines of only asterisks or hashes. Junk character removal: replacement
characters from bad encoding, null bytes, dotted leaders collapsed to ellipses.
Short noise lines: anything under 3 characters that isn't an empty line.
Whitespace collapse: multiple spaces to one, 3-plus newlines to 2.
That double newline is sacred — it's the paragraph boundary signal we keep."

### Live demo — step by step

**On screen:** Run the `step_by_step_clean` cell.

"Watch the delta column. Unicode normalisation: minus 7 chars. Those were ligatures.
Header removal: minus 38 chars. That entire header line gone. Whitespace collapse:
minus 22 chars. Total noise removed: 9.4 percent.

And the critical thing: the real content — the statistics, the percentages, the
country data — all of it is still there. We removed noise, not information."

### Full corpus cleaning

**On screen:** Run `clean_pages(all_pages)`.

"1,466 pages in, 1,383 pages out. 83 filtered — 5.7 percent.
Those 83 are cover pages, blank pages, table-of-contents section dividers.
They contain no health data. Removing them makes retrieval cleaner and faster."

---

## Section 3 — Chunking strategies (10:00 – 17:00)

"Now we chunk. This is the part everyone skips and everyone regrets."

### FIXED — the baseline

**On screen:** Show FIXED chunk output. Point to a mid-sentence split.

"FIXED splits every 800 characters regardless of content. Watch what happens.
[Read the chunk aloud, trailing off at the mid-sentence cut]
The sentence continues in the NEXT chunk. The LLM gets half a thought.
If that thought was the key fact answering the user's question, retrieval
finds the right chunk but generation fills in the wrong answer.

FIXED is the baseline. We show it so you know what you're avoiding."

### RECURSIVE — the workhorse

**On screen:** Show RECURSIVE chunk. Show the same content, complete sentences.

"RECURSIVE has a separator priority list. Paragraph break first — the strongest signal.
Then line break. Then sentence endings. Then semi-colon — common in statistical prose
like '23%; up from 18%'. Then comma, word, character.

It tries to split on the highest-priority separator that respects the chunk size.
Usually that's a paragraph or sentence boundary. Watch the result:
[Read the chunk — complete sentence, complete thought].

Every chunk is a complete idea. That's what you want in a retrieval system."

### SENTENCE — for narrative documents

"The Ethiopia mini-report FR363 is narrative-heavy. Long paragraphs, almost no
double newlines. RECURSIVE gets confused — it tries to split on paragraph breaks
that don't exist and falls back to splitting mid-sentence.

For this document, SENTENCE strategy puts sentence endings at the top of the
priority list. The tradeoff: chunks may cross paragraph boundaries. But for a
narrative document, that's often the right call."

### The stats table

**On screen:** Run `compare_strategies`. Pause on the table.

"Total chunks: FIXED 8,847, RECURSIVE 8,234, SENTENCE 8,102.
Short chunks under 100 chars: FIXED 234, RECURSIVE 31, SENTENCE 28.
That short_chunks column is your noise indicator. FIXED produces 7x more noise chunks.

Average chars: similar across strategies. The difference is in the distribution —
RECURSIVE has a lower standard deviation, meaning more consistent chunk sizes.
Consistent sizes mean more predictable context window usage in the LLM.

**Decision: RECURSIVE for the course default.** It handles our mix of documents —
dense tables in Nigeria and Kenya, narrative in Ethiopia — better than the alternatives."

### Chunk size sensitivity

**On screen:** Run the sensitivity analysis cell.

"400 chars: 19,000 chunks. Too many — retrieval noise, higher embedding cost.
1200 chars: 6,800 chunks. Fewer API calls, but chunks exceed the sweet spot for
precision. The LLM gets too much context and gets confused about what's relevant.

800 chars is our sweet spot. Episode 9 will confirm this with RAGAS scores.
Until then: 800 chars, 150 overlap, RECURSIVE. Lock it in."

### Metadata — the differentiator

**On screen:** Show a chunk's metadata dict.

"Every chunk carries: source path, file name, page number, total pages, country,
year, report type, report title, chunk index, chunk count, char count, strategy name,
and a 16-character content hash for idempotent upserts.

This metadata is stored alongside the embedding in pgvector. And it's what makes
this system different from a generic RAG chatbot.

Any system can retrieve text. Only a system that attached country and year at
index time can answer: 'Show me data from Kenya between 2020 and 2022.'
We do that in Episode 7. But we make it possible right here."

### Parent-child — the preview

**On screen:** Show the parent-child pair output.

"Quick preview of something we build fully in Episode 15.
Parent chunks: 1,600 chars. Children: 400 chars. Each child carries a parent_id.

Retrieval happens at child level — small chunks, high precision.
Generation uses the parent — big chunk, full context for the LLM.

You get the precision of small chunks AND the context richness of large chunks.
It's the best of both worlds. We'll benchmark it against RECURSIVE in Episode 15
and show you exactly how much RAGAS improves."

---

## Section 4 — Full ingestion script (17:00 – 20:00)

**On screen:** Switch to terminal. Show `scripts/ingest.py --help`.

"The ingestion script wraps everything we just built into a single CLI command.
Let me walk through the key parts."

**Show the terminal output of `python scripts/ingest.py --dry-run`:**

"Dry run — no DB writes. This is what you run first to verify your pipeline.
Step 1: Loading. Step 2: Cleaning. Step 3: Chunking with stats.
Then it stops. Run this before every real ingestion to catch problems early."

**Run the full ingestion:**

"Now the real thing: `python scripts/ingest.py --metadata data/metadata.json`

Step 4: Embedding. Step 5: Upserting. Watch the progress bar fill.
47 seconds for 1,383 pages. That's about 34 pages per second — fast enough.

8,234 chunks indexed. Each one embedded, metadata attached, stored in pgvector.
Total cost: less than one cent in API fees.

You can also run this with the local ONNX backend:
`python scripts/ingest.py --backend onnx --metadata data/metadata.json`
Zero API cost. Episode 3 covers the quality tradeoff between the two."

---

## Section 5 — RAGAS test set (20:00 – 21:30)

**On screen:** Show the test_questions list in the notebook.

"I've set up a 30-question evaluation test set. We don't run RAGAS until Episode 9.
But we create the questions now, while we understand the corpus.

Five categories: factual single-hop, regional comparisons, education-health relationships,
multi-country queries, and difficult edge cases including one out-of-scope medical advice
question that our guardrails in Episode 16 should reject.

These 30 questions become our measurement system for the rest of the course.
Every technique we add — reranking, query rewriting, parent-child retrieval —
gets measured against this set. The RAGAS score table in docs/RAGAS_scores.md
tracks every improvement. That table is what you show in interviews."

---

## Wrap-up and next episode (21:30 – 23:00)

"What we built today:

One: An 8-step cleaning pipeline that handles every artefact in our DHS corpus.
Built as pure functions — independently testable, individually explainable.

Two: Four chunking strategies benchmarked on real data.
RECURSIVE wins. 800 chars, 150 overlap. Locked in.

Three: Parent-child pairs — the concept is introduced, full implementation in Episode 15.

Four: A full CLI ingestion script that takes raw PDFs and produces indexed chunks.

Five: A 30-question evaluation set — our measuring stick for the rest of the course.

**Next episode: Embeddings.**

We take those 8,234 chunks and turn them into vectors.
Two backends: OpenAI text-embedding-3-small at $0.02 per million tokens,
and a local ONNX model at zero cost.
We compare their outputs on health domain queries and decide when to use each.
And we talk about why 'maternal mortality' and 'mothers dying in childbirth'
end up at almost the same point in 1,536-dimensional space.

GitHub branch episode/02 is pushed — link in the description. The companion notebook
is there too. Questions in the comments. See you in Episode 3."

---

## Production notes

| Item | Detail |
|------|--------|
| Screen layout | Code on left 70%, terminal on right 30% throughout |
| Font | Jetbrains Mono 16px, One Dark Pro theme |
| Key visual | The `compare_strategies` table — pause 5 seconds on it |
| Key moment | FIXED vs RECURSIVE side-by-side chunk comparison at ~11:00 |
| Editing | Keep the dry-run terminal output in real time — don't cut |
| Chapters | 0:00 Pre-roll / 0:30 Hook / 1:30 Raw artefacts / 5:00 Cleaning pipeline / 10:00 Chunking strategies / 17:00 Ingestion script / 20:00 RAGAS test set / 21:30 Wrap-up |
| Thumbnail | Terminal showing the stats table. Text: "Chunking Strategies" — 2 words max |
| End screen | 15s — Subscribe + Episode 3 card |
| Upload | Friday 9:00 AM EST |
| Pin comment | GitHub branch link + `make ingest` command within 30 min |
