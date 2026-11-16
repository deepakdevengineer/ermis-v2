# Validation rule reference

Every finding code the engine can produce, the CLAUDE.md section it comes from,
and its severity. Codes are stable identifiers — the UI and tests reference
them rather than message text.

Severity means:

- **ERROR** — a definite rule violation, visible in the text alone.
- **WARNING** — probable violation, or a decision only the audio can settle.
  Warnings do not make a transcription invalid.
- **INFO** — informational, no action needed.

---

## Formatting — §4, §34

| Code | Severity | Rule |
|---|---|---|
| `uppercase` | ERROR | §4.1 lowercase only |
| `forbidden-punctuation` | ERROR | §4.2 no `, ? ! ; :` or `"` |
| `sentence-period` | ERROR | §34 the period is reserved for spoken letters |
| `forbidden-symbol` | ERROR | §4.4 symbols are written as the spoken word |
| `forbidden-character` | ERROR | §35 no accents/diacritics for an English locale; no tabs or newlines |
| `double-space` | ERROR | §41 single space between tokens |
| `leading-trailing-space` | WARNING | §41 no extra leading/trailing space |
| `empty-transcription` | ERROR | §42 a clip needs text or a whole-file tag |

## Numbers — §4.3

| Code | Severity | Rule |
|---|---|---|
| `digit-used` | ERROR | numbers are written as the words that were spoken |

The suggested spelling is advisory only. `306` may be "three oh six", "three
zero six" or "three hundred six" — the audio decides, so the validator proposes
but never rewrites.

## Lexical — §4.5, §7, §15, §31, §32

| Code | Severity | Rule |
|---|---|---|
| `word-normalization` | ERROR | §4.5 `okay` → `ok` |
| `backchannel-spelling` | ERROR | §7 canonical spellings, incl. two-word forms |
| `honorific-abbreviated` | ERROR | §31 `mr.` → `mister` |
| `contraction-spelling` | WARNING | §32 missing apostrophe |
| `stretched-letters` | ERROR | §15 `haaaaaate` → `hate` |

`contraction-spelling` is a warning because §32 forbids forcing a contraction:
only the audio says whether the speaker said "don't" or "do not". Ambiguous
pairs that are also real words — `its`/`it's`, `were`/`we're`, `lets`/`let's` —
are never flagged.

## Tags — §8, §13, §38, §39, §41

| Code | Severity | Rule |
|---|---|---|
| `unknown-tag` | ERROR | §38 not in the V2 inventory |
| `retired-tag` | ERROR | §39 V1 tag; message names the V2 replacement |
| `pause-tag` | ERROR | §13 V2 defines no pause tag |
| `malformed-tag` | ERROR | §38 stray `[ ] < >` |
| `duplicate-tag` | ERROR | §8/§7 same sound in immediate succession uses one tag |
| `tag-order` | ERROR | §41 co-located tags in alphabetical order |

Retired tags recognised: `[cough]` `[sigh]` `[throat-clear]` `[sneeze]`
`[lip-smack]` `[yawn]` `[breath]` `[hum]` `<hum>` → all `[hn]`; and `<i>`
`<sg>` `<ws>` `<pws>` `<lv>`, removed with no replacement.

`[bg]` is exempt from `tag-order`. §20 documents `[laughter] [bg]` as correct
and §21 places `[bg]` positionally, so adjacent tags around it are a sequence,
not one location.

## Whole-file conditions — §10–§12, §35, §36, §42

| Code | Severity | Rule |
|---|---|---|
| `whole-file-not-alone` | ERROR | §42 `<ga> <na> <ns> <nt> <sc>` must stand alone |
| `multiple-whole-file-tags` | ERROR | §42 only one condition can apply |

`(())`, `<ct>`, `[laughter]` and `[hn]` are valid as a complete transcription
*and* valid inline — §14's `(()) but i don't know what to get` and §19's
`<ol> … </ol> <ct>` are both correct. Only the five angle tags are exclusive.

## Speakers — §17, §18A, §22

| Code | Severity | Rule |
|---|---|---|
| `speaker-single-tagged` | ERROR | §17 single-speaker audio takes no `<s1>` |
| `speaker-numbering` | ERROR | §18A.2 numbering follows first appearance, no gaps |
| `speaker-unknown-reference` | ERROR | §18A.10 tag refers to a speaker not in the panel |
| `speaker-repeated-turn` | WARNING | §17 re-tagged with no speaker change |
| `speaker-too-many` | WARNING | §22 treat a large group as one speaker |

`speaker-single-tagged` is suppressed when `<ol>` or `<ct>` is present: both
denote speech from more than one speaker, so `<s1> hello <ct>` is well-formed.
Pass `speaker_count=1` to assert single-speaker audio explicitly.

`speaker-unknown-reference` only fires when `known_speakers` is supplied from
the speaker panel.

## Overlap and cross talk — §18, §19

| Code | Severity | Rule |
|---|---|---|
| `overlap-unclosed` | ERROR | every `<ol>` is closed |
| `overlap-unmatched-close` | ERROR | `</ol>` with no opener |
| `overlap-nested` | ERROR | blocks cannot nest |
| `overlap-empty` | ERROR | a block has content |
| `overlap-missing-speakers` | ERROR | §18 at least two speakers, each tagged inside |
| `overlap-no-leading-speaker` | ERROR | §18 content begins with a speaker tag |
| `cross-talk-in-overlap` | ERROR | §19 close the block, then use `<ct>` |

## Unintelligible and mis-spoken — §14, §15

| Code | Severity | Rule |
|---|---|---|
| `single-parentheses` | ERROR | §14 use `(())`, not `()` |
| `unintelligible-spacing` | ERROR | §14 no spaces directly inside `(())` |
| `unintelligible-malformed` | ERROR | §14 unclosed marker |
| `misspoken-empty` | ERROR | §15 `{{}}` must contain the word |
| `misspoken-malformed` | ERROR | §15 unclosed, or spaces inside |

Text inside `(())` and `{{}}` is ordinary transcription and obeys the character
rules, so `((Songs))` and `((song5))` are flagged.

## Truncation — §6

| Code | Severity | Rule |
|---|---|---|
| `multi-hyphen` | ERROR | exactly one hyphen marks a truncation |
| `stray-hyphen` | ERROR | a hyphen attaches to the audible fragment |

Compounds and brand names keep their hyphen: `twenty-three`, `t-pain`,
`oral-b`, `mm-hm`. Whether a hyphen marks a *genuine* cut-off is audible only,
so the validator trusts the annotator and reports nothing.

## Letter sequences — §23, §24, §33

| Code | Severity | Rule |
|---|---|---|
| `letter-sequence-spacing` | ERROR | §23 letter, period, space: `o. p. e. c.` |
| `period-after-i` | WARNING | §23 no period after the one-letter word `i` |

`period-after-i` is a warning, and is suppressed when a neighbouring token is
also a letter-sequence token — in `i. b. m.` the period is correct.

## Audio cross-check — §49

| Code | Severity | Rule |
|---|---|---|
| `event-not-transcribed` | WARNING | a **confirmed** event has no corresponding tag |
| `pause-detected` | INFO | a pause is metadata and is not inserted as a tag |

Produced by `app/ermis/crosscheck.py`, which needs analysis events as well as
text. Wired into the main window, which re-runs it after every confirmation.

Only **confirmed** events are cross-checked. Warning that an *unconfirmed*
detection is "missing" would pressure the annotator into writing a tag for
something they have not agreed is there -- the failure mode §47 and §56.22
exist to prevent. Cross-check findings are WARNING and INFO only, so they never
make a transcription invalid: the annotator's ears outrank the detector.

`pause-detected` is phrased as a prohibition. A pause can never be *missing*,
because V2 has no pause tag (§13).

---

## Adding a rule

1. Add the tag or lexical entry to `tags.py` or `rules.py`. Nothing outside
   those two modules may hard-code a tag literal or spelling.
2. Add a `Code` member in `findings.py`.
3. Implement the check in `validator.py`, citing the section in `rule_ref`.
4. Add a trigger to `TRIGGERS` in `tests/test_coverage.py` — the suite fails if
   a code has no input that produces it.
5. Add the rule's own tests to the matching `tests/test_*.py`.
6. If the rule touches documented-correct text, extend the false-positive
   corpus in `test_coverage.py`.

Before adding a check, ask whether the text alone can settle it. If it needs the
audio, make it a `WARNING` or leave it to the human-confirmation workflow. §59:
audio is the source of truth.

---

## Analysis engine reference (Phases 3 and 7)

Detectors produce **analysis events**, never tags. Every `possible_*` event
requires human confirmation before it can contribute to a transcription
(§46, §47). Tuning constants are module-level and documented in place.

| Detector | Module | Event type | Confidence cap |
|---|---|---|---|
| Voice activity | `detection/vad.py` | `speech` | 0.95 |
| Silence / pause | `detection/silence.py` | `silence`, `pause` | 1.0 (measured) |
| Audio quality | `audio/quality.py` | `no_audio`, `garbled_audio`, `no_speech` | 1.0 / 0.85 / 0.6 |
| Laughter | `detection/laughter.py` | `possible_laughter` | 0.70 |
| Non-lexical | `detection/nonlexical.py` | `possible_hn` | 0.65 |
| Artifact | `detection/artifact.py` | `possible_artifact` | 0.75 |
| Background | `detection/background.py` | `possible_background` | 0.60 |
| Overlap | `detection/overlap.py` | `possible_overlap` | 0.55 |
| Diarization | `speakers/diarization.py` | `speech` + `speaker_change` | 0.80 (count) |

### Rules the detectors enforce structurally

Rather than relying on the annotator to remember them:

- **Laughter is never `[hn]`** (§8). The pipeline runs the laughter detector
  first, and the `[hn]` detector drops candidates overlapping a laughter one.
- **`[artifact]` is technical noise only** (§9). The artifact detector drops
  candidates overlapping any vocal candidate, so a cough is not proposed as an
  artifact.
- **One sound, one tag** (§8). Adjacent `[hn]` candidates within 0.45 s merge
  into a single event, so two coughs yield one `[hn]` — case B of §55.
  Adjacent artifact runs merge too.
- **A pause never carries a tag** (§13). `PAUSE` maps to `None` in the tag
  table, `confirm("[pause]")` raises, the schema `CHECK` rejects it, and the
  event panel disables the confirm control for pauses entirely.
- **`<ns>` yields to another applicable tag** (§12). The quality analyser is
  told whether a non-lexical candidate was found and withholds `<ns>` if so.
- **Steady noise is not `<ga>`** (§10, §12). Poor SNR only points to `<ga>`
  when speech was detected; steady broadband noise with no speech proposes
  `<ns>`, since §12 lists "faint white noise" as exactly that case.
- **Speaker numbering follows first appearance** (§18A.2), never loudness,
  gender, confidence or cluster size — and it is recomputed after every merge,
  split, add or remove so no numbering gap can persist.
- **Automatic passes never overwrite human metadata** (§18A.11). `human_edited`
  is checked in `Database.upsert_speaker` and preserved across merges.

### Rules the detectors deliberately refuse to decide

Each of these is surfaced with both options stated in the event note:

- `[artifact]` vs `<ga>` — §9's rule is "can speech still be transcribed?",
  which is a judgement about intelligibility.
- `<ol>` vs `<ct>` — §19 splits on whether the overlap is understandable.
- `[bg]` attribution — §21 turns on whether the source is an *identified*
  speaker, which only the annotator knows.
- Gender and nativity — §18A.3, §18A.4 and §37 forbid inferring these. They
  stay `unknown` until a human sets them, and no detector writes them.
