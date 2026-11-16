# Ermis V2 Local Audio Annotation & QA Assistant
## Project Specification for Claude Code

> **ACTIVE DEVIATION (2026-08-28): sections 1, 18A.4, 46, 47 and 56.1 are no
> longer what the code does.** On the project owner's explicit instruction, the
> application now drafts the transcription from the audio with a local speech
> recogniser, auto-inserts confident event tags, and suggests speaker nativity
> from fluency features -- instead of requiring every word, every tag and every
> descriptor to come from a person. The rest of this specification still
> governs: spoken form, the tag inventory, whole-file logic, the validator, and
> the audit trail are all unchanged. See section 60 at the end of this file for
> exactly what changed, what guarantees remain, and how to switch it off.

## 1. PROJECT OBJECTIVE

Build a LOCAL Windows application that assists a human annotator/verifier with
Ermis Transcription V2 audio annotation and quality assurance.

IMPORTANT:
- This project must follow Ermis Transcription V2 rules exactly.
- The official workflow prohibits using Automatic Speech Recognition (ASR)
  to produce the transcription.
- Therefore, DO NOT implement an automatic ASR-to-final-transcription pipeline.
- The application may perform audio analysis, segmentation, silence detection,
  event detection, speaker analysis, validation, and QA assistance.
- Final transcription must remain human-entered/reviewed.
- The application must never silently invent words or tags.

The application should help a human produce a correct Ermis V2 transcription
and detect possible rule violations before submission.

--------------------------------------------------
## 2. PRIMARY GOALS
--------------------------------------------------

The application should provide:

1. Local audio playback.
2. Waveform/timeline visualization.
3. Audio segmentation.
4. Speech/non-speech detection for navigation and QA.
5. Pause/silence detection as INTERNAL metadata only.
6. Detection/flagging of non-lexical sounds.
7. Detection/flagging of laughter.
8. Detection/flagging of brief audio artifacts.
9. Detection/flagging of possible background speech.
10. Speaker segmentation/diarization assistance.
11. Overlapping speech detection assistance.
12. Manual transcription editor.
13. Ermis V2 tag insertion assistance.
14. Real-time transcription-rule validation.
15. Whole-file condition validation.
16. Speaker metadata collection.
17. Export of annotation/QA results.
18. Completely local processing by default.

--------------------------------------------------
## 3. IMPORTANT DISTINCTION:
## AUDIO ANALYSIS VS FINAL TRANSCRIPTION
--------------------------------------------------

The system has TWO layers.

### Layer A: Audio Analysis

The application may analyze audio and produce internal events such as:

- SPEECH
- SILENCE
- PAUSE
- POSSIBLE_LAUGHTER
- POSSIBLE_NON_LEXICAL_SOUND
- POSSIBLE_ARTIFACT
- POSSIBLE_BACKGROUND_SPEECH
- POSSIBLE_OVERLAP
- SPEAKER_CHANGE
- AUDIO_GARBLED
- NO_AUDIO_SIGNAL
- NO_SPEECH

These are analysis events.

Example:

00:00.00 - 00:03.40  SPEECH
00:03.40 - 00:05.20  PAUSE
00:05.20 - 00:07.90  SPEECH
00:07.90 - 00:08.60  POSSIBLE_LAUGHTER

### Layer B: Ermis Transcription

The human annotator enters the final transcription.

The application validates the transcription against Ermis V2 rules.

DO NOT automatically convert:

SPEECH -> ASR text -> final Ermis transcription

This is prohibited for the official workflow.

--------------------------------------------------
## 4. SPOKEN-FORM FUNDAMENTALS
--------------------------------------------------

Final transcription must follow spoken-form rules.

### 4.1 Lowercase

Use lowercase letters only.

Correct:
i want to play music

Incorrect:
I Want To Play Music

### 4.2 No normal sentence punctuation

Do not use:
- commas
- question marks
- exclamation marks
- normal sentence periods

Allowed punctuation/symbols are defined by the Ermis rules.

### 4.3 Numbers

Numbers must be written as words.

Correct:
twenty-three
five minutes
three hundred

Incorrect:
23
5 minutes
300

Preserve how the number is spoken.

Examples:
three oh six
three zero six

### 4.4 Symbols

Convert spoken symbols to words.

Examples:
$ -> dollars
% -> percent
& -> and

### 4.5 "ok"

Normalize "okay" to:

ok

### 4.6 Do not clean up speech

Preserve:
- grammatical errors
- repetitions
- mistaken words
- fillers
- backchannels
- stutters
- non-standard grammar

Example:

Audio:
he don't like it

Correct:
he don't like it

Do NOT correct grammar.

--------------------------------------------------
## 5. REPETITIONS
--------------------------------------------------

Every clearly spoken repetition must be retained.

Audio:
very very very much

Correct:
very very very much

Do not collapse repetitions.

Whole-word stutter:

Audio:
i i i i i want to go

Correct:
i i i i i want to go

Do not add hyphens to fully voiced repeated words.

--------------------------------------------------
## 6. TRUNCATIONS / STUTTERS
--------------------------------------------------

Use "-" only when a word/sound is actually cut off.

Examples:

elev-

sh- song

wh- wh- what

Rules:

- Preserve the audible fragment.
- Put the hyphen exactly at the truncation.
- Do not add a hyphen merely because speech is fast.
- A fully audible word gets no truncation hyphen.
- A self-correction gets a hyphen only when the first attempt is actually
  truncated.

Example:

play that sh- song

NOT:

play that sh song

NOT:

play that sh-- song

A continuation can begin with a hyphen:

-ey stop

--------------------------------------------------
## 7. BACKCHANNELS AND FILLED PAUSES
--------------------------------------------------

These are important and must NOT be removed.

Use the following exact spellings:

| Audio / Intent | Output |
|---|---|
| closed-mouth agreement | mm-hm |
| open-mouth agreement | uh-huh |
| thinking | hmm |
| short open-vowel hesitation | uh |
| closed-lip/nasal hesitation | um |
| closed-mouth disagreement | mm-mm |
| open-mouth disagreement | nuh-huh |
| surprise/realization | oh |
| excitement/impressed | ooh |
| unidentifiable filled pause | [fp] |

Important:

mm-hm
NOT mhm

uh-huh
NOT uh huh

mm-mm
NOT mm mm

nuh-huh
NOT nuh-uh

hmm
NOT hm
NOT hmmm

uh
NOT uhh

um
NOT umm

ooh
NOT ohh

If the exact filled-pause vowel cannot be determined:

[fp]

If two filled pauses occur back-to-back and are both [fp]:

use one [fp].

--------------------------------------------------
## 8. NON-LEXICAL VOCAL SOUNDS
--------------------------------------------------

### Laughter

Use:

[laughter]

Laughter is a separate tag.

Do NOT convert laughter to [hn].

### Other non-lexical vocal sounds

Use:

[hn]

This includes:
- cough
- sigh
- throat-clear
- sneeze
- lip-smack
- yawn
- cry
- groan
- humming
- noticeable sharp inhalation/exhalation

Do NOT use [hn] for ordinary breathing.

Examples:

play [hn] music

i'm [hn] so tired

If the same sound occurs repeatedly in immediate succession:

use ONE tag.

Example:

two coughs

Correct:
play [hn] music

NOT:
play [hn] [hn] music

For ongoing non-lexical sound:

place one tag where it begins.

--------------------------------------------------
## 9. AUDIO ARTIFACT
--------------------------------------------------

Use:

[artifact]

for brief technical audio noise such as:
- pop
- click
- static burst
- feedback squeal
- momentary glitch

Example:

play [artifact] music

Important distinction:

[artifact]
=
brief isolated technical noise where surrounding speech remains usable.

<ga>
=
recording/microphone distortion is severe enough to compromise evaluation.

Decision rule:

Can meaningful speech still be transcribed around the noise?

YES -> [artifact]

NO -> <ga>

Do not use [artifact] for persistent background hiss/hum.

--------------------------------------------------
## 10. GARBLED AUDIO
--------------------------------------------------

Use:

<ga>

when recording/microphone problems make the audio too distorted,
damaged, or skipping to evaluate properly.

If the recording is fundamentally compromised:

Final transcription:
<ga>

Nothing else should accompany the whole-file <ga> condition.

Do NOT use <ga> for intentionally pitch-shifted audio used for privacy.

--------------------------------------------------
## 11. NO AUDIO
--------------------------------------------------

If the file contains no audio signal at all:

<na>

This is a whole-file condition.

Do not add anything else.

--------------------------------------------------
## 12. NO SPEECH
--------------------------------------------------

If there is audio but no human speech and no other applicable tag:

<ns>

This is a whole-file condition.

Examples:
- room noise only
- faint white noise
- non-human/environmental sound without another applicable tag

Do NOT use <ns> if another applicable tag exists.

Do NOT use <ns> merely because there is a pause.

--------------------------------------------------
## 13. PAUSE / SILENCE
--------------------------------------------------

IMPORTANT:

Ermis V2 does NOT define a transcription tag for normal pauses.

The application may detect silence/pause internally.

Example internal event:

{
  "type": "pause",
  "start": 3.42,
  "end": 5.18,
  "duration": 1.76
}

But DO NOT insert:

[pause]

into the final Ermis transcription.

Pause detection is an analysis/QA feature only.

--------------------------------------------------
## 14. UNINTELLIGIBLE SPEECH
--------------------------------------------------

Use double parentheses:

(())

for genuinely unintelligible speech.

No spaces inside the parentheses.

Examples:

play ()

is NOT correct.

Correct:

play (())

If a reasonable guess can be made:

play ((songs))

If only part of a word is audible:

play ((musi-))

Rules:

- Replay audio.
- Slow audio if necessary.
- Use context.
- Research unfamiliar words when appropriate.
- Use (()) only when genuinely unclear.
- Do not use (()) for audio recording problems.
- Use <ga> for recording/microphone problems.

If the entire file is unintelligible speech:

(())

alone.

If some words are understandable:

(()) but i don't know what to get

--------------------------------------------------
## 15. MIS-SPOKEN WORDS
--------------------------------------------------

If the speaker mispronounces a word but the intended word is unambiguous,
normalize to the dictionary spelling.

Example:

mushic -> music

libary -> library

Do NOT reproduce stretched letters.

Example:

haaaaaate

becomes:

hate

If the intended word is genuinely ambiguous:

use:

{{word}}

Example:

{{authentication}}

or another plausible dictionary word.

Double curly braces are for ambiguous mis-spoken words.

They are NOT for:
- normal unintelligible speech
- audio artifacts
- recording problems
- accents/dialects

--------------------------------------------------
## 16. NON-STANDARD SPEECH AND GRAMMAR
--------------------------------------------------

Transcribe what the speaker actually says.

Examples:

he don't like that song

put a apple on my list

Do NOT grammatically correct the speaker.

Mistaken words should remain if they are actually what the speaker said,
unless the special mis-spoken-word normalization rule applies.

--------------------------------------------------
## 17. MULTIPLE SPEAKERS
--------------------------------------------------

Speaker tags:

<s1>
<s2>
<s3>
...

Assign speaker numbers according to first appearance.

Example:

<s1> play <s2> songs

Speaker tags appear at the beginning of each speaker turn.

IMPORTANT:

Single-speaker audio does NOT get <s1>.

Correct:

play music

NOT:

<s1> play music

--------------------------------------------------
## 18. OVERLAPPING SPEECH
--------------------------------------------------

Use:

<ol>
...
</ol>

for intelligible simultaneous speech.

Example:

<ol> <s1> stop <s2> stop </ol>

Every speaker must be re-tagged inside the overlap block.

Example:

<s1> set a timer <ol> <s2> how long <s1> for five minutes </ol>

--------------------------------------------------
--------------------------------------------------
## 18A. SPEAKER PROFILE & METADATA
--------------------------------------------------

Speaker metadata is a REQUIRED part of the application.

The application must maintain a speaker profile for every identified speaker.

### 18A.1 Number of Speakers

The application must display:

Total Speakers: N

Example:

Total Speakers: 2

Speaker 1
Speaker 2

The number of speakers must be derived from the audio analysis/diarization
and must remain editable by the human annotator.

IMPORTANT:
- Do not blindly trust automatic speaker-count detection.
- The user must be able to merge two incorrectly separated speakers.
- The user must be able to split one incorrectly merged speaker.
- The user must be able to add/remove a speaker manually.
- Speaker IDs must remain consistent after editing.

--------------------------------------------------
## 18A.2 Speaker IDs
--------------------------------------------------

Speakers must be represented internally as:

s1
s2
s3
...

Final transcription uses:

<s1>
<s2>
<s3>

Speaker numbering follows FIRST APPEARANCE in the audio.

Example:

First speaker heard -> s1
Second speaker heard -> s2
Third speaker heard -> s3

Do not assign speaker numbers based on:
- loudness
- gender
- nativity
- confidence
- estimated importance

--------------------------------------------------
## 18A.3 Speaker Gender
--------------------------------------------------

For each speaker provide:

gender:
  msv
  fsv
  unknown

Definitions:

msv = masculine-sounding voice
fsv = feminine-sounding voice
unknown = cannot reliably determine

IMPORTANT:

Gender classification must be treated as a voice-sound descriptor,
not as a factual identity claim.

The system must NOT force a classification.

If confidence is insufficient:

unknown

The human annotator must be able to override the automatically suggested value.

Example:

Speaker 1
ID: s1
Gender: msv

Speaker 2
ID: s2
Gender: fsv

Speaker 3
ID: s3
Gender: unknown

--------------------------------------------------
## 18A.4 Speaker Nativity
--------------------------------------------------

For each speaker provide:

nativity:
  native
  nonnative
  unknown
  tts

Definitions:

native:
Native English speaker.

nonnative:
Non-native English speaker.

unknown:
Cannot reliably determine.

tts:
Computer-generated / synthetic / TTS speech.

IMPORTANT:

Nativity must NOT be guessed from:
- audio quality
- microphone quality
- volume
- accent alone when evidence is insufficient
- background noise
- short non-speech sounds

If uncertain:

unknown

Human annotator must be able to override the suggested classification.

--------------------------------------------------
## 18A.5 Speaker Profile Object
--------------------------------------------------

Each speaker should be represented internally as:

{
  "speaker_id": "s1",
  "display_name": "Speaker 1",
  "gender": "unknown",
  "nativity": "unknown",
  "first_seen": 2.31,
  "last_seen": 47.82,
  "confidence": 0.87
}

Example:

{
  "speaker_id": "s2",
  "display_name": "Speaker 2",
  "gender": "fsv",
  "nativity": "native",
  "first_seen": 8.41,
  "last_seen": 51.22,
  "confidence": 0.91
}

--------------------------------------------------
## 18A.6 Speaker Detection Workflow
--------------------------------------------------

Audio:

00:00 - 00:05 -> Speaker A
00:05 - 00:08 -> Speaker B
00:08 - 00:12 -> Speaker A

Application:

Total Speakers: 2

Speaker 1
- ID: s1
- First seen: 00:00
- Last seen: 00:12
- Gender: unknown
- Nativity: unknown

Speaker 2
- ID: s2
- First seen: 00:05
- Last seen: 00:08
- Gender: unknown
- Nativity: unknown

Transcription structure:

<s1> ...
<s2> ...
<s1> ...

--------------------------------------------------
## 18A.7 Speaker Timeline
--------------------------------------------------

The UI must show speaker activity on the audio timeline.

Example:

00:00 ───────────────────────────── 00:20

S1  █████████       ███████████
S2          ███████

This allows the annotator to verify:
- speaker changes
- speaker turns
- possible overlap
- incorrect diarization

--------------------------------------------------
## 18A.8 Speaker Merge / Split
--------------------------------------------------

The user must be able to correct diarization.

### Merge

If system incorrectly identifies:

s1
s2

as two speakers but they are actually the same person:

Merge -> s1

All associated events and transcription segments must be updated.

### Split

If system incorrectly identifies one speaker as:

s1

but there are actually two people:

Split into:

s1
s2

The split must update:
- timeline
- speaker assignments
- transcription tags
- speaker metadata

--------------------------------------------------
## 18A.9 Speaker Metadata UI
--------------------------------------------------

Create a dedicated Speaker Panel.

Example:

-----------------------------------------
SPEAKERS
-----------------------------------------

Total Speakers: 3

Speaker 1
ID: s1
Gender: [msv ▼]
Nativity: [native ▼]
Confidence: 92%

Speaker 2
ID: s2
Gender: [unknown ▼]
Nativity: [nonnative ▼]
Confidence: 81%

Speaker 3
ID: s3
Gender: [fsv ▼]
Nativity: [unknown ▼]
Confidence: 74%

[Merge Speakers]
[Split Speaker]
[Add Speaker]
-----------------------------------------

All fields must be editable.

--------------------------------------------------
## 18A.10 Validation
--------------------------------------------------

The QA engine must validate:

1. Speaker count.
2. Speaker IDs.
3. First-appearance numbering.
4. Missing speaker metadata.
5. Invalid gender value.
6. Invalid nativity value.
7. Speaker tags that reference nonexistent speakers.
8. Speaker IDs skipped unexpectedly.
9. Incorrect speaker changes.
10. Speaker tags missing around speaker turns.
11. Speaker tags inside overlap blocks.
12. Inconsistent speaker assignment.

Allowed gender values:

msv
fsv
unknown

Allowed nativity values:

native
nonnative
unknown
tts

No other values are allowed.

--------------------------------------------------
## 18A.11 Important Rule
--------------------------------------------------

Automatic speaker detection is ASSISTIVE ONLY.

The system must show confidence and allow human correction.

Example:

Detected:
Speaker count = 3
Confidence = 76%

The user can change:

Speaker count = 2

The application must preserve the user's final decision.

Never silently overwrite human speaker assignments.
## 19. CROSS TALK
--------------------------------------------------

Use:

<ct>

for overlapping speech from multiple speakers that cannot be understood.

If the entire clip is unintelligible overlapping speech:

<ct>

alone.

If speech starts intelligibly and becomes unintelligible:

close the <ol> block first, then use <ct>.

Correct structure:

<ol> <s1> wait <s2> no </ol> <ct>

Do NOT nest <ct> inside <ol>.

Important:

<ct> means unintelligible overlapping speech.

Do NOT use <ct> merely because speaker order is uncertain.

--------------------------------------------------
## 20. SIMULTANEOUS SPEAKER SOUNDS
--------------------------------------------------

If identified speakers make non-lexical sounds simultaneously,
use <ol> and identify each speaker.

Example:

<ol> <s1> [laughter] <s2> [laughter] </ol>

If the sound comes from an unidentified person:

[bg]

Example:

[laughter] [bg]

--------------------------------------------------
## 21. BACKGROUND SPEECH / SOUNDS
--------------------------------------------------

Use:

[bg]

when the source cannot be attributed to an identified speaker.

This includes:
- background speech
- background laughter
- background coughing
- background sneezing
- other non-lexical sounds from unidentified people

Do NOT use [bg] just because speech is quiet or distant.

If the speaker is identifiable, use their speaker tag.

If an already identified speaker becomes temporarily unintelligible:

do NOT convert them to [bg].

Use their speaker tag and appropriate unintelligible treatment.

For continuous background speech:

place [bg] where it becomes noticeable, according to the transcription rules.

If an unidentified sound occurs during an identified speaker's speech,
place [bg] appropriately after the interrupted content.

--------------------------------------------------
## 22. LARGE GROUP / CROWD
--------------------------------------------------

For large groups (15+ people) responding simultaneously:

- If the response is intelligible but individuals cannot be distinguished,
  treat the crowd as one speaker.
- Use a new speaker tag for the crowd.
- If unintelligible, use <ct>.
- Do NOT create a separate speaker tag for every crowd member.

Example:

<s1> ready <s2> yes

where <s2> represents the crowd.

--------------------------------------------------
## 23. LETTER SEQUENCES
--------------------------------------------------

Individual spoken letters:

letter + period + space

Example:

a. b. c.

Example:

t. o. r.

Do NOT add a period after a one-letter word such as:

i

--------------------------------------------------
## 24. ACRONYMS
--------------------------------------------------

If pronounced as a word:

opec

If pronounced as individual letters:

o. p. e. c.

The audio pronunciation determines the transcription.

--------------------------------------------------
## 25. ALPHANUMERIC SEQUENCES
--------------------------------------------------

For codes/flight numbers/serial numbers:

- letters follow letter-sequence rules
- numbers follow number rules
- preserve spoken order

Example:

ABC123 spoken as individual letters and numbers:

a. b. c. one two three

Example:

UA729:

u. a. seven twenty-nine

Example:

X9R2K:

x. nine r. two k.

--------------------------------------------------
## 26. PROPER NOUNS / CATALOG ENTITIES
--------------------------------------------------

Catalog entities include:
- songs
- albums
- artists
- movies
- books
- apps
- channels
- cities
- countries
- brands
- other searchable titled entities

Use official/common spelling where appropriate.

Remove disallowed branding characters when pronunciation is not affected.

Example:

Panic! at the Disco

becomes:

panic at the disco

For branded single-letter + word names:

t-pain

oral-b

Keep the hyphen.

Do not blindly transcribe a homophone if context clearly indicates a known entity.

Research uncertain entities.

--------------------------------------------------
## 27. PEOPLE'S NAMES
--------------------------------------------------

Use standard/common spelling.

If multiple common spellings exist:

any accepted common spelling may be valid.

For celebrities/public figures/fictional characters:

verify the correct spelling.

--------------------------------------------------
## 28. USERNAMES
--------------------------------------------------

Transcribe usernames as heard.

Example:

john dot one two three

at dark underscore wolf

cool underscore guy ninety-nine

Do not automatically transform spoken words into actual @ / _ / digit syntax.

--------------------------------------------------
## 29. URLS AND EMAIL ADDRESSES
--------------------------------------------------

Transcribe them as individual spoken words.

Example:

is it christmas dot com

NOT:

ischristmas.com

Email:

jane smith eight at company dot com

NOT:

janesmith8@company.com

--------------------------------------------------
## 30. YOUTUBE / CHANNEL NAMES
--------------------------------------------------

Use official spelling by default.

Apply catalog/entity rules.

For channel names containing:
- personal names
- multiple proper nouns
- individually spoken letters
- numbers

apply appropriate spoken-form rules.

--------------------------------------------------
## 31. HONORIFICS
--------------------------------------------------

Spell personal titles fully.

Examples:

mister johnson

missus smith

Do not use abbreviations such as:

mr.
mrs.

--------------------------------------------------
## 32. CONTRACTIONS
--------------------------------------------------

Transcribe according to what was actually spoken.

If words are clearly merged:

i'm

you're

don't

gimme

If the speaker clearly articulates the expanded form:

i am

give me

Do NOT force a contraction simply because one exists.

Allowed standard contractions include:

i'm
you're
we're
they're
he's
she's
it's
what's
who's
where's
that's
i've
you've
we've
they've
could've
should've
would've
i'll
you'll
he'll
she'll
we'll
they'll
i'd
you'd
he'd
she'd
we'd
they'd
isn't
aren't
wasn't
weren't
don't
doesn't
didn't
won't
can't
couldn't
shouldn't
wouldn't
hasn't
haven't
hadn't
mustn't
ain't
let's

Informal contractions may be used when actually spoken:

gimme
gonna
gotta
gotcha
lemme
wanna
whatcha
dunno
c'mon
'cause
y'all
kinda
'til
'er
'em
imma

When uncertain, prefer the expanded form unless the spoken contraction is clear.

--------------------------------------------------
## 33. PLURALS / POSSESSIVES OF LETTER SEQUENCES
--------------------------------------------------

Plural:

c. d.s

Possessive:

i. b. m.'s

The ending attaches after the final period.

--------------------------------------------------
## 34. SYMBOLS AND PUNCTUATION
--------------------------------------------------

Allowed structural characters include:

<>
[]
(())
{{}}
-
.
'

Use these only for their defined purposes.

Do NOT introduce:
- commas
- sentence periods
- question marks
- exclamation marks
- arbitrary punctuation

Period is used for individual spoken letters.

--------------------------------------------------
## 35. NON-TARGET LANGUAGES
--------------------------------------------------

If foreign/non-target language appears:

### If understandable:

Transcribe it.

### If understandable after dictionary/search verification:

Transcribe it.

### If partly unintelligible:

Use (()) for the unclear section.

### If entire utterance is unintelligible non-target language:

<nt>

### Background non-target language from unidentified speaker:

[bg]

NOT:

<nt>

For target-language locale where non-target words use accents/diacritics
that are not allowed, normalize according to locale rules.

For English locale, use Latin/plain characters where required.

Example:

José González

may become:

jose gonzalez

when accents are not permitted.

--------------------------------------------------
## 36. SENSITIVE CONTENT
--------------------------------------------------

Use:

<sc>

when the utterance contains sensitive content under the workflow's judgement.

This is a whole-clip condition.

If applicable:

<sc>

alone.

--------------------------------------------------
## 37. USER DESCRIPTORS
--------------------------------------------------

The application should collect speaker metadata separately from transcription.

### Gender

Allowed:

msv
fsv
unknown

Meaning:

msv = masculine-sounding voice
fsv = feminine-sounding voice
unknown = ambiguous/unreliable

DO NOT infer gender from insufficient evidence.

If distortion prevents classification:

unknown

### Nativity

Allowed:

native
nonnative
unknown
tts

Rules:

native:
native English speaker

nonnative:
non-native English speaker

tts:
computer-generated / TTS / agent speech

unknown:
cannot determine

IMPORTANT:

Do not guess from voice distortion.

If the speaker only produces laughter/coughing/sighing/etc. without
intelligible speech:

nativity = unknown

--------------------------------------------------
## 38. TAG INVENTORY
--------------------------------------------------

The final validator must recognize exactly these V2 tags:

### Speaker

<s1>
<s2>
<s3>
...

### Overlap

<ol>
</ol>

### Cross talk

<ct>

### Filled pause

[fp]

### Non-lexical vocal sound

[hn]

### Laughter

[laughter]

### Background

[bg]

### Unintelligible

(())

### Mis-spoken

{{}}

### Artifact

[artifact]

### Whole-file tags

<ga>
<na>
<ns>
<nt>
<sc>

Do NOT introduce retired V1 tags.

--------------------------------------------------
## 39. RETIRED / REMOVED TAGS
--------------------------------------------------

DO NOT use these V1 tags:

[cough]
[sigh]
[throat-clear]
[sneeze]
[lip-smack]

All of these are now:

[hn]

Also DO NOT use:

<hum>

Humming is:

[hn]

V2 also removed:

<i>
<sg>
<ws>
<pws>
<lv>

Do not implement these as valid V2 output tags.

--------------------------------------------------
## 40. TAG PLACEMENT
--------------------------------------------------

Tags must be placed where the corresponding audio feature occurs.

Examples:

play [hn] music

[laughter] you're funny

play [artifact] music

For an interrupted word:

ama- [hn] -zing

When a tag starts the transcription:

[artifact] stop

Do not add unnecessary spaces before/after tags.

--------------------------------------------------
## 41. MULTIPLE TAGS
--------------------------------------------------

When multiple tags are required at the same location:

- use alphabetical order
- separate tags by a single space
- do not add an extra leading/trailing space

The validator must detect:
- incorrect ordering
- duplicated tags
- unsupported tags
- invalid spacing

--------------------------------------------------
## 42. WHOLE-FILE TAG LOGIC
--------------------------------------------------

These conditions are mutually important.

### Blank/no signal

<na>

### Recording damaged/garbled

<ga>

### No speech and no other applicable tag

<ns>

### Entire utterance is unintelligible

(())

### Entire utterance is unintelligible non-target language

<nt>

### Sensitive content

<sc>

### Only laughter

[laughter]

### Only cough/sigh/humming/etc.

[hn]

Whole-file tags must not be combined with ordinary transcription text.

Example:

WRONG:
<na> hello

WRONG:
<ga> hello

CORRECT:
<na>

CORRECT:
<ga>

--------------------------------------------------
## 43. DICTIONARY / RESEARCH
--------------------------------------------------

For English spelling verification:

1. Merriam-Webster
2. American Heritage
3. Cambridge only for British-specific vocabulary

American spelling is preferred.

For catalog entities, verify through appropriate authoritative/search resources
according to the project documentation.

The application should support a manual "Research Required" flag instead of
automatically changing uncertain words.

Never silently change a human transcription because a search result suggests
another spelling.

--------------------------------------------------
## 44. QA VALIDATION ENGINE
--------------------------------------------------

Implement a deterministic validation engine.

The validator should check:

### Formatting

- lowercase violations
- forbidden punctuation
- forbidden symbols
- invalid spaces
- invalid tag syntax

### Numbers

- numeric digits where words are required

### Backchannels

- mhm -> flag, expected mm-hm
- uh huh -> flag, expected uh-huh
- mm mm -> flag, expected mm-mm
- nuh uh -> flag, expected nuh-huh
- hm -> flag, expected hmm
- umm -> flag, expected um
- uhh -> flag, expected uh

### Tags

- unknown tag
- retired V1 tag
- wrong tag
- duplicate immediate non-lexical tag
- wrong whole-file tag combination

### Speaker tags

- speaker numbering order
- missing speaker tags
- <s1> used in single-speaker audio
- malformed <ol> blocks

### Overlap

- unclosed <ol>
- unmatched </ol>
- speaker tags missing inside overlap
- <ct> nested inside <ol>

### Unintelligible

- malformed (())
- invalid spacing inside (())
- misuse for recording problems

### Mis-spoken

- malformed {{}}
- empty {{}}
- nested invalid syntax

### Truncation

- invalid hyphen placement
- hyphens added to complete words
- missing hyphen for clearly truncated speech

### Retired tags

Automatically flag:

[cough]
[sigh]
[throat-clear]
[sneeze]
[lip-smack]
<hum>
<ws>
<pws>
<lv>
<sg>
<i>

--------------------------------------------------
## 45. AUDIO ANALYSIS ENGINE
--------------------------------------------------

Audio analysis is allowed as an ASSISTIVE/QA feature.

Recommended components:

- local audio reader
- waveform extraction
- VAD
- silence detector
- audio event classifier
- optional speaker diarization
- overlap detector
- audio quality analyzer

Possible local stack:

Python
FFmpeg
sounddevice
PySide6
NumPy
SciPy
librosa
Silero VAD
optional pyannote for speaker analysis
local audio-event model where appropriate

Do NOT use an ASR model to automatically generate the final transcription.

--------------------------------------------------
## 46. INTERNAL EVENT MODEL
--------------------------------------------------

Every detected event should use a structured representation.

Example:

{
  "event_id": "evt_001",
  "type": "possible_laughter",
  "start": 7.42,
  "end": 8.15,
  "confidence": 0.91,
  "speaker": "s1",
  "requires_human_confirmation": true
}

Possible event types:

speech
pause
silence
possible_laughter
possible_hn
possible_artifact
possible_background
possible_overlap
speaker_change
garbled_audio
no_audio
no_speech

IMPORTANT:

"possible_" events must never be automatically converted into final tags
without human confirmation.

--------------------------------------------------
## 47. HUMAN CONFIRMATION WORKFLOW
--------------------------------------------------

For every uncertain audio event:

1. Highlight the timeline.
2. Play the relevant audio.
3. Show detected event.
4. Show confidence.
5. Let user confirm/change/reject.
6. Insert corresponding tag only after confirmation.
7. Keep an audit record.

Example:

Detected:
POSSIBLE_LAUGHTER
00:07.42 - 00:08.15
confidence: 0.91

UI:

[Play Segment]
[Confirm laughter]
[Change to hn]
[Reject]

--------------------------------------------------
## 48. TRANSCRIPTION EDITOR
--------------------------------------------------

Create an editor with:

- plain text transcription area
- waveform
- timeline
- speaker markers
- event markers
- tag insertion buttons
- validation panel
- audio playback speed control
- replay selected segment
- undo/redo
- save
- export

Recommended tag buttons:

[fp]
[hn]
[laughter]
[bg]
[artifact]
(())
{{}}
<s1>
<s2>
<ol>
</ol>
<ct>

Whole-file buttons:

<ga>
<na>
<ns>
<nt>
<sc>

--------------------------------------------------
## 49. VALIDATION UX
--------------------------------------------------

Validation should be non-destructive.

Never automatically rewrite user transcription.

Show:

ERROR
WARNING
INFO

Examples:

ERROR:
Forbidden punctuation detected: "?"

ERROR:
Retired V1 tag [cough] detected.
Use [hn] under V2.

WARNING:
Possible speaker tag missing.

WARNING:
Detected possible laughter around 00:12.4 but no
[laughter] tag exists in the transcription.

INFO:
1.8 second pause detected.
Pause is internal metadata and should not be inserted
as a transcription tag.

--------------------------------------------------
## 50. AUDIT LOG
--------------------------------------------------

Maintain a local audit trail.

Example:

{
  "timestamp": "...",
  "action": "confirmed_event",
  "event_type": "laughter",
  "audio_start": 7.42,
  "audio_end": 8.15
}

The audit log should record:
- detected events
- confirmations
- rejections
- transcription edits
- validation results
- metadata changes

--------------------------------------------------
## 51. PROJECT STRUCTURE
--------------------------------------------------

Recommended structure:

ermis-local-assistant/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── audio/
│   │   ├── loader.py
│   │   ├── playback.py
│   │   ├── waveform.py
│   │   ├── preprocessing.py
│   │   └── quality.py
│   │
│   ├── detection/
│   │   ├── vad.py
│   │   ├── silence.py
│   │   ├── laughter.py
│   │   ├── nonlexical.py
│   │   ├── artifact.py
│   │   ├── background.py
│   │   └── overlap.py
│   │
│   ├── speakers/
│   │   ├── diarization.py
│   │   └── metadata.py
│   │
│   ├── ermis/
│   │   ├── tags.py
│   │   ├── rules.py
│   │   ├── validator.py
│   │   ├── normalization.py
│   │   └── whole_file.py
│   │
│   ├── models/
│   │   └── event.py
│   │
│   ├── storage/
│   │   ├── database.py
│   │   └── audit.py
│   │
│   └── ui/
│       ├── main_window.py
│       ├── waveform_view.py
│       ├── transcription_editor.py
│       ├── validation_panel.py
│       └── speaker_panel.py
│
├── tests/
│   ├── test_rules.py
│   ├── test_tags.py
│   ├── test_numbers.py
│   ├── test_backchannels.py
│   ├── test_speakers.py
│   ├── test_overlap.py
│   ├── test_whole_file.py
│   └── test_truncations.py
│
├── data/
│   ├── projects/
│   └── models/
│
├── docs/
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── CLAUDE.md

--------------------------------------------------
## 52. DATABASE
--------------------------------------------------

Use SQLite locally.

Suggested tables:

projects
audio_files
speakers
events
transcriptions
validation_results
audit_logs

Example event:

id
audio_file_id
type
start_time
end_time
confidence
speaker_id
confirmed
created_at

--------------------------------------------------
## 53. PRIVACY
--------------------------------------------------

Default behavior:

- local processing
- no automatic upload
- no external transcription API
- no telemetry
- no cloud storage
- user controls all exported data

If any external service is ever introduced,
make it explicitly opt-in.

--------------------------------------------------
## 54. TESTING STRATEGY
--------------------------------------------------

Create unit tests for every major rule.

Mandatory tests:

1. lowercase validation
2. number normalization validation
3. forbidden punctuation
4. backchannel spelling
5. [hn]
6. [laughter]
7. [artifact]
8. [bg]
9. <ga>
10. <na>
11. <ns>
12. <nt>
13. <sc>
14. (())
15. {{}}
16. speaker tags
17. <ol>
18. <ct>
19. truncation
20. repetitions
21. contractions
22. letter sequences
23. alphanumeric sequences
24. catalog entities
25. whole-file conditions
26. retired V1 tags

--------------------------------------------------
## 55. EDGE CASE TESTS
--------------------------------------------------

Test examples conceptually like:

### Case A
Audio:
speaker laughs then speaks

Expected:
[laughter] ...

### Case B
Audio:
speaker coughs twice

Expected:
[hn]

not:

[hn] [hn]

### Case C
Audio:
brief static then speech

Expected:
[artifact] speech

### Case D
Audio:
recording skips and is impossible to evaluate

Expected:
<ga>

### Case E
Audio:
completely blank

Expected:
<na>

### Case F
Audio:
audio exists but no speech and no other applicable tag

Expected:
<ns>

### Case G
Audio:
speech partly unintelligible

Expected:
(()) plus intelligible content

### Case H
Audio:
entire speech unintelligible

Expected:
(())

### Case I
Two speakers, sequential

Expected:
<s1> ... <s2> ...

### Case J
Two speakers intelligibly overlap

Expected:
<ol> <s1> ... <s2> ... </ol>

### Case K
Two speakers overlap and content is unintelligible

Expected:
<ct>

### Case L
Identifiable speaker + unidentified background cough

Expected:
speaker content + [bg]

### Case M
Speaker stutters whole words

Expected:
word word word

No hyphen.

### Case N
Speaker cuts off a word

Expected:
fragment-

### Case O
Closed-mouth agreement

Expected:
mm-hm

### Case P
Open-mouth agreement

Expected:
uh-huh

--------------------------------------------------
## 56. DO NOT DO THESE THINGS
--------------------------------------------------

DO NOT:

1. Use ASR to generate the official final transcription.
2. Automatically hallucinate missing words.
3. Automatically "correct" grammar.
4. Remove repetitions.
5. Remove fillers.
6. Remove backchannels.
7. Turn laughter into [hn].
8. Turn cough into [cough].
9. Use old V1 tags.
10. Add [pause] to final transcription.
11. Add normal sentence punctuation.
12. Use digits instead of spoken number words.
13. Convert spoken URLs into actual URL syntax.
14. Convert spoken email addresses into email syntax.
15. Guess speaker metadata when evidence is insufficient.
16. Use <ct> merely because speaker order is uncertain.
17. Use <ga> for a brief artifact.
18. Use (()) for recording/microphone corruption.
19. Use <ns> when another applicable tag exists.
20. Combine whole-file tags with ordinary transcription.
21. Automatically overwrite human-entered transcription.
22. Treat model confidence as ground truth.

--------------------------------------------------
## 57. DEVELOPMENT ORDER
--------------------------------------------------

Implement in this order.

PHASE 1
Project setup
- Python
- PySide6
- SQLite
- logging
- configuration

PHASE 2
Audio engine
- load WAV/MP3/etc.
- playback
- waveform
- playback speed
- segment replay

PHASE 3
Audio analysis
- VAD
- silence/pause detection
- audio quality detection
- event timeline

PHASE 4
Ermis rule engine
- tag parser
- syntax validator
- whole-file validator
- punctuation validator
- number validator
- speaker validator
- overlap validator
- backchannel validator
- retired-tag validator

PHASE 5
Transcription editor
- manual typing
- tag insertion
- timeline synchronization
- validation feedback

PHASE 6
Speaker metadata
- speaker list
- gender
- nativity
- unknown handling

PHASE 7
Audio-event assistance
- laughter
- [hn] candidates
- artifacts
- background
- overlap
- speaker changes

PHASE 8
Audit/storage/export

PHASE 9
Testing and packaging

--------------------------------------------------
## 58. ACCEPTANCE CRITERIA
--------------------------------------------------

The MVP is considered successful when:

1. User can load an audio file locally.
2. User can play/pause/seek/replay audio.
3. User can view waveform.
4. User can see detected speech/silence regions.
5. Pause is visible internally but is never inserted as an Ermis tag.
6. User can manually enter transcription.
7. User can insert all valid V2 tags.
8. Application detects invalid V1 tags.
9. Application validates lowercase.
10. Application validates punctuation.
11. Application validates backchannel spelling.
12. Application validates speaker tags.
13. Application validates overlap syntax.
14. Application validates whole-file tags.
15. Application can store speaker metadata.
16. Application can save project state locally.
17. Application can export the final human-reviewed annotation.
18. No ASR-generated transcription is used as the official transcription.
19. No cloud service is required for the core application.
20. Every uncertain automated audio event requires human confirmation.

--------------------------------------------------
## 59. ENGINEERING PRINCIPLE
--------------------------------------------------

The most important principle is:

AUDIO IS THE SOURCE OF TRUTH.

The application must prefer:

"What is actually audible?"

over:

"What would normally make sense?"

Never clean up the speaker's language.

Never invent missing content.

Never remove disfluencies.

Never silently normalize a human decision.

When uncertain:
- expose uncertainty
- replay the audio
- ask for human confirmation
- preserve the audit trail

The system is an ASSISTANT and QA TOOL,
not an autonomous transcription submitter.

--------------------------------------------------
## 60. DEVIATION LOG
--------------------------------------------------

### 60.1 ASR drafting (2026-08-28)

Requested by the project owner, in these words: the app should produce
"all words and sabhi kuch" from the audio after recording, instead of asking the
annotator to type them.

This contradicts:

- section 1 ("DO NOT implement an automatic ASR-to-final-transcription
  pipeline", "Final transcription must remain human-entered/reviewed")
- section 47 (a tag may only be inserted after human confirmation)
- section 56.1 ("Use ASR to generate the official final transcription")
- section 58.18 and 58.20 (acceptance criteria)

The owner was told this conflicts with the official Ermis V2 workflow and
confirmed the request. The compliance decision is theirs. What was built:

    audio -> local recogniser -> spoken form -> speaker turns
          -> event tags -> validator -> editor

### 60.2 What still holds

1. LOCAL ONLY. Recognition uses a model on this machine. No audio is uploaded.
   `privacy.allow_external_asr` still refuses to be enabled, and
   `asr.local_files_only = true` forbids even fetching model weights.

2. NEVER PRESENTED AS HUMAN WORK. Draft text is stored with
   `transcriptions.source = 'asr_draft'`, a value the database CHECK constraint
   admits and no other machine label can use. The export states
   `"source": "asr_draft"`, `"asr_used": true`, `"human_reviewed": false`, and
   refuses to write an unreviewed draft unless the caller passes
   `allow_unreviewed=True`. The window title reads
   `[UNREVIEWED MACHINE DRAFT]`. A save becomes `human_edited` only after a
   real keystroke in the editor.

3. AUTO-CONFIRMATION IS LABELLED. A tag a draft inserts is audited as
   `auto_confirmed_event`, never as `confirmed_event`, and the event carries the
   note "auto-confirmed by ASR draft; verify against the audio". Detections
   below `asr.min_event_confidence` are not auto-confirmed at all.

4. UNCERTAINTY IS NOT HIDDEN. A word recognised below
   `asr.min_word_confidence` is written as `((word))` rather than asserted.
   Nothing could be recognised over audible speech becomes `(())`. A blank or
   garbled file is never sent to the recogniser; it drafts `<na>` / `<ga>`
   alone.

5. NO RULE WAS RELAXED. The validator, the tag inventory, whole-file logic and
   the retired-V1-tag checks are untouched, and every draft is validated before
   it reaches the editor.

6. FORM ONLY, NEVER CONTENT. Normalisation lowercases, removes sentence
   punctuation and spells numbers out. Grammar, repetitions, fillers,
   backchannels and mistaken words are left exactly as recognised
   (sections 4.6, 5, 16, 56.3).

7. NATIVITY IS STILL NEVER GUESSED. Gender is suggested from pitch and stays
   `unknown` in the overlap band or with too little voiced audio; `nativity`
   remains `unknown` for a human to set (sections 18A.3, 18A.4, 18A.11).

### 60.4 Nativity suggestion (2026-08-28)

Also requested by the project owner: report each speaker's nativity from the
audio, alongside the speaker count and gender.

This contradicts section 18A.4 and section 37, which say nativity must NOT be
guessed and explicitly rule out accent, audio quality, volume and background
noise as evidence.

What was built (`app/speakers/nativity.py`):

1. `tts` -- synthetic speech is genuinely detectable: pitch that barely moves,
   near-zero cycle-to-cycle jitter, and digital silence between phrases instead
   of a room. Two or more of those together is reported with moderate
   confidence.

2. `native` / `nonnative` -- from fluency features: speaking rate, long-pause
   ratio, filler density, and how confidently the English recogniser scored the
   words.

3. `unknown` -- the default, and the answer whenever the evidence is thin:
   fewer than `speakers.min_words_for_nativity` words, less than
   `speakers.min_speaking_seconds` of that speaker's speech, or a fluency score
   inside `speakers.nativity_decision_margin`.

The honest limits, stated because section 18A.4 is right about them:

- Recogniser confidence is partly a function of accent, so feature 2 is a proxy
  for the very evidence the specification excludes. It carries the smallest
  weight of the four and is never sufficient alone.
- A slow, careful native speaker and a fluent non-native speaker both defeat
  these features.
- Confidence is capped at 0.75. The field is never presented as certain, is
  written only above `speakers.min_nativity_confidence`, is never written over a
  human edit, and stays editable in the speaker panel (section 18A.11).

Gender continues to work as section 18A.3 requires: pitch-based, `unknown`
inside the masculine/feminine overlap band or with too little voiced audio.

Switch it off with:

    speakers.suggest_nativity = false

### 60.5 Operational limit worth knowing

Drafting needs a speech model resident in memory: about 2.6 GB peak for the
default `whisper-small.en`, 1.0 GB for `base.en`. On a machine with 8 GB, the
window and a command-line run cannot both hold one. The application checks free
memory before loading and either falls back to a smaller model (saying so) or
refuses -- because a load that does not fit segfaults rather than raising, and
that cannot be recovered from.

Recording, analysis, validation, export and the audit trail need no model at
all: with `asr.enabled = false` the tool runs in a few hundred megabytes.

### 60.6 How to return to the specification as written

Set these and the application behaves as sections 1, 18A.4 and 56.1 require:

    asr.enabled = false               # no recogniser at all
    asr.auto_draft_on_load = false    # drafting only when explicitly asked
    speakers.suggest_nativity = false # nativity stays unknown for a human

With `asr.enabled = false` the tool is again analysis, validation and QA over
text a human typed.
