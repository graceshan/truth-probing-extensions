# T2B fixed inventor-sample review, v1

**Review complete; factual resolution incomplete. T2A remains provisional.**

Based on commit `82fa91e710f89d7d1b8228674aa5998b0e4eb121`. Reviewer: **Codex-assisted external review, not human verification**. All 30 rows retain the frozen draw order, original identity, label and statement hash. The queue and every T2A artifact are unchanged. `review_policy.md` was written before source adjudication and states the residence, geography, slash-list and joint-subject rules. No score or prediction informed these judgments.

## Outcomes

**4 supported false, 2 supported true (confirmed original-label errors), 24 unresolved.** The sample denominator remains 30, not 6. Two confirmed errors among the 30 inspected rows are a detected count, not an estimate treating the 24 unresolved rows as correct. No zero-error claim is justified. Even a hypothetical fully resolved 0/30 result would not certify the remaining negatives: the conservative binomial one-sided 95% upper bound would be about 9.5% (`1 - 0.05**(1/30)`). That hypothetical bound is not applied to this sample.

| Partition | Sampled | Supported false | Supported true | Unresolved |
|---|---:|---:|---:|---:|
| train | 20 | 3 | 2 | 15 |
| validation | 5 | 0 | 0 | 5 |
| test | 5 | 1 | 0 | 4 |

The two confirmed errors are:

- **Draw 4, `inventors:36`: Fridtjof Nansen / the U.K.** Nobel documents the ministerial posting to Britain through May 1908; the accompanying chronology identifies London in 1906–08. A residential diplomatic assignment satisfies the pre-recorded rule.
- **Draw 5, `inventors:153`: Alexander Fleming / France.** V. D. Allison’s published memoir, visually inspected on printed page 91, describes the 1914–18 Army laboratory at Boulogne. The Nobel biography independently records wartime service and return to London in 1918. The extended residential work posting satisfies the same rule; a mere tour or visit would not.

The common failure pattern is an incomplete positive residence set: a person’s main country or usual academic affiliation omits a foreign diplomatic or wartime research posting. Neither observation licenses automatic inference from all travel, citizenship, birthplace, patents or honorary appointments.

## All 30 outcomes, unchanged draw order

Draw indices below are the original zero-based indices. Every unresolved case remains in the denominator and has a completed review with explicit limitations.

| Draw | Source row | Partition | Exact affirmative | Judgment |
|---:|---|---|---|---|
| 0 | `inventors:404` | train | Thomas Edison lived in the U.K. | supported_false |
| 1 | `inventors:246` | train | Felix Hoffmann lived in the U.S. | unresolved |
| 2 | `inventors:76` | test | Alfred Nobel lived in the U.S. | unresolved |
| 3 | `inventors:270` | train | Rudolf Kálmán lived in the U.K. | unresolved |
| 4 | `inventors:36` | train | Fridtjof Nansen lived in the U.K. | supported_true |
| 5 | `inventors:153` | train | Alexander Fleming lived in France. | supported_true |
| 6 | `inventors:195` | train | Orville and Wilbur Wright lived in France. | unresolved |
| 7 | `inventors:218` | train | Igor Tamm lived in the U.S. | unresolved |
| 8 | `inventors:326` | test | Herbert Akroyd Stuart lived in Russia. | unresolved |
| 9 | `inventors:111` | train | Philo Farnsworth lived in Germany. | unresolved |
| 10 | `inventors:209` | train | John Walker lived in the U.S. | unresolved |
| 11 | `inventors:13` | train | Edwin Link lived in the U.K. | unresolved |
| 12 | `inventors:66` | validation | Samuel Morse lived in Germany. | unresolved |
| 13 | `inventors:212` | train | Charles Townes lived in Russia. | unresolved |
| 14 | `inventors:393` | validation | Semyon Kirlian lived in the U.K. | unresolved |
| 15 | `inventors:141` | train | Édouard-Léon Scott de Martinville lived in the U.K. | unresolved |
| 16 | `inventors:350` | train | Konrad Zuse lived in Poland/France. | unresolved |
| 17 | `inventors:143` | test | Richard Leach Maddox lived in Austria. | unresolved |
| 18 | `inventors:299` | train | Ami Argand lived in the U.S. | unresolved |
| 19 | `inventors:200` | train | Carl Elsener lived in the U.S. | unresolved |
| 20 | `inventors:327` | validation | Daniel Fahrenheit lived in the U.S. | unresolved |
| 21 | `inventors:50` | train | James Clerk Maxwell lived in the U.S. | supported_false |
| 22 | `inventors:24` | train | August Horch lived in Russia. | unresolved |
| 23 | `inventors:263` | test | Charles Strite lived in Italy. | unresolved |
| 24 | `inventors:172` | train | Georges Claude lived in Italy. | unresolved |
| 25 | `inventors:320` | train | Christian Huygens lived in Scotland. | supported_false |
| 26 | `inventors:119` | validation | László Bíró lived in the U.S. | unresolved |
| 27 | `inventors:146` | validation | R. Buckminster Fuller lived in Scotland. | unresolved |
| 28 | `inventors:354` | train | John Wesley Hyatt lived in the U.K. | unresolved |
| 29 | `inventors:59` | test | Ignazio Porro lived in the U.S. | supported_false |

## Evidence and reasoning by row

URLs, titles, retrieval dates, response hashes, inspected excerpts/locators and limitations are in `evidence.json`. The paragraphs below are review reasoning, not verbatim biography text. Short excerpts were checked against retrieved page text; the Fleming memoir was read as scanned images. Full downloaded pages are not redistributed in this package. Hashes identify the retrieved responses, while locators and short excerpts make the reasoning reviewable without relying on a search snippet. A response hash alone is not evidence of truth.

### Draw 0: Thomas Edison lived in the U.K.

**supported_false** — The NPS life history positively traces his successive childhood, working and family homes through his death, including the itinerant period and later Florida stays. This sustained US residential chronology supports false for UK residence; it is not an inference merely from his nationality.

Limitations: Undocumented brief travel is possible; this is defeasible historical support, not a universal mathematical proof.

Inspected sources:
- [Edison Biography - Thomas Edison National Historical Park (U.S. National Park Service)](https://www.nps.gov/edis/learn/historyculture/edison-biography.htm) — `edison`; retrieved 2026-10-01.

### Draw 1: Felix Hoffmann lived in the U.S.

**unresolved** — German training and Bayer employment are documented, but neither inspected account supplies a sufficiently complete residence history through retirement and death to exclude a US interval.

Limitations: German nationality, company location and Swiss death do not prove lifelong absence from the US.

Inspected sources:
- [Felix Hoffmann | Science History Institute](https://www.sciencehistory.org/education/scientific-biographies/felix-hoffmann/) — `hoffmann_shi`; retrieved 2026-10-01.
- [Felix Hoffmann - Wikipedia](https://en.wikipedia.org/wiki/Felix_Hoffmann) — `hoffmann_wiki`; retrieved 2026-10-01.

### Draw 2: Alfred Nobel lived in the U.S.

**unresolved** — The institutional source explicitly says he visited the US and John Ericsson during two years of foreign training. It does not distinguish a short visit from a residential study/work stay.

Limitations: Do not promote “visited” to lived, or treat it as proof of never having lived there; duration/accommodation evidence is needed.

Inspected sources:
- [Alfred Nobel - his life and work - NobelPrize.org](https://www.nobelprize.org/alfred-nobel/alfred-nobel-his-life-and-work/) — `nobel_full`; retrieved 2026-10-01.

### Draw 3: Rudolf Kálmán lived in the U.K.

**unresolved** — US and Zurich career appointments are well documented, but childhood and post-retirement/temporary appointments are insufficiently covered to rule out UK residence.

Limitations: The professional biography supports career locations, not exhaustive negative residence evidence.

Inspected sources:
- [Rudolf Kalman (1930 - 2016) - Biography - MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/Biographies/Kalman/) — `kalman`; retrieved 2026-10-01.

### Draw 4: Fridtjof Nansen lived in the U.K.

**supported_true** — A multi-year posting as Norway’s minister to Britain, specifically in London in 1906–08, is a residential diplomatic assignment under the recorded policy, not a holiday or nationality inference. The original false label conflicts with it.

Limitations: Nobel supplies the posting; the secondary chronology corroborates London dates. A household address was not needed for the residential-posting criterion.

Inspected sources:
- [Fridtjof Nansen – Biographical - NobelPrize.org](https://www.nobelprize.org/prizes/peace/1922/nansen/biographical/) — `nansen`; retrieved 2026-10-01.
- [Fridtjof Nansen - Wikipedia](https://en.wikipedia.org/wiki/Fridtjof_Nansen) — `nansen_wiki`; retrieved 2026-10-01.

### Draw 5: Alexander Fleming lived in France.

**supported_true** — Allison’s published recollection explicitly places Fleming’s Army laboratory in Boulogne during 1914–18, breaking his London work; Nobel independently records wartime service and return in 1918. This extended overseas work posting counts as residence under the recorded rule and contradicts the false label.

Limitations: This is an adjudicated residential posting, not an assertion that mere battlefield travel constitutes residence; no domicile-register claim is made.

Inspected sources:
- [Sir Alexander Fleming – Biographical - NobelPrize.org](https://www.nobelprize.org/prizes/medicine/1945/fleming/biographical/) — `fleming`; retrieved 2026-10-01.
- [V. D. Allison, Personal Recollections of Sir Almroth Wright and Sir Alexander Fleming, Ulster Medical Journal 43(2), 1974, pp. 89–98](https://pmc.ncbi.nlm.nih.gov/articles/PMC2385475/) — `fleming_memoir_p91`; retrieved 2026-10-01.

### Draw 6: Orville and Wilbur Wright lived in France.

**unresolved** — The source distinguishes Wilbur’s 1908 stay and both brothers’ 1909 demonstration/training period at Pau. It does not unambiguously establish ordinary residence for both joint subjects rather than temporary demonstration accommodation.

Limitations: Both-person and temporary-stay semantics are unresolved; do not label true from Wilbur alone or false from American nationality.

Inspected sources:
- [Wright brothers - Wikipedia](https://en.wikipedia.org/wiki/Wright_brothers) — `wright_wiki`; retrieved 2026-10-01.

### Draw 7: Igor Tamm lived in the U.S.

**unresolved** — The Nobel entry accounts for Soviet education/career but is a Nobel-era summary, with no full residence chronology before or after it.

Limitations: Soviet posts and membership of an American academy establish neither US residence nor its negation.

Inspected sources:
- [Igor Y. Tamm – Biographical - NobelPrize.org](https://www.nobelprize.org/prizes/physics/1958/tamm/biographical/) — `tamm`; retrieved 2026-10-01.

### Draw 8: Herbert Akroyd Stuart lived in Russia.

**unresolved** — The ANU entry establishes English education/work and later Australian migration/home, but its engineering-focused narrative does not resolve all intermediate stays.

Limitations: No Russian residence was established; no complete residence history or explicit nonresidence evidence was obtained. This is not a supported-false judgment.

Inspected sources:
- [Biography - Herbert Akroyd Stuart - Australian Dictionary of Biography](https://adb.anu.edu.au/biography/stuart-herbert-akroyd-8705) — `akroyd_adb`; retrieved 2026-10-01.

### Draw 9: Philo Farnsworth lived in Germany.

**unresolved** — The institutional profile is short; the secondary biography explicitly records a 1934 European trip and German business agreement. The length and residential character of the German stay are unspecified.

Limitations: German patents/contracts and foreign television adoption are not personal residence evidence.

Inspected sources:
- [NIHF Inductee Philo Farnsworth Invented the Television System](https://www.invent.org/inductees/philo-taylor-farnsworth) — `farnsworth`; retrieved 2026-10-01.
- [Philo Farnsworth - Wikipedia](https://en.wikipedia.org/wiki/Philo_Farnsworth) — `farnsworth_wiki`; retrieved 2026-10-01.

### Draw 10: John Walker lived in the U.S.

**unresolved** — The local heritage history gives English homes and work, but explicitly acknowledges sparse detail in the early professional period. That gap prevents the strict negative-residence standard being met.

Limitations: A predominantly English chronology is not a complete proof against an earlier US stay.

Inspected sources:
- [John Walker - Inventor of the Friction Match | Stockton Heritage](https://heritage.stockton.gov.uk/articles/people/john-walker-inventor-of-the-friction-match/) — `walker`; retrieved 2026-10-01.

### Draw 11: Edwin Link lived in the U.K.

**unresolved** — US business and Mediterranean research are documented, but overseas project schedules/living arrangements are not fully described.

Limitations: Company addresses and invention locations cannot rule out a UK residence.

Inspected sources:
- [NIHF Inductee Edwin Link Invented the Airplane Flight Simulator](https://www.invent.org/inductees/edwin-link) — `link`; retrieved 2026-10-01.
- [Edwin Albert Link - Wikipedia](https://en.wikipedia.org/wiki/Edwin_Link) — `edwin_wiki`; retrieved 2026-10-01.

### Draw 12: Samuel Morse lived in Germany.

**unresolved** — Inspected biography covers English study and European art/patent travel. It does not settle all German stays, and the institutional chronology endpoints were blocked.

Limitations: Search failures and omission of Germany from selected travels are not false evidence.

Inspected sources:
- [Samuel Morse - Wikipedia](https://en.wikipedia.org/wiki/Samuel_Morse) — `morse_wiki`; retrieved 2026-10-01.

### Draw 13: Charles Townes lived in Russia.

**unresolved** — The detailed autobiographical account includes US bases and French/Japanese fellowships, but it is not a full residential/travel history through 2015.

Limitations: Russian residence was not established or exhaustively excluded; scientific collaboration or travel alone is insufficient.

Inspected sources:
- [Charles H. Townes – Biographical - NobelPrize.org](https://www.nobelprize.org/prizes/physics/1964/townes/biographical/) — `townes`; retrieved 2026-10-01.

### Draw 14: Semyon Kirlian lived in the U.K.

**unresolved** — Only a citation-poor biographical lead describing Krasnodar work and housing was inspectable. It cannot meet the evidence requirement for a UK negative.

Limitations: The lead itself flags missing citations; a credible full life/residence account remains necessary.

Inspected sources:
- [Semyon Kirlian - Wikipedia](https://en.wikipedia.org/wiki/Semyon_Kirlian) — `kirlian`; retrieved 2026-10-01.

### Draw 15: Édouard-Léon Scott de Martinville lived in the U.K.

**unresolved** — The archive-based source establishes Paris recording work and artifact provenance, not his whole life’s residences.

Limitations: Absence of a UK reference in an invention history is insufficient.

Inspected sources:
- [Publications :: FirstSounds.ORG](https://www.firstsounds.org/research/scott.php) — `scott`; retrieved 2026-10-01.

### Draw 16: Konrad Zuse lived in Poland/France.

**unresolved** — The biography places childhood in Braunsberg, East Prussia; it also describes the computer’s later French location, not Zuse’s residence. Poland/France has undefined slash truth semantics and historical-geography ambiguity.

Limitations: Do not map historic East Prussia to current Poland silently, infer person residence from a machine, or choose an AND/OR meaning for slash.

Inspected sources:
- [Konrad Zuse (1910 - 1995) - Biography - MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/Biographies/Zuse/) — `zuse_math`; retrieved 2026-10-01.

### Draw 17: Richard Leach Maddox lived in Austria.

**unresolved** — Secondary history reports several overseas and English residences, but has citation gaps and does not account for all early/middle-life intervals.

Limitations: Austria cannot be ruled out merely because the listed residences omit it.

Inspected sources:
- [Richard Leach Maddox - Wikipedia](https://en.wikipedia.org/wiki/Richard_Leach_Maddox) — `maddox`; retrieved 2026-10-01.

### Draw 18: Ami Argand lived in the U.S.

**unresolved** — The institutional dictionary documents European study and businesses, but is too concise to establish absence of all US residential intervals.

Limitations: European company locations and birthplace do not suffice; fuller correspondence/residence history is needed.

Inspected sources:
- [Argand, Ami](https://hls-dhs-dss.ch/fr/articles/025832/) — `argand_dhs2`; retrieved 2026-10-01.

### Draw 19: Carl Elsener lived in the U.S.

**unresolved** — Swiss apprenticeship and business are described, but the European journeyman period is nonspecific and the complete residence history is missing.

Limitations: Do not infer no US residence from Swiss nationality, knife production or a family company history.

Inspected sources:
- [Karl Elsener (inventor) - Wikipedia](https://en.wikipedia.org/wiki/Karl_Elsener_(inventor)) — `elsener_inventor`; retrieved 2026-10-01.

### Draw 20: Daniel Fahrenheit lived in the U.S.

**unresolved** — A secondary chronology records European movements but was not independently verified against its underlying biographies. The phrase the U.S. also postdates his life, and original semantics do not decide whether present-day territory or contemporary polity is meant.

Limitations: No convenient anachronism-based false judgment; both evidence and temporal-geography interpretation need resolution.

Inspected sources:
- [Daniel Gabriel Fahrenheit - Wikipedia](https://en.wikipedia.org/wiki/Daniel_Gabriel_Fahrenheit) — `fahrenheit_wiki`; retrieved 2026-10-01.

### Draw 21: James Clerk Maxwell lived in the U.S.

**supported_false** — The detailed university biography accounts for childhood, Edinburgh/Cambridge education, Aberdeen/London posts, Glenlair and final Cambridge years up to death. The continuous sequence of documented residences and work bases supports false for a US residence, rather than using Scottish nationality as evidence.

Limitations: Short undocumented journeys cannot be ruled out absolutely; the supported-false claim concerns residence and is defeasible.

Inspected sources:
- [James Clerk Maxwell (1831 - 1879) - Biography - MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/Biographies/Maxwell/) — `maxwell`; retrieved 2026-10-01.

### Draw 22: August Horch lived in Russia.

**unresolved** — The inspected career account documents German locations but warns of single-source coverage and does not account for all earlier or later residences.

Limitations: The unavailable Audi lead added no evidence; German enterprises do not exclude Russian residence.

Inspected sources:
- [August Horch - Wikipedia](https://en.wikipedia.org/wiki/August_Horch) — `horch_wiki`; retrieved 2026-10-01.

### Draw 23: Charles Strite lived in Italy.

**unresolved** — The patent explicitly establishes Minneapolis residence at one date. It says nothing about whether he lived in Italy at another time.

Limitations: Other fetched biography leads were inaccessible, irrelevant or internally unreliable; a single dated residence cannot establish the negative.

Inspected sources:
- [US1394450A - Bread-toaster](https://patents.google.com/patent/US1394450A/en) — `strite_patent`; retrieved 2026-10-01.

### Draw 24: Georges Claude lived in Italy.

**unresolved** — NIHF establishes selected French career phases and postwar imprisonment, but does not cover his full foreign work and private residence history.

Limitations: French nationality, Paris inventions and French imprisonment are insufficient for the Italian negative.

Inspected sources:
- [NIHF Inductee Georges Claude Invented Neon Tubes](https://www.invent.org/inductees/georges-claude) — `claude`; retrieved 2026-10-01.

### Draw 25: Christian Huygens lived in Scotland.

**supported_false** — The university biography traces Dutch education/home, explicit Paris residence, repeated Hague returns, final Dutch years and identified English visits. This detailed sequence supports no Scottish residence; England is not silently treated as Scotland.

Limitations: Original Christian spelling is identified with this inventor’s conventional Christiaan name. The conclusion remains defeasible, not universal absence proof.

Inspected sources:
- [Christiaan Huygens (1629 - 1695) - Biography - MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/Biographies/Huygens/) — `huygens`; retrieved 2026-10-01.

### Draw 26: László Bíró lived in the U.S.

**unresolved** — The inspected institutional entry concerns the ballpoint invention, Budapest background and British patent use, and omits a full migration/residence timeline.

Limitations: No US-negative inference from a short inventor profile; follow up with a reliable full migration biography.

Inspected sources:
- [NIHF Inductee Laszlo Biro, Who Invented the Ballpoint Pen](https://www.invent.org/inductees/laszlo-josef-biro) — `biro`; retrieved 2026-10-01.

### Draw 27: R. Buckminster Fuller lived in Scotland.

**unresolved** — BFI identifies several US residences/appointments and extensive world travel. It does not establish the character or duration of all Scottish stays.

Limitations: Lectures, global travel and honorary appointments cannot be equated with or exclude residence.

Inspected sources:
- [Biography – Buckminster Fuller Institute](https://www.bfi.org/about-fuller/biography/) — `fuller`; retrieved 2026-10-01.

### Draw 28: John Wesley Hyatt lived in the U.K.

**unresolved** — NIHF documents birthplace and inventions, but not successive homes across his life.

Limitations: A New York birthplace and US inventions cannot establish that he never lived in the UK.

Inspected sources:
- [National Inventors Hall of Fame Inductee John Hyatt Invented Celluloid](https://www.invent.org/inductees/john-wesley-hyatt) — `hyatt`; retrieved 2026-10-01.

### Draw 29: Ignazio Porro lived in the U.S.

**supported_false** — The detailed archival biography follows Sardinian military education/service, France relocation, Florence settlement, Milan move and final years; the university history corroborates the European sequence. This positive life chronology supports the US statement being false.

Limitations: Foreign patent ownership is not residence; the evidence remains defeasible and the Italian account was read with Codex-assisted translation.

Inspected sources:
- [PORRO, Ignazio - Enciclopedia - Treccani](https://www.treccani.it/enciclopedia/ignazio-porro_(Dizionario-Biografico)/) — `porro_treccani`; retrieved 2026-10-01.
- [Molecular Expressions: Science, Optics and You - Timeline - Ignazio Porro](https://micro.magnet.fsu.edu/optics/timeline/people/porro.html) — `porro_fsu`; retrieved 2026-10-01.

## Separate Franklin and alias findings

**Franklin/France: supported true, outside the sample.** The Library of Congress [Portrait of Franklin, item #obj1](https://www.loc.gov/exhibits/franklin/franklin-epitaph.html#obj1) explicitly says he lived in the Paris suburb spelled “Passay” on that page from 1777–1785. The recovered registry record, including its exact `rejected_true_or_ambiguous` status, is embedded unchanged in `supplementary_reviews.json` and hash-bound to the historical registry. This new adjudication does not rewrite it, invent a source-row pair, or enter the 30-row error count.

**Simjian corroboration:** [Florida Inventors Hall of Fame](https://floridainvents.org/luther-george-simjian/) independently documents Luther George / Luther G. Simjian and the matching Yale/ATM career. It corroborates the full-name person; it does not independently spell out the short alias “Luther Simjian.” That short/full linkage remains supported by the separately preserved T2A biography. No identity mapping or quarantine is changed.

**Farnsworth corroboration:** [National Inventors Hall of Fame](https://www.invent.org/inductees/philo-taylor-farnsworth) uses Philo Farnsworth in the page title and Philo Taylor Farnsworth in its heading/body for the same television inventor. It independently corroborates the existing alias linkage. No historical evidence record is overwritten.

## Recommendations, not applied

`recommended_exclusions.csv` recommends **26 affirmative/negation pairs (52 original source rows)**: 2 pairs for confirmed errors and 24 for unresolved facts. Each row preserves source/statement hashes, original label, partition, person key and paired ID. Recommended exclusion is precautionary, not a claim that every unresolved label is wrong. T2A’s admitted/excluded manifests are unchanged.

`proposed_rule_changes.json` bounds follow-up to inventor source facts across existing outer train/D/E:

1. Independently adjudicate documented overseas diplomatic, military/research, employment and study postings. Exclude confirmed conflicting affirmative/negation pairs in a subsequent reviewed overlay; do not relabel or move records.
2. Build a separate targeted queue for ambiguous slash-country claims, historical/current geography and joint-person subjects. Freeze any additional semantic rule before reviewing that population. Do not infer residence from artifact location (Zuse’s computer), a business agreement (Farnsworth), or one member of a joint subject (the Wrights).
3. Preserve Franklin/France as an ineligible historical negative candidate with new positive evidence attached separately.

Other unresolved cases need fuller residential histories or archive corroboration, not a new random sample. In particular, low-quality/inaccessible sources remain evidence gaps. The Strite Wikipedia lead contained inconsistent biographical details, so it was not used; the original patent establishes only one Minneapolis residence date. The Kirlian lead has missing-citation warnings and is explicitly insufficient. A fetched ANU URL initially redirected to a different person; it was discarded and the correct Akroyd Stuart entry was inspected. These limitations are preserved in the retrieval log and evidence notes.

## Files and validation

- `review_policy.md`: rules recorded before adjudication.
- `reviews.json`: all 30 identities, outcomes, evidence references, reasoning and limitations.
- `evidence.json`: inspected source records and concise quotations/paraphrases.
- `retrieval_log.json`: attempted page retrievals, including blocked and unused leads; HTTP 200 does not itself establish relevance.
- `supplementary_reviews.json`: Franklin and alias corroboration, denominator contribution zero.
- `recommended_exclusions.csv`: paired recommendations only.
- `proposed_rule_changes.json`: bounded score-blind proposals only.
- `package_manifest.json`: input/package hashes, scope and outcome counts.
- `validate.py`, `test_validation.py`: identity, evidence, coverage, pairing and integrity checks.

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B data/clean_protocol/selection_repair_v1/source_review_v1/validate.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m unittest discover -s data/clean_protocol/selection_repair_v1/source_review_v1 -p test_validation.py -v
git diff --check
```

Validation run: **16 tests passed**, including mutations of every original identity field, plus full-package/input hash validation. Diff checks passed. These are record-integrity tests, not independent human factual verification.

Validation must preserve exact queue coverage/order; check all immutable identity fields, allowed statuses, authoritative evidence for resolved judgments, excerpts/locators, separate supplementary denominator, paired recommendations, historical registry bytes and every package/input hash. The tests validate the review records and guards; they do not independently certify the historical judgments. T2A stays provisional pending review of these adjudications. No A/P reservation, fitting, activation access, compound generation or final-test prediction inspection occurred, and no other worktree was used.
