# MDPI (ISPRS IJGI) Submission — Preparation Checklist

Tracking file for converting the paper to an MDPI-formatted submission.
Target journal: **ISPRS International Journal of Geo-Information** (`ijgi`) — change the
single class option in `main.tex` to retarget (e.g. `urbansci`); bibliography style
(`mdpi.bst`) is unchanged either way.

Status legend: ✅ done · ⬜ to do · 🔶 decision needed

**Resolved 2026-06-26:** fixed the two bad DOIs (`Sevtsuk2016` → `10.51347/jum.v20i2.4056`;
`vanderbie_sfnetworks_2021` → corrected authors + CRAN, dropped the non-existent JOSS DOI);
added **Beauchamp (1965)** and **Dekker (2005)** for harmonic closeness; added
`\authorcontributions` + `\institutionalreview`/`\informedconsent` ("Not applicable") in the
correct MDPI back-matter order. Remaining MINOR bib cleanups (§B) and de-anonymisation
items (§E) are still open.

---

## A. Build & layout — ✅ done

- ✅ `main.tex` on `Definitions/mdpi.cls`, anonymised, shared content pulled from
  parent `paper/` via `TEXINPUTS` (mirrors `epb_submission/`).
- ✅ `build.py` compiles + assembles `mdpi_submission_<date>.zip` (bundles `Definitions/`).
- ✅ Compiles clean: **23 pp, 0 undefined references/citations, 61 bib entries**.
- ✅ Shared abstract read via `catchfile` (MDPI `\abstract` is a command and rejects a
  live `\input`).
- ✅ Title set to "Divergence in Closeness Centrality Formulations…" (matches body term).
- ✅ Table/figure overflow fixed (was the SAGE two-column layout assumption):
  - Definitions table (`3_centralities`) → `\linewidth`-relative column widths.
  - 9-column data tables (`S2`) → `\fitwidth` shrink-to-fit wrapper (graphicx-only, dual-class safe).
  - Tall portrait images (`6_analysis`, `S2`) → added `width=\linewidth` cap.
  - Remaining overfulls are <7 pt (one body-text line-break, one sub-mm table residual) — negligible.
- ✅ EPB build re-verified after shared-file edits (18 pp, still compiles).

---

## B. Reference audit — fixes needed in `references.bib`

Full web verification of all **61 cited references** (existence, metadata, DOI/URL,
contextual use). **Every cited work exists and is correctly used in context; 0 undefined
citations.** Issues found are metadata-only:

### 🔶 PROBLEM (2) — fix before submission

- **`Sevtsuk2016`** — **wrong DOI**. `10.1080/10464883.2012.714912` resolves to an
  unrelated book review (Fausch, *J. Architectural Education* 2012). Correct DOI:
  **`10.51347/jum.v20i2.4056`** (Urban Morphology 20(2):89–106, 2016).
- **`vanderbie_sfnetworks_2021`** — **invalid DOI + wrong metadata**. DOI
  `10.21105/joss.03286` does not resolve (no JOSS paper exists for sfnetworks). Authors
  are wrong ("van der Bie…" → should be **van der Meer, L.; Abad, L.; Gilardi, A.;
  Lovelace, R.**); type should be **software/manual**, not `@article`/JOSS. Replace with
  the package's own citation (CRAN/r-universe).

### ⬜ MINOR (metadata/URL cleanups)

- `Iacono2008` — author field contains the institute "Humphrey, Hubert H" (the Humphrey
  Institute), not a person. Real authors: **Iacono, M.; Krizek, K.; El-Geneidy, A.**
- `krenz_kimon_developments_2022` — venue `Urbanism` → **`Urban Design`** (城市设计).
- `cooper_sdna_2020` — title has extra word: "spatial **design** network" → published
  title is "sDNA: 3-d **spatial network** analysis…".
- `stahle_place_2023` — year vs URL mismatch (URL → release **v3.3.1 = 2024**, bib says
  2023). Also "Berghauser, Pont" mis-split → **Berghauser Pont, Meta**.
- `Turner2005a` — type/venue: listed as UCL Bartlett techreport; actually **GeoComputation
  2005 proceedings**.
- `Rochat2009` — type/venue: techreport vs **ASNA 2009 conference paper** (EPFL Infoscience).
- `Rutherford1979` — author missing initial: **G. Scott Rutherford**.
- `Bates2007` — DOI URL has trailing `//` → `https://doi.org/10.1108/9780857245670-002`.
- `Marchiori2000` — `url` is a generic Elsevier landing string; replace with DOI
  `10.1016/S0378-4371(00)00311-3`.
- Generic/stale URLs to clean or replace with the article DOI: `Hansen1959`,
  `Duncan2011`, `Baradaran2001`, `Scheurer2007`, `crtm_zones_2020` (use direct dataset
  link `datos.crtm.es/datasets/crtm::zonificacionzt1259/about`).
- `Alexander1967` — 1967 Ekistics reprint of the 1965 original (acceptable; note only).

### Housekeeping

- ⬜ 26 unused entries in `references.bib` (cited only in the excluded sections 2 & 5) —
  harmless (don't appear in the bibliography), optional to prune.
- ⬜ Cosmetic: `mdpi.bst` doesn't define BibTeX month abbreviations (`jan`/`jul`/…), so a
  few entries omit the month. Either ignore or spell months out in `references.bib`.

---

## C. Reference(s) to add for harmonic closeness — 🔶 decision

The paper currently cites only Marchiori & Latora (2000) and Rochat (2009) for harmonic
closeness. Earlier origins exist and strengthen the lineage:

- **Beauchamp, M.A. (1965), "An improved index of centrality," *Behavioral Science*
  10(2):161–163, DOI `10.1002/bs.3830100205`.** ⭐ Earliest proposal of the
  **sum-of-reciprocal-distances** (harmonic) form of closeness — introduced specifically
  to handle graphs that are *not strongly connected* (disconnected / "windowed"
  subgraphs = the localised-analysis case). This is the pre-2000 reference. Credited by
  Boldi & Vigna (2014) and the harmonic-centrality literature as the original.
- **Dekker, A.H. (2005), "Conceptual Distance in Social Network Analysis,"
  *Journal of Social Structure* 6(3).** Independent re-proposal under the name **"valued
  centrality"**; motivates harmonic by closeness's sensitivity to a single large distance
  / missing link. Optional, to complete the lineage
  Beauchamp 1965 → Marchiori & Latora 2000 → Dekker 2005 → Rochat 2009.

---

## D. MDPI requirements / compliance

- ✅ Abstract 139 words (limit 200).
- ✅ Keywords: 8 (range 3–10).
- ✅ References: MDPI numbered style (`mdpi.bst`).
- ✅ LaTeX template: `Definitions/mdpi.cls`.
- ⬜ **Back-matter order** must be: Supplementary Materials → **Author Contributions** →
  Funding → Data Availability Statement → Acknowledgments → Conflicts of Interest →
  References. **`\authorcontributions{}` is currently missing** — add it (placeholder
  "withheld for review" while anonymised) before `\funding`.
- 🔶 Supplementary Materials are inlined as a section; MDPI normally wants them as a
  **separate file** referenced via `\supplementary{}`. Inline is fine for review; split
  on acceptance if required.
- ⬜ Figures: ensure ≥300 dpi raster / vector PDF; captions below figures (already so).
- ⬜ Submission system (not in the .tex): cover letter, 3–5 suggested reviewers,
  highlights (`highlights.txt` exists), ORCID for each author.

---

## E. Outstanding — on de-anonymisation / acceptance

- ⬜ Real author names, affiliations, ORCID iDs, corresponding-author email
  (replace "Withheld for Review" in `\Author`/`\AuthorNames`/`\address`/`\corres`).
- ⬜ Funding details; complete Author Contributions (CRediT).
- ⬜ Insert permanent repository links (data prep + analysis code) in the Data Availability
  Statement and Methods (currently withheld + `\todo`).
- ⬜ Switch class option `submit` → `accept` only when instructed by the editorial office.

---

*Reference verification performed by parallel web-check of all 61 cited entries
(existence, metadata, DOI/URL resolution, contextual fit). Generated as a working
tracker — update as items are resolved.*
