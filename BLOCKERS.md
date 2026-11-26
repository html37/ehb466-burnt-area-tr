# Blockers (tasks that need a human)

Nothing blocks the automated pipeline. The items below can only be done by the students; the project continues with
the stated fallback.

1. **CRediT roles / task distribution (required for the paper).** `_inputs/team_roles.md` does not exist, so the
   paper's Author Contributions section contains a marked placeholder `[TODO: confirm roles]`.
   *Action:* write who did what (as presented in the interim talk) into `paper/sec_body.tex`, section
   "Author Contributions" (CRediT terms: Conceptualization, Data curation, Formal analysis, Investigation,
   Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review & editing), and
   update the speaker split in `reports/final_presentation_outline.md`. Then run `tectonic paper/main.tex`.
   *Fallback:* placeholder stays visible (do not submit like that).
2. **Instructor's interim feedback.** `_inputs/interim_feedback.md` does not exist, so no interim comments could be
   addressed explicitly. *Action:* if the instructor gave comments on 23 Nov, list them and check each against the
   paper (or ask Claude Code in a new session to address them).
3. **Optional: check labels.** `labeling/points.csv` (300 points) – fill `label` 0/1 following
   `labeling/NASIL_ETIKETLENIR.md`, then run `python scripts/75_eval_checklabels.py` and `python run_all.py`.
   *Fallback:* the paper lists independent check labelling as future work.
4. **Own commits and own words.** The course grades individual contributions (repository history, oral check) –
   each member should read the code walk-through in `reports/SUNUM_VE_SORU_CEVAP_TR.md`, proofread/rewrite the
   paper text in their own words and commit their own edits.
