# Demo video — narration script and shot list

**Length: 312 words**, which is about 2:14 read at a natural 140 words per
minute, or 2:05 at 150. If you want it strictly under two minutes, drop the last
sentence of section 3 and the last sentence of section 5 — both are asides, and
the beats survive without them.

Read it straight through in one take if you can; I cut the visuals to fit your
audio afterwards, not the other way round.

**Who it's for:** someone who clicked the repo from a CV and will give it ninety
seconds before deciding whether to read the README. So it opens with what the
thing is, and the strongest beat (the leakage finding) lands before the halfway
mark — not saved for the end.

**Tone:** the same first-person engineering voice as the README. You're walking
a competent peer through it, not pitching.

---

## How to record

- Just the voice. Don't try to narrate live while clicking — record audio alone,
  send it to me, and I'll cut the picture to match.
- Any format (`.m4a`, `.wav`, `.mp3`). Phone voice-memo quality is fine.
- Leave a beat of silence between the numbered sections; it gives me clean cut
  points.
- If you fluff a line, pause two seconds and say it again — I'll take the second
  one.

---

## The script

### 1 · What it is — 0:00–0:20

> This is a real-time fraud detection pipeline on AWS. Kinesis and Lambda for
> streaming inference, a Feast feature store, MLflow for the model registry, and
> the whole stack defined in Terraform. I want to show you three things it does,
> and one thing it taught me.

**On screen:** README top — title, CI badge, the four result bullets.

---

### 2 · The streaming path — 0:20–0:38

> A producer pushes transactions into Kinesis. Lambda picks up each batch, pulls
> that sender's history from Redis through Feast, scores it, and writes the
> verdict to Postgres — with the model version and the latency on every row.
> Warm, that's forty-four milliseconds.

**On screen:** split — producer terminal streaming on the left, `psql` tailing
`decisions` on the right, rows appearing. Then the latency report.
**Needs:** AWS stack up.

---

### 3 · What it taught me — 0:38–1:17

> Now the thing it taught me. I re-measured the model honestly — split by time,
> threshold tuned on validation, bootstrap intervals throughout. And a
> three-condition rule I wrote by hand beat every model I'd trained. PaySim
> generates fraud by emptying the account, so the balance columns give the answer
> away. Take them out and the model falls from point nine seven PR-AUC to point
> two one. That's the most important thing to know about any result on this
> dataset, so it's in the README rather than buried in a notebook.

**On screen:** `reports/figures/evaluation-leakage.png`, held. Then the
README's Evaluation table.
**Needs:** nothing — committed artifacts.

---

### 4 · Drift monitoring — 1:17–1:29

> Monitoring. I inject transactions with amounts about a thousand times normal.
> Evidently catches the distribution shift, pushes the metrics to Prometheus,
> Grafana lights up, and SNS sends the alert.

**On screen:** `inject_drift.py` running → Grafana drift panel going red →
the alert email.
**Needs:** AWS stack up (SNS, RDS). Grafana is local.

---

### 5 · What drift cannot see — 1:29–2:02

> But drift only watches the inputs. If fraud changes shape *within* the normal
> range, the inputs barely move and recall quietly collapses. So a second flow
> measures whether the model was actually right — against labels that have
> matured. Real fraud labels are chargebacks; they arrive weeks later. The
> dashboard shows how much of each window it can't judge yet, because recall
> over labels that haven't arrived is a lie with a decimal point on it.

**On screen:** the realized-performance panels — recall against the floor line,
then the label-maturity panel beside it.
**Needs:** nothing — I can replay this locally, as in the committed screenshot.

---

### 6 · Close — 2:02–2:14

> It costs fifteen cents an hour running, two dollars a month torn down. Every
> number in the README comes from a file a committed script generated. Code's
> linked below.

**On screen:** cost table, then the repo page.

---

## Shot checklist before recording

Sections 3, 5 and 6 I can capture from what's already committed. Sections 2 and
4 need the stack up:

- [ ] `./scripts/measurement_session.sh up`
- [ ] `.env` refreshed, `python -m fraud_mlops.db migrate`
- [ ] MLflow server running, a model on `@production`
- [ ] `make monitoring-up`
- [ ] Feast materialised — otherwise every lookup misses and the feature store
      contributes nothing on camera
- [ ] SNS email subscription confirmed
- [ ] `./scripts/measurement_session.sh down` afterwards

## What I do with the audio

1. Cut each section's visual to your actual timings rather than the estimates
   above.
2. Capture the browser shots headlessly (Playwright records video directly), and
   render the terminal shots.
3. Mux with ffmpeg, export 1080p MP4.
4. Hand you the file; the README link is the last TODO left in it.
