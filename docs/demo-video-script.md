# Demo video — narration script and shot list

**Length: 313 words**, about 2:14 read at a natural 140 words per minute. If you
want it under two minutes, cut the last sentence of section 3 and the last
sentence of section 5 — both are asides and the beats hold without them.

Read it straight through in one take if you can; I cut the visuals to fit your
audio afterwards, not the other way round.

**Who it's for:** someone who clicked the repo from a CV and will give it ninety
seconds before deciding whether to read the README. So it opens with what the
thing is, and the strongest beat (the leakage finding) lands before the halfway
mark — not saved for the end.

**Tone:** your README voice — declarative, concrete, one judgment per beat. It
leans on the constructions you already use there: arrow notation for the
pipeline, flat statements of fact before the consequence, no hedging. Change any
line that doesn't sound like you saying it; the timings below adjust.

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

### 1 · What it is — 0:00–0:23

> This is a real-time fraud detection pipeline on AWS. Kinesis and Lambda for
> streaming inference, Feast for the feature store, MLflow for the registry, and
> the whole stack in Terraform. The goal was never the best possible fraud model.
> It was to build the system around a model the way a real team would.

**On screen:** README top — title, CI badge, the four result bullets.

---

### 2 · The streaming path — 0:23–0:47

> The streaming path is the core. Producer, Kinesis, Lambda. Each batch triggers
> the function: look up the sender's history from Redis through Feast, score the
> transaction, write the verdict to Postgres. Nothing waits on the result — it's
> fire-and-forget. Every row carries the model version and the ingest-to-decision
> latency. Warm, the function runs in forty-four milliseconds.

**On screen:** split — producer terminal streaming on the left, `psql` tailing
`decisions` on the right, rows appearing. Then the latency report.
**Needs:** AWS stack up.

---

### 3 · What it taught me — 0:47–1:21

> Then I re-measured the model properly. Split by time, threshold tuned on a
> validation window, bootstrap intervals on everything. A three-condition rule I
> wrote by hand beat every model I had trained. PaySim generates fraud by draining
> the origin account, so the balance columns hand you the label. Remove them and
> PR-AUC falls from point nine seven to point two one. That is the most important
> fact about any result on this dataset, so it goes in the README.

**On screen:** `reports/figures/evaluation-leakage.png`, held. Then the
README's Evaluation table.
**Needs:** nothing — committed artifacts.

---

### 4 · Drift monitoring — 1:21–1:32

> Monitoring. I inject transactions with amounts a thousand times normal.
> Evidently catches the shift, pushes the metrics to Prometheus, Grafana lights
> up, SNS sends the alert.

**On screen:** `inject_drift.py` running → Grafana drift panel going red →
the alert email.
**Needs:** AWS stack up (SNS, RDS). Grafana is local.

---

### 5 · What drift cannot see — 1:32–2:02

> Drift only watches the inputs. If fraud changes shape inside the normal range,
> the inputs barely move and recall drops with nothing firing. So a second flow
> measures whether the model was right, against labels that have matured. Fraud
> labels are chargebacks — they arrive weeks after the transaction. The dashboard
> shows how much of each window it cannot judge yet. Recall computed over labels
> that have not arrived means nothing.

**On screen:** the realized-performance panels — recall against the floor line,
then the label-maturity panel beside it.
**Needs:** nothing — I can replay this locally, as in the committed screenshot.

---

### 6 · Close — 2:02–2:14

> Fifteen cents an hour running, two dollars a month torn down. Every number in
> the README comes from a file a committed script generated. Code is below.

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
