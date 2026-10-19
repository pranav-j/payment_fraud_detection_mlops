#!/usr/bin/env bash
# Batched AWS measurement session: bring the stack up, measure, tear it down.
#
# Everything that costs money needs AWS resources running, so it is all batched
# into one session to keep the stack up as briefly as possible.
#
# SAFETY. This script will not apply or destroy anything without you typing a
# confirmation at the prompt. Before each of those two steps it prints the
# Terraform plan summary and, separately, every destroy and replace line — the
# check that was skipped the day an apply replaced the production RDS (ADR-019).
#
#   ./scripts/measurement_session.sh plan      # show what each phase would do
#   ./scripts/measurement_session.sh up        # review plan, confirm, apply
#   ./scripts/measurement_session.sh measure   # run the measurements
#   ./scripts/measurement_session.sh down      # review plan, confirm, destroy
#   ./scripts/measurement_session.sh verify    # confirm idle state afterwards
#
# Run them in that order, checking the output between each. There is
# deliberately no "do everything" mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_DATE="${SESSION_DATE:-$(date +%F)}"
OUT_DIR="reports/perf/${SESSION_DATE}"
REGION="${AWS_DEFAULT_REGION:-ap-south-1}"

# Load levels for the synchronous path, in requests/second.
LOAD_LEVELS=(5 20 50)
LOAD_DURATION="${LOAD_DURATION:-3m}"

# Streaming producer settings.
PRODUCER_RATE="${PRODUCER_RATE:-10}"
PRODUCER_COUNT="${PRODUCER_COUNT:-3000}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

confirm() {
  local prompt="$1" answer
  printf '\n\033[1;33m%s\033[0m\n' "$prompt"
  read -r -p 'Type exactly YES to continue: ' answer
  [[ "$answer" == "YES" ]] || die "Not confirmed. Nothing was changed."
}

require_env() {
  [[ -f .env ]] || die ".env not found. Copy .env.example and fill it in."
  set -a; . ./.env; set +a
  : "${RDS_HOST:?RDS_HOST not set}"
  : "${RDS_USER:?RDS_USER not set}"
}

# Print the plan summary, then every destroy/replace line on its own.
# A destroy on this stack is almost always a mistake and has cost real data
# before, so it gets its own loud section rather than being buried in the plan.
review_plan() {
  local plan_file="$1"; shift
  log "Terraform plan"
  terraform -chdir=infra plan -out="$plan_file" "$@" | tail -40

  log "Every destroy / replace line in that plan"
  if terraform -chdir=infra show -no-color "$plan_file" \
      | grep -E "will be destroyed|must be replaced|will be replaced" ; then
    warn "The plan destroys or replaces resources. Read every line above."
  else
    echo "  (none)"
  fi
}

cmd_plan() {
  log "Session plan — nothing will be created"
  cat <<EOF

  Output directory : ${OUT_DIR}
  Region           : ${REGION}

  up       terraform apply (after plan review + confirmation)
           ~15 minutes for the stack to come up

  measure  1. Locust against the ALB at ${LOAD_LEVELS[*]} req/s, ${LOAD_DURATION} each
           2. Producer at ${PRODUCER_RATE} txn/s, ${PRODUCER_COUNT} transactions
           3. Ingest-to-decision percentiles from the decisions table
           4. Lambda cold/warm durations from CloudWatch REPORT lines
           5. The performance-drop demo (optional, see --with-demo)

  down     terraform destroy (after plan review + confirmation)

  verify   confirm nothing billable is still running

  Estimated cost of the session itself: see reports/cost/estimate.md.
  At \$0.154/hour, a two-hour session is roughly \$0.31 plus data transfer.

EOF
  terraform -chdir=infra init -input=false >/dev/null
  review_plan /tmp/fraud-mlops-session.tfplan
  warn "This was a plan only. Run '$0 up' to apply."
}

cmd_up() {
  log "Bringing the stack up"
  terraform -chdir=infra init -input=false >/dev/null
  review_plan infra/tfplan

  confirm "This will CREATE billable AWS resources in ${REGION}."
  terraform -chdir=infra apply infra/tfplan

  log "Stack outputs"
  terraform -chdir=infra output

  warn "Before measuring:"
  cat <<'EOF'
  1. Update .env with the new RDS_HOST / REDIS_CONNECTION_STRING / ALB DNS
  2. python -m fraud_mlops.db migrate
  3. Start the MLflow server:  bash scripts/run_mlflow_server.sh
  4. Register a model if the registry is empty (notebook 03, or the retrain flow)
  5. Materialise features:     python flows/materialize_features.py
  6. Warm the Lambda once so the first measurement isn't all cold start
EOF
}

cmd_measure() {
  require_env
  mkdir -p "$OUT_DIR"
  local alb
  alb="$(terraform -chdir=infra output -raw alb_dns_name)"
  [[ -n "$alb" ]] || die "No ALB DNS in terraform output. Is the stack up?"

  log "Recording session parameters"
  cat > "$OUT_DIR/session.json" <<EOF
{
  "session_date": "${SESSION_DATE}",
  "region": "${REGION}",
  "alb_dns_name": "${alb}",
  "load_levels_rps": [$(IFS=,; echo "${LOAD_LEVELS[*]}")],
  "load_duration": "${LOAD_DURATION}",
  "producer_rate_per_second": ${PRODUCER_RATE},
  "producer_count": ${PRODUCER_COUNT},
  "instance_sizes": {
    "rds": "db.t4g.micro, 20 GB gp2, Single-AZ",
    "redis": "t4g.micro",
    "fargate_api": "0.5 vCPU / 1 GB, ARM64",
    "fargate_mlflow": "1 vCPU / 2 GB, ARM64",
    "lambda": "1024 MB, arm64",
    "kinesis": "1 shard"
  }
}
EOF
  echo "  wrote $OUT_DIR/session.json"

  log "Health check"
  curl -fsS "http://${alb}/health" || die "ALB health check failed."
  echo
  curl -fsS "http://${alb}/model-info" || true
  echo

  # ── 1. Synchronous path ────────────────────────────────────────────────
  for rps in "${LOAD_LEVELS[@]}"; do
    log "Load test: ${rps} req/s for ${LOAD_DURATION}"
    # Locust's rate is users / mean wait time; locustfile uses between(0.5,1.5),
    # so mean wait is 1s and users ~= target RPS.
    uv run --with locust locust \
      -f load_test/locustfile.py \
      --host "http://${alb}" \
      --headless \
      --users "$rps" \
      --spawn-rate "$rps" \
      --run-time "$LOAD_DURATION" \
      --csv "${OUT_DIR}/locust-${rps}rps" \
      --only-summary
  done

  # ── 2. Streaming path ──────────────────────────────────────────────────
  log "Producing ${PRODUCER_COUNT} transactions at ${PRODUCER_RATE}/s"
  uv run python scripts/produce_transactions.py \
    --rate "$PRODUCER_RATE" \
    --count "$PRODUCER_COUNT" \
    --record-ground-truth \
    | tail -5

  log "Waiting 60s for the stream to drain"
  sleep 60

  # ── 3 & 4. Latency ─────────────────────────────────────────────────────
  log "Ingest-to-decision latency"
  uv run python scripts/streaming_latency.py --hours 2 --out "$OUT_DIR"

  log "Lambda duration, cold vs warm"
  uv run python scripts/lambda_latency.py --hours 2 --out "$OUT_DIR"

  log "Measurements written to ${OUT_DIR}"
  ls -la "$OUT_DIR"
}

cmd_demo() {
  require_env
  log "Performance-drop demo (produces traffic, waits for labels, alerts)"
  uv run python scripts/demo_performance_drop.py "$@"
}

# The hourly resources, and only those. A bare `terraform destroy` also takes
# out aws_db_instance.main — the RDS holding the MLflow registry and the
# decisions table — and then fails partway through on the ECR repositories,
# which carry prevent_destroy. That combination is how ADR-019's incident
# happened. RDS is left alone here; stop it separately if you want to.
TEARDOWN_TARGETS=(
  aws_ecs_service.api
  aws_ecs_service.mlflow
  aws_lb_listener.http
  aws_lb_target_group.api
  aws_lb.main
  aws_lambda_event_source_mapping.kinesis
  aws_kinesis_stream.transactions
)

cmd_down() {
  log "Tearing down the hourly resources"

  local target_args=()
  for t in "${TEARDOWN_TARGETS[@]}"; do target_args+=("-target=$t"); done

  terraform -chdir=infra plan -destroy -input=false "${target_args[@]}" \
    -out=infra/tfplan-destroy | tail -30

  log "Every destroy line in that plan"
  terraform -chdir=infra show -no-color infra/tfplan-destroy | grep -E "will be destroyed" || true

  warn "RDS is NOT in this list, on purpose: it holds the MLflow registry and the"
  warn "decisions table. Destroying it is the ADR-019 incident. Stop it separately"
  warn "if you want to, with: aws rds stop-db-instance --db-instance-identifier fraud-mlops-db"

  confirm "This will DESTROY the ALB, Kinesis stream and both Fargate services in ${REGION}."
  terraform -chdir=infra apply infra/tfplan-destroy

  log "Stopping the Redis EC2 instance"
  local rid
  rid="$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=${PROJECT:-fraud-mlops}-redis" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)"
  if [[ -n "$rid" && "$rid" != "None" ]]; then
    aws ec2 stop-instances --region "$REGION" --instance-ids "$rid" \
      --query 'StoppingInstances[0].CurrentState.Name' --output text
  else
    echo "  (no running Redis instance)"
  fi
}

cmd_verify() {
  log "Confirming idle state in ${REGION}"

  echo "RDS instances:"
  aws rds describe-db-instances --region "$REGION" \
    --query 'DBInstances[].[DBInstanceIdentifier,DBInstanceStatus]' --output text || true

  echo "Running EC2 instances:"
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text || true

  echo "Load balancers:"
  aws elbv2 describe-load-balancers --region "$REGION" \
    --query 'LoadBalancers[].[LoadBalancerName,State.Code]' --output text || true

  echo "Kinesis streams:"
  aws kinesis list-streams --region "$REGION" --query 'StreamNames' --output text || true

  echo "ECS services with desired count > 0:"
  aws ecs list-clusters --region "$REGION" --query 'clusterArns' --output text || true

  warn "Anything listed above is still costing money. Expected: nothing but empty results."
  echo
  echo "What SHOULD remain (and is in the idle figure in reports/cost/estimate.md):"
  echo "  S3 bucket, ECR repositories, Secrets Manager secrets, CloudWatch log groups"
}

case "${1:-}" in
  plan)    cmd_plan ;;
  up)      cmd_up ;;
  measure) cmd_measure ;;
  demo)    shift; cmd_demo "$@" ;;
  down)    cmd_down ;;
  verify)  cmd_verify ;;
  *)
    cat <<EOF
Usage: $0 {plan|up|measure|demo|down|verify}

  plan     Show what the session would do. Creates nothing.
  up       Apply Terraform, after plan review and confirmation.
  measure  Run the load tests and latency measurements.
  demo     Run the performance-drop demo.
  down     Destroy, after plan review and confirmation.
  verify   Confirm nothing billable is still running.

See docs/measurement-session.md for the full checklist.
EOF
    exit 1
    ;;
esac
