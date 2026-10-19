resource "aws_secretsmanager_secret" "rds_uri" {
  name        = "${var.project}/rds/uri"
  description = "Full RDS connection URI for MLflow backend store"
}

# The dialect here must match the driver installed in Dockerfile.mlflow.
# It said +psycopg (psycopg v3) while the image installs psycopg2-binary, so
# the MLflow task died on "No module named 'psycopg'" every time it started.
resource "aws_secretsmanager_secret_version" "rds_uri" {
  secret_id     = aws_secretsmanager_secret.rds_uri.id
  secret_string = "postgresql+psycopg2://fraud_admin:${var.db_password}@${aws_db_instance.main.address}:5432/mlflow?sslmode=require" # pragma: allowlist secret
}
