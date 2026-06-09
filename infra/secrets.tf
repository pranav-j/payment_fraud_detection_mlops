resource "aws_secretsmanager_secret" "rds_uri" {
  name        = "${var.project}/rds/uri"
  description = "Full RDS connection URI for MLflow backend store"
}

resource "aws_secretsmanager_secret_version" "rds_uri" {
  secret_id = aws_secretsmanager_secret.rds_uri.id
  secret_string = "postgresql+psycopg://fraud_admin:${var.db_password}@${aws_db_instance.main.address}:5432/mlflow?sslmode=require" # pragma: allowlist secret
}
