resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.project}-mlflow"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project}-api"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project}-scorer"
  retention_in_days = 7
}

resource "aws_sns_topic" "drift_alerts" {
  name = "${var.project}-drift-alerts"
}

resource "aws_sns_topic_subscription" "drift_email" {
  topic_arn = aws_sns_topic.drift_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
