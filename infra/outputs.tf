output "alb_dns_name" {
  description = "ALB public DNS name — use this to call the API"
  value       = aws_lb.main.dns_name
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint"
  value       = aws_db_instance.main.address
  sensitive   = true
}

output "ecr_mlflow_url" {
  description = "ECR repository URL for MLflow image"
  value       = aws_ecr_repository.mlflow.repository_url
}

output "ecr_api_url" {
  description = "ECR repository URL for API image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_lambda_url" {
  description = "ECR repository URL for Lambda image"
  value       = aws_ecr_repository.lambda.repository_url
}

output "kinesis_stream_arn" {
  description = "Kinesis stream ARN"
  value       = aws_kinesis_stream.transactions.arn
}

output "redis_private_ip" {
  description = "Redis EC2 private IP (for Lambda and ECS)"
  value       = aws_instance.redis.private_ip
}
