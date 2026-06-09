resource "aws_lambda_function" "scorer" {
  function_name = "${var.project}-scorer"
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda.repository_url}:v5"
  role          = aws_iam_role.lambda.arn
  architectures = ["arm64"]
  memory_size   = 1024
  timeout       = 120

  vpc_config {
    subnet_ids         = tolist(data.aws_subnets.main.ids)
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      RDS_HOST              = aws_db_instance.main.address
      RDS_PORT              = "5432"
      RDS_DB                = "mlflow"
      RDS_USER              = "fraud_admin"
      RDS_PASSWORD          = var.db_password
      MLFLOW_TRACKING_URI   = "postgresql+psycopg://fraud_admin:${var.db_password}@${aws_db_instance.main.address}:5432/mlflow?sslmode=require"
      REDIS_CONNECTION_STRING = "${aws_instance.redis.private_ip}:6379"
    }
  }
}

resource "aws_lambda_event_source_mapping" "kinesis" {
  event_source_arn              = aws_kinesis_stream.transactions.arn
  function_name                 = aws_lambda_function.scorer.arn
  starting_position             = "LATEST"
  batch_size                    = 10
  bisect_batch_on_function_error = true
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = data.aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = data.aws_route_tables.main.ids

  tags = { Name = "${var.project}-s3-endpoint" }
}

data "aws_route_tables" "main" {
  vpc_id = data.aws_vpc.main.id
}
