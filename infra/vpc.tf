# Use the existing default VPC — no need to create a new one
data "aws_vpc" "default" {
  filter {
    name   = "isDefault"
    values = ["true"]
  }
}

# Use existing subnets in the RDS VPC
# We target the non-default VPC where RDS lives
data "aws_vpc" "main" {
  id = "vpc-07417979b25261bdc"
}

data "aws_subnets" "main" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
}

# ALB security group
resource "aws_security_group" "alb" {
  name        = "${var.project}-alb-sg"
  description = "ALB: inbound HTTP from internet"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-alb-sg" }

  lifecycle {
    ignore_changes = [description, tags]
  }
}

# ECS security group
resource "aws_security_group" "ecs" {
  name        = "${var.project}-ecs-sg"
  description = "ECS tasks"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    from_port = 5000
    to_port   = 5000
    protocol  = "tcp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-ecs-sg" }

  lifecycle {
    ignore_changes = [description, tags]
  }
}

# Redis security group
resource "aws_security_group" "redis" {
  name        = "${var.project}-redis-sg"
  description = "Redis on EC2"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.lambda.id]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # tighten in production
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-redis-sg" }
}

# Lambda security group
resource "aws_security_group" "lambda" {
  name        = "${var.project}-lambda-sg"
  description = "Lambda function outbound to RDS"
  vpc_id      = data.aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-lambda-sg" }

  lifecycle {
    ignore_changes = [description, tags]
  }
}
