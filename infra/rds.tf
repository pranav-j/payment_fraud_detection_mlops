resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-db-subnet-group"
  subnet_ids = data.aws_subnets.main.ids

  tags = { Name = "${var.project}-db-subnet-group" }
}

resource "aws_security_group" "rds" {
  name        = "${var.project}-rds-sg"
  description = "Created by RDS management console"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [
      aws_security_group.ecs.id,
      aws_security_group.lambda.id,
    ]
  }

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["${var.developer_ip}/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    ignore_changes = [description, tags]
  }

  tags = { Name = "${var.project}-rds-sg" }
}

resource "aws_db_instance" "main" {
  identifier        = "${var.project}-db"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t4g.micro"
  allocated_storage = 20
  storage_type      = "gp2"

  db_name  = "mlflow"
  username = "fraud_admin"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  skip_final_snapshot     = true
  deletion_protection     = false
  publicly_accessible     = true

  tags = { Name = "${var.project}-db" }

  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      engine_version,
      parameter_group_name,
      ca_cert_identifier,
      latest_restorable_time,
      password,
    ]
  }
}
