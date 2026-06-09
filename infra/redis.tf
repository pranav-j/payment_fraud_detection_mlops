data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

resource "aws_instance" "redis" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = "t4g.micro"
  key_name                    = var.redis_key_name
  subnet_id                   = tolist(data.aws_subnets.main.ids)[0]
  vpc_security_group_ids      = [aws_security_group.redis.id]
  associate_public_ip_address = true

  user_data = <<-EOF
    #!/bin/bash
    dnf update -y
    dnf install -y redis6
    systemctl enable redis6
    systemctl start redis6
    sed -i "s/^bind 127.0.0.1/bind 0.0.0.0/" /etc/redis6/redis6.conf
    redis6-cli CONFIG SET maxmemory 700mb
    redis6-cli CONFIG SET maxmemory-policy allkeys-lru
    systemctl restart redis6
  EOF

  tags = { Name = "${var.project}-redis" }
}
