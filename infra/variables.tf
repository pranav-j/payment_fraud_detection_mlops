variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Project name used as a prefix for all resources"
  type        = string
  default     = "fraud-mlops"
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "redis_key_name" {
  description = "EC2 key pair name for Redis instance"
  type        = string
  default     = "fraud-mlops-key"
}

variable "alert_email" {
  description = "Email address for drift alerts"
  type        = string
  default     = "pranavjayaprakash1999@gmail.com"
}

variable "developer_ip" {
  description = "Developer Mac IP for RDS direct access"
  type        = string
}
