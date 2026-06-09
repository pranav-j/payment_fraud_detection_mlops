terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "fraud-mlops-kidiloski"
    key            = "terraform/state.tfstate"
    region         = "ap-south-1"
    use_lockfile = true
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}
