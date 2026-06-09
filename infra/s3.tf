# S3 bucket already exists — we import it rather than recreate
# to avoid destroying existing model artifacts and drift reports
data "aws_s3_bucket" "main" {
  bucket = "fraud-mlops-kidiloski"
}
