resource "aws_kinesis_stream" "transactions" {
  name        = "${var.project}-transactions"
  shard_count = 1

  tags = { Name = "${var.project}-transactions" }
}
