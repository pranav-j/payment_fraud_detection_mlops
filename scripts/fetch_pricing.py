#!/usr/bin/env python3
"""Pull ap-south-1 on-demand rates from the AWS Price List API.

The cost estimate has to come from somewhere defensible. Typing rates off a
pricing page into a YAML file means they are wrong within a quarter and nobody
knows when. This queries the Price List API instead, and records for every rate
its SKU, the rate description AWS gives it, and the publication date of the
price list it came from.

The Price List API is free and read-only — it creates nothing and starts
nothing. It does need credentials, and `pricing:GetProducts` permission.

    python scripts/fetch_pricing.py
    python scripts/fetch_pricing.py --out reports/cost/pricing-inputs.json

Rates it cannot find are written with `"value": null` and an explanatory note,
so a gap shows up in the estimate as a gap rather than as a zero.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger("fetch_pricing")

REGION = "ap-south-1"
# The Price List API itself only lives in a few regions; the rates it returns
# are for whatever regionCode is filtered on.
PRICING_API_REGION = "us-east-1"

# Each entry: a readable key, the service code, and the TERM_MATCH filters that
# identify exactly one on-demand price dimension.
#
# These filters are the fiddly part. AWS's product catalogue has many SKUs per
# instance type (capacity reservations, dedicated tenancy, bring-your-own
# licence), and leaving a dimension unfiltered silently returns the wrong one.
RATE_QUERIES: list[dict[str, Any]] = [
    {
        "key": "ec2_t4g_micro_hour",
        "label": "EC2 t4g.micro (Redis host), on-demand Linux",
        "service": "AmazonEC2",
        "filters": {
            "instanceType": "t4g.micro",
            "operatingSystem": "Linux",
            "tenancy": "Shared",
            "preInstalledSw": "NA",
            "capacitystatus": "Used",
            "licenseModel": "No License required",
        },
    },
    {
        "key": "rds_db_t4g_micro_hour",
        "label": "RDS db.t4g.micro PostgreSQL, Single-AZ",
        "service": "AmazonRDS",
        "filters": {
            "instanceType": "db.t4g.micro",
            "databaseEngine": "PostgreSQL",
            "deploymentOption": "Single-AZ",
        },
    },
    {
        "key": "rds_gp2_gb_month",
        "label": "RDS gp2 storage",
        "service": "AmazonRDS",
        "filters": {
            "productFamily": "Database Storage",
            "volumeType": "General Purpose",
            "deploymentOption": "Single-AZ",
            "databaseEngine": "PostgreSQL",
        },
    },
    {
        "key": "alb_hour",
        "label": "Application Load Balancer, per hour",
        "service": "AWSELB",
        "filters": {
            "productFamily": "Load Balancer-Application",
            "usagetype": "APS3-LoadBalancerUsage",
        },
    },
    {
        "key": "alb_lcu_hour",
        "label": "ALB Load Balancer Capacity Unit, per hour",
        "service": "AWSELB",
        "filters": {
            "productFamily": "Load Balancer-Application",
            "usagetype": "APS3-LCUUsage",
        },
    },
    {
        "key": "kinesis_shard_hour",
        "label": "Kinesis Data Streams provisioned shard, per hour",
        "service": "AmazonKinesis",
        # APS3-Storage-ShardHour is the provisioned-mode shard. The similarly
        # named APS3-Extended-ShardHour is extended retention and costs more.
        "filters": {"usagetype": "APS3-Storage-ShardHour"},
    },
    {
        "key": "kinesis_put_payload_unit",
        "label": "Kinesis PUT payload unit (25 KB), each",
        "service": "AmazonKinesis",
        "filters": {"usagetype": "APS3-PutRequestPayloadUnits"},
    },
    {
        "key": "lambda_arm_gb_second",
        "label": "Lambda arm64 duration, per GB-second",
        "service": "AWSLambda",
        "filters": {"usagetype": "APS3-Lambda-GB-Second-ARM"},
    },
    {
        "key": "lambda_arm_request",
        "label": "Lambda arm64 request, each",
        "service": "AWSLambda",
        "filters": {"usagetype": "APS3-Request-ARM"},
    },
    {
        "key": "fargate_arm_vcpu_hour",
        "label": "Fargate ARM vCPU, per hour",
        "service": "AmazonECS",
        "filters": {"usagetype": "APS3-Fargate-ARM-vCPU-Hours:perCPU"},
    },
    {
        "key": "fargate_arm_gb_hour",
        "label": "Fargate ARM memory, per GB-hour",
        "service": "AmazonECS",
        "filters": {"usagetype": "APS3-Fargate-ARM-GB-Hours"},
    },
    {
        "key": "s3_standard_gb_month",
        "label": "S3 Standard storage, per GB-month (first 50 TB)",
        "service": "AmazonS3",
        "filters": {"productFamily": "Storage", "storageClass": "General Purpose"},
    },
    {
        "key": "ecr_storage_gb_month",
        "label": "ECR image storage, per GB-month",
        "service": "AmazonECR",
        # Without the usagetype this matches APS3-AsyncActions-ImageSigning
        # first — a $0.02 "per Count" rate that has nothing to do with storage.
        # Exactly the failure the filters are there to prevent.
        "filters": {"usagetype": "APS3-TimedStorage-ByteHrs"},
    },
    {
        "key": "secretsmanager_secret_month",
        "label": "Secrets Manager, per secret per month",
        "service": "AWSSecretsManager",
        "filters": {"usagetype": "APS3-AWSSecretsManager-Secrets"},
    },
    {
        "key": "cloudwatch_logs_ingest_gb",
        "label": "CloudWatch Logs ingestion, per GB",
        "service": "AmazonCloudWatch",
        "filters": {"usagetype": "APS3-DataProcessing-Bytes"},
    },
    {
        "key": "public_ipv4_hour",
        "label": "Public IPv4 address, per hour",
        "service": "AmazonVPC",
        "filters": {"usagetype": "APS3-PublicIPv4:InUseAddress"},
    },
]


def _extract_on_demand(price_item: dict) -> tuple[float, str, str] | None:
    """Pull (price, unit, description) out of one Price List product."""
    terms = price_item.get("terms", {}).get("OnDemand", {})
    for term in terms.values():
        for dimension in term.get("priceDimensions", {}).values():
            usd = dimension.get("pricePerUnit", {}).get("USD")
            if usd is None:
                continue
            value = float(usd)
            # Free tiers and zero-rated dimensions are real entries; skip them
            # so a $0 tier doesn't get mistaken for the actual rate.
            if value == 0.0:
                continue
            return value, dimension.get("unit", ""), dimension.get("description", "")
    return None


def fetch_rate(client: Any, query: dict) -> dict:
    """One rate, with everything needed to audit where it came from."""
    filters = [
        {"Type": "TERM_MATCH", "Field": "regionCode", "Value": REGION},
        *[
            {"Type": "TERM_MATCH", "Field": field, "Value": value}
            for field, value in query["filters"].items()
        ],
    ]

    result: dict[str, Any] = {
        "label": query["label"],
        "service": query["service"],
        "filters": query["filters"],
        "value": None,
        "unit": None,
        "aws_description": None,
        "sku": None,
        "price_list_published": None,
        "note": None,
    }

    try:
        response = client.get_products(ServiceCode=query["service"], Filters=filters, MaxResults=20)
    except Exception as e:
        result["note"] = f"Price List API call failed: {e}"
        logger.warning("%s: %s", query["key"], result["note"])
        return result

    candidates = [json.loads(p) for p in response.get("PriceList", [])]
    if not candidates:
        result["note"] = "No product matched these filters — they may need narrowing."
        logger.warning("%s: no match", query["key"])
        return result

    for item in candidates:
        extracted = _extract_on_demand(item)
        if extracted is None:
            continue
        value, unit, description = extracted
        result.update(
            {
                "value": value,
                "unit": unit,
                "aws_description": description,
                "sku": item.get("product", {}).get("sku"),
                "price_list_published": item.get("publicationDate"),
            }
        )
        if len(candidates) > 1:
            result["note"] = (
                f"{len(candidates)} products matched; took the first with a non-zero "
                f"on-demand rate. Check `aws_description` reads as expected."
            )
        logger.info("%s = $%s per %s", query["key"], value, unit)
        return result

    result["note"] = f"{len(candidates)} products matched but none had a non-zero rate."
    logger.warning("%s: %s", query["key"], result["note"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/cost/pricing-inputs.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = boto3.client("pricing", region_name=PRICING_API_REGION)
    rates = {q["key"]: fetch_rate(client, q) for q in RATE_QUERIES}

    missing = [k for k, v in rates.items() if v["value"] is None]
    payload = {
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "region": REGION,
        "currency": "USD",
        "source": "AWS Price List API (pricing:GetProducts), on-demand terms",
        "rates_missing": missing,
        "rates": rates,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nWrote {args.out}")
    print(f"  {len(rates) - len(missing)} of {len(rates)} rates resolved")
    if missing:
        print(f"  Missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
