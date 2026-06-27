# AWS artifact storage (S3)

Private S3 bucket for **training runs**, **teacher data**, and **promoted model checkpoints** produced by the Colonist 1v1 pipeline.

## What gets stored

| Prefix | Local path | Contents |
|--------|------------|----------|
| `runs/<run_id>/` | `runs/<run_id>/` | PPO checkpoints, `bc.pt`, eval JSON, `run_manifest.json`, `training_events.jsonl`, logs |
| `data/<dataset_id>/` | `data/<dataset_id>/` | Teacher parquet from data generation |
| `models/<name>/` | (optional) | Long-lived promoted checkpoints (e.g. `models/production/`) |

Example: local `runs/ec2_proxy_500k/` → `s3://<bucket>/runs/ec2_proxy_500k/`.

## Cost (dev-sized, rough)

| Item | Estimate |
|------|----------|
| Storage | ~$0.023/GB-month (S3 Standard, us-east-1) |
| Requests | Pennies for occasional sync |
| Example | 20 GB of checkpoints ≈ **$0.50/month** |

Versioning is enabled; non-current versions in `runs/` expire after 90 days in **dev** (disabled in **prod** stack parameter).

## Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured (`aws sts get-caller-identity` works)
- Permission to create S3 buckets and IAM managed policies in the target account/region

## Deploy the bucket

```bash
# From repo root
./scripts/aws_storage_deploy.sh

# Or with options:
AWS_REGION=us-east-1 ENVIRONMENT=dev ./scripts/aws_storage_deploy.sh
```

Stack name default: `catanatron-artifacts-dev`. Outputs include `BucketName` and `ArtifactsPolicyArn`.

### Attach permissions on EC2

1. Open the EC2 instance **IAM role** in the console.
2. Attach the managed policy from stack output `ArtifactsPolicyArn` (e.g. `catanatron-artifacts-dev-rw`).
3. On the instance, set:

```bash
export CATANATRON_S3_BUCKET="<BucketName from stack output>"
export AWS_DEFAULT_REGION="us-east-1"   # same region as the bucket
```

4. After training, sync a run:

```bash
./scripts/aws_sync_run.sh runs/ec2_proxy_500k
```

Or enable automatic upload at the end of `scripts/ec2_run_training.sh`:

```bash
CATANATRON_S3_BUCKET=my-bucket ./scripts/ec2_run_training.sh
```

### Local machine (download a run)

```bash
export CATANATRON_S3_BUCKET=my-bucket
./scripts/aws_sync_run.sh runs/ec2_proxy_500k --pull
```

## Python helper (optional)

```bash
pip install boto3
python scripts/aws_storage.py upload-run runs/my_run --bucket "$CATANATRON_S3_BUCKET"
python scripts/aws_storage.py download-run runs/my_run --bucket "$CATANATRON_S3_BUCKET"
python scripts/aws_storage.py list-runs --bucket "$CATANATRON_S3_BUCKET"
```

## Delete the stack

```bash
aws cloudformation delete-stack --stack-name catanatron-artifacts-dev
```

With default `RetainBucketOnDelete=true`, the bucket and objects are **kept** when the stack is removed. Empty the bucket first if you need a full delete.
