#!/usr/bin/env bash
# Deploy the Colonist artifacts S3 bucket (CloudFormation).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$ROOT/infra/aws/storage-stack.yaml"

STACK_NAME="${STACK_NAME:-catanatron-artifacts-${ENVIRONMENT:-dev}}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
PROJECT_NAME="${PROJECT_NAME:-catanatron}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
BUCKET_NAME="${BUCKET_NAME:-}"
RETAIN_BUCKET="${RETAIN_BUCKET:-true}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install and configure it first." >&2
  exit 1
fi

echo "==> Account / region"
aws sts get-caller-identity --output table
echo "Region: $AWS_REGION  Stack: $STACK_NAME  Environment: $ENVIRONMENT"

PARAMS=(
  "ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME"
  "ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
  "ParameterKey=RetainBucketOnDelete,ParameterValue=$RETAIN_BUCKET"
)
if [[ -n "$BUCKET_NAME" ]]; then
  PARAMS+=("ParameterKey=BucketName,ParameterValue=$BUCKET_NAME")
fi

echo "==> Deploy CloudFormation stack"
aws cloudformation deploy \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --parameter-overrides "${PARAMS[@]}" \
  --no-fail-on-empty-changeset \
  --tags "Project=$PROJECT_NAME" "Environment=$ENVIRONMENT"

echo ""
echo "==> Stack outputs"
aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs" \
  --output table

BUCKET="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" \
  --output text)"

POLICY="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ArtifactsPolicyArn'].OutputValue" \
  --output text)"

echo ""
echo "Next steps:"
echo "  export CATANATRON_S3_BUCKET=$BUCKET"
echo "  export AWS_DEFAULT_REGION=$AWS_REGION"
echo "  # Attach IAM policy to your EC2 role: $POLICY"
echo "  ./scripts/aws_sync_run.sh runs/<run_id>"
