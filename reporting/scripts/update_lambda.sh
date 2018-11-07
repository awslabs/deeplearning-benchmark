#!/bin/bash
# Updates the Lambda code in AWS to generate the report daily.
# Ensure AWS environment variables 
# The AWS CLI requires the following environment variables to be set accordingly.
# * AWS_ACCESS_KEY_ID
# * AWS_SECRET_ACCESS_KEY
# * AWS_DEFAULT_REGION
# Run this from the main directory, e.g. scripts/update_lambda.sh

set -exuo pipefail

# Create a temporary directory and copy the code to that directory.
TMP_DIR=$(mktemp -d)
cp -r . ${TMP_DIR}

# Prepare the zip file.
pushd ${TMP_DIR}
pip3 install -r requirements.txt -t ./
ZIP_PACKAGE='bai-report-lambda.zip'
zip -r ${ZIP_PACKAGE} .

# Upload the zip 
echo "Updating Lambda with package: ${TMP_DIR}/${ZIP_PACKAGE}"
aws lambda update-function-code --function-name benchmark-ai-report --zip fileb://${ZIP_PACKAGE}
popd

