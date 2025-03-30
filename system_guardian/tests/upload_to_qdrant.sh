#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Animation for loading
function show_loading() {
  local pid=$1
  local message=$2
  local spin='-\|/'
  local i=0
  while kill -0 $pid 2>/dev/null; do
    i=$(( (i+1) % 4 ))
    printf "\r${BLUE}%s${NC} ${spin:$i:1}" "$message"
    sleep 0.1
  done
  printf "\r"
}

# API endpoint
ENDPOINT="http://localhost:5566/api/vector-db/knowledge/upload"

# Get the first 8 markdown files sorted alphabetically
FILES=$(find docs/test_resolution_knowledge -type f -name "*.md" | sort | head -8)
FILE_COUNT=$(echo "$FILES" | wc -l)

if [ -z "$FILES" ]; then
  echo -e "${RED}No markdown files found in the directory.${NC}"
  exit 1
fi

echo -e "${GREEN}Found $FILE_COUNT files to upload${NC}"
echo -e "${YELLOW}Starting upload to Qdrant...${NC}"

# Counter for success and failures
SUCCESS=0
FAILED=0

# Loop through each file and upload
for file in $FILES; do
  echo -e "${BLUE}Uploading: ${NC}$file"
  
  # Create a background process for the curl call
  (curl -s -X POST \
    -F "file=@$file" \
    "$ENDPOINT" > /dev/null) &
  
  # Get the PID of the background process
  curl_pid=$!
  
  # Show loading animation
  show_loading $curl_pid "Uploading $file "
  
  # Check if the curl process exited successfully
  wait $curl_pid
  exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully uploaded:${NC} $file"
    SUCCESS=$((SUCCESS+1))
  else
    echo -e "${RED}✗ Failed to upload:${NC} $file"
    FAILED=$((FAILED+1))
  fi
  
  # Wait a bit to avoid overwhelming the server
  sleep 0.5
done

echo
echo -e "${GREEN}==== Upload Summary ====${NC}"
echo -e "${GREEN}Successfully uploaded:${NC} $SUCCESS files"
if [ $FAILED -gt 0 ]; then
  echo -e "${RED}Failed to upload:${NC} $FAILED files"
fi
echo -e "${GREEN}======================${NC}" 