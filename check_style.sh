#!/bin/bash
set -e

# Directories
SRC_DIR="./include"
TEST_DIR="./tests"

echo "📝 Discovering source/header files..."

FILES=$(find "$SRC_DIR" "$TEST_DIR" \
    -path "$TEST_DIR/build" -prune -o \
    -type f \( \
        -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' -o \
        -name '*.h' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.hh' \
    \) -print)

if [ -z "$FILES" ]; then
    echo "⚠️ No files found to process."
    exit 0
fi

echo "🎯 Files to check:"
echo "$FILES"

echo "🎨 Running clang-format..."
for file in $FILES; do
    echo "Formatting $file"
    clang-format -i "$file"
done

echo "🔍 Running clang-tidy..."
for file in $FILES; do
    echo "Linting $file"
    clang-tidy "$file" --extra-arg=-std=c++20 -- -I"$SRC_DIR" || true
done

echo "✅ Formatting and linting complete."
