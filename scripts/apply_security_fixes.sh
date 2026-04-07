#!/bin/bash

# 🔒 Security Remediation Script
# Applies fixes for identified security issues in Fake News Investigator

set -e

echo "🔒 Applying Security Fixes for Fake News Investigator"
echo "=============================================="

# Backup original file
echo "📋 Creating backup..."
cp data/setup_data.py data/setup_data.py.backup.$(date +%Y%m%d_%H%M%S)

# Fix 1: HuggingFace unsafe downloads (add trust_remote_code=False)
echo "🛠️  Fixing HuggingFace unsafe downloads..."

# Replace first instance (line 44)
sed -i.tmp '44s/ds = load_dataset("ucsbnlp\/liar", revision="main")/ds = load_dataset("ucsbnlp\/liar", revision="main", trust_remote_code=False)/' data/setup_data.py

# Replace second instance (line 48)
sed -i.tmp '48s/ds = load_dataset("ucsbnlp\/liar", revision="main")/ds = load_dataset("ucsbnlp\/liar", revision="main", trust_remote_code=False)/' data/setup_data.py

# Add security comments
sed -i.tmp '43a\        # Security: Explicitly disable remote code execution for safety' data/setup_data.py
sed -i.tmp '48a\        # Security: Fallback also disables remote code execution' data/setup_data.py

# Clean up temp files
rm -f data/setup_data.py.tmp

# Fix 2: Add security documentation for URL validation (already secure, just document)
echo "📝 Adding security documentation..."

# Add comment before the URL validation to explain security
sed -i.tmp '236a\    # Security: URL scheme validation prevents SSRF attacks' data/setup_data.py
rm -f data/setup_data.py.tmp

echo "✅ Security fixes applied successfully!"

# Verification
echo "🔍 Verifying fixes..."

# Check that trust_remote_code=False was added
if grep -q "trust_remote_code=False" data/setup_data.py; then
    echo "  ✅ HuggingFace trust_remote_code=False added"
else
    echo "  ❌ Failed to add trust_remote_code=False"
    exit 1
fi

# Check that security comments were added
if grep -q "Security: Explicitly disable remote code execution" data/setup_data.py; then
    echo "  ✅ Security documentation added"
else
    echo "  ❌ Failed to add security documentation"
    exit 1
fi

echo "🎉 All security fixes applied and verified!"
echo ""
echo "📋 Summary of changes:"
echo "  - Added trust_remote_code=False to HuggingFace dataset loading"
echo "  - Added security documentation comments"
echo "  - Original file backed up to data/setup_data.py.backup.*"
echo ""
echo "🔄 Next steps:"
echo "  1. Test that dataset loading still works: python data/setup_data.py"
echo "  2. Run security scan to verify fixes: bandit -r data/setup_data.py"
echo "  3. Commit changes: git add data/setup_data.py && git commit -m 'security: fix HuggingFace unsafe downloads'"
echo ""
echo "🛡️  Security posture improved!"