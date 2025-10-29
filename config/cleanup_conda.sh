#!/bin/bash
# Conda Cleanup Script
# ====================
# This script helps free up disk space from conda cache and unused packages.

echo "======================================================================"
echo "Conda Cleanup Script"
echo "======================================================================"
echo ""

# Check available space before cleanup
echo "Current disk usage:"
df -h . | awk 'NR==2 {print "Available: "$4" / "$2" ("$5" used)"}'
echo ""

# Show conda cache size
echo "Checking conda cache size..."
CACHE_SIZE=$(du -sh ~/anaconda3/pkgs 2>/dev/null | awk '{print $1}')
if [ -n "$CACHE_SIZE" ]; then
    echo "Conda package cache: $CACHE_SIZE"
else
    echo "Conda cache not found or empty"
fi
echo ""

read -p "Do you want to clean conda cache and unused packages? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Cleaning conda cache..."
    
    # Remove package tarballs
    echo "  - Removing package tarballs..."
    conda clean --tarballs -y
    
    # Remove package cache
    echo "  - Removing package cache..."
    conda clean --packages -y
    
    # Remove index cache
    echo "  - Removing index cache..."
    conda clean --index-cache -y
    
    # Remove unused packages
    echo "  - Removing unused packages..."
    conda clean --all -y
    
    echo ""
    echo "Cleanup complete!"
    echo ""
    
    # Check available space after cleanup
    echo "Disk usage after cleanup:"
    df -h . | awk 'NR==2 {print "Available: "$4" / "$2" ("$5" used)"}'
    echo ""
    
    echo "Space freed successfully!"
else
    echo "Cleanup cancelled."
fi

echo ""
echo "Additional space-saving tips:"
echo "  1. Remove old conda environments:"
echo "     conda env list"
echo "     conda env remove -n <env-name>"
echo ""
echo "  2. Check for large files in your workspace:"
echo "     du -h --max-depth=1 . | sort -hr | head -20"
echo ""
echo "  3. Remove pip cache:"
echo "     pip cache purge"
echo ""
echo "======================================================================"
