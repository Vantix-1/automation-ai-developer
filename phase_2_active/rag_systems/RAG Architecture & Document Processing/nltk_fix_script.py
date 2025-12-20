"""
🔧 NLTK Data Setup Script
Fixes the punkt_tab missing resource error
Run this once before using text_splitting.py
"""

import nltk
import sys

def download_nltk_data():
    """Download all required NLTK data packages"""
    
    print("📦 Downloading NLTK Data Packages...")
    print("=" * 60)
    
    # List of required packages
    packages = [
        ('punkt', 'Punkt Tokenizer Models'),
        ('punkt_tab', 'Punkt Tokenizer Tables (New)'),
        ('averaged_perceptron_tagger', 'POS Tagger'),
        ('maxent_ne_chunker', 'Named Entity Chunker'),
        ('words', 'Word Lists'),
        ('stopwords', 'Stopwords'),
    ]
    
    successful = []
    failed = []
    
    for package, description in packages:
        try:
            print(f"\n📥 Downloading {description}...")
            nltk.download(package, quiet=False)
            successful.append(package)
            print(f"   ✅ {package} downloaded successfully")
        except Exception as e:
            failed.append((package, str(e)))
            print(f"   ⚠️ Failed to download {package}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"\n✅ Successful: {len(successful)}/{len(packages)}")
    for pkg in successful:
        print(f"   • {pkg}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(packages)}")
        for pkg, error in failed:
            print(f"   • {pkg}: {error}")
    
    print("\n" + "=" * 60)
    
    if len(successful) == len(packages):
        print("🎉 All NLTK packages downloaded successfully!")
        print("✅ You can now run text_splitting.py without errors")
    else:
        print("⚠️ Some packages failed to download")
        print("💡 Try running this script again or download manually:")
        print("   >>> import nltk")
        print("   >>> nltk.download('punkt_tab')")
    
    return len(failed) == 0

def verify_installation():
    """Verify NLTK data is properly installed"""
    
    print("\n🔍 VERIFYING INSTALLATION")
    print("=" * 60)
    
    # Test punkt_tab
    try:
        from nltk.tokenize import sent_tokenize
        test_text = "This is a test. Does it work? Yes, it does!"
        sentences = sent_tokenize(test_text)
        print(f"\n✅ punkt_tab working correctly")
        print(f"   Test: '{test_text}'")
        print(f"   Result: {len(sentences)} sentences detected")
        return True
    except LookupError as e:
        print(f"\n❌ punkt_tab still not found: {e}")
        print("\n💡 Manual fix:")
        print("   1. Open Python interpreter")
        print("   2. Run: import nltk")
        print("   3. Run: nltk.download('punkt_tab')")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def show_nltk_info():
    """Show NLTK data directory information"""
    
    print("\n📂 NLTK DATA DIRECTORIES")
    print("=" * 60)
    
    import nltk.data
    
    print("\nNLTK will search for data in these locations:")
    for i, path in enumerate(nltk.data.path, 1):
        print(f"   {i}. {path}")
    
    print("\n💡 If downloads fail, you can manually place files in any of these directories")

def main():
    """Main setup function"""
    
    print("🚀 NLTK Data Setup for Text Splitting")
    print("Fixing: Resource punkt_tab not found error")
    print("=" * 60)
    
    # Show NLTK info
    show_nltk_info()
    
    # Download packages
    input("\n Press Enter to start downloading NLTK data...")
    success = download_nltk_data()
    
    # Verify
    if success:
        verify_installation()
    
    print("\n" + "=" * 60)
    print("Setup 