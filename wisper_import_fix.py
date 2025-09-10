 Quick patch to fix whisper import in main.py
# Run this script in your echo-forge directory

import re

def fix_whisper_import():
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Replace the problematic whisper import with a try/except block
        old_import = "import whisper"
        new_import = """try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: Whisper not available, voice features disabled")
    WHISPER_AVAILABLE = False
    whisper = None"""
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            # Also need to protect whisper usage
            content = content.replace(
                "whisper.load_model",
                "whisper.load_model if WHISPER_AVAILABLE else None"
            )
            
            with open('main.py', 'w') as f:
                f.write(content)
            
            print("✅ Fixed whisper import in main.py")
        else:
            print("ℹ️  Whisper import not found or already fixed")
            
    except Exception as e:
        print(f"❌ Error fixing whisper import: {e}")

if __name__ == "__main__":
    fix_whisper_import()
