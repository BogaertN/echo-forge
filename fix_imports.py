import re

def fix_main_py():
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Replace problematic imports with try/except blocks
    imports_to_fix = [
        ('import soundfile as sf', 'try:\n    import soundfile as sf\nexcept ImportError:\n    print("soundfile not available")\n    sf = None'),
        ('import librosa', 'try:\n    import librosa\nexcept ImportError:\n    print("librosa not available")\n    librosa = None'),
    ]
    
    for old_import, new_import in imports_to_fix:
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"✅ Fixed {old_import}")
    
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("✅ All imports fixed!")

fix_main_py()
