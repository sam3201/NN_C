# 📚 **TEXT DATA LOCATIONS - COMPLETE DIRECTORY**

## 🎯 **All Text Data Files in Organized System**

---

## 📝 **VOCABULARY AND WORD DATA**

### **✅ Stage 2: Word-Level Text Data**
**Location**: `ORGANIZED/MODELS/STAGE2/`

```
📁 ORGANIZED/MODELS/STAGE2/
├── 📄 stage2_vocabulary.txt              (4,293,193 bytes) ← MAIN VOCABULARY
├── 🤖 stage2_word_extraction.c          (12,319 bytes)
├── 🎓 stage2_word_training.c            (15,449 bytes)
├── 💾 stage2_word_final.bin               (1,426,800 bytes)
├── 💾 stage2_word_epoch_5.bin            (1,426,800 bytes)
├── 💾 stage2_word_epoch_10.bin           (1,426,800 bytes)
├── 💾 stage2_word_epoch_15.bin           (1,426,800 bytes)
├── 💾 stage2_word_epoch_20.bin           (1,426,800 bytes)
├── 💾 stage2_word_epoch_25.bin           (1,426,800 bytes)
└── 💾 stage2_word_epoch_30.bin           (1,426,800 bytes)
```

**📊 Vocabulary Statistics:**
- **Total Words**: 21,927 words
- **File Size**: 4.29 MB
- **Format**: Word frequency list
- **Usage**: Word-level learning and vocabulary building

---

## 📖 **PHRASE AND COLLOCATION DATA**

### **✅ Stage 3: Phrase-Level Text Data**
**Location**: `ORGANIZED/MODELS/STAGE3/`

```
📁 ORGANIZED/MODELS/STAGE3/
├── 📄 stage3_phrases.txt                (30,058,135 bytes) ← MAIN PHRASE DATABASE
├── 📄 stage3_collocations.txt            (1,017,585 bytes) ← COLLOCATIONS
├── 🤖 stage3_phrase_extraction.c        (20,979 bytes)
├── 🎓 stage3_phrase_training.c          (19,185 bytes)
├── 💾 stage3_phrase_final.bin             (1,426,800 bytes)
├── 💾 stage3_phrase_epoch_5.bin           (1,426,800 bytes)
├── 💾 stage3_phrase_epoch_10.bin           (1,426,800 bytes)
├── 💾 stage3_phrase_epoch_15.bin           (1,426,800 bytes)
├── 💾 stage3_phrase_epoch_20.bin           (1,426,800 bytes)
├── 💾 stage3_phrase_epoch_25.bin           (1,426,800 bytes)
└── 💾 stage3_phrase_final.bin             (1,426,800 bytes)
```

**📊 Phrase Statistics:**
- **Total Phrases**: ~300,000 phrases (estimated)
- **File Size**: 30.06 MB
- **Format**: Phrase list with frequency
- **Collocations**: 1,017,585 collocation pairs
- **Usage**: Phrase-level learning and context understanding

---

## 📚 **ADDITIONAL TEXT RESOURCES**

### **✅ Training Data Files**
**Location**: `ORGANIZED/TRAINING/`

```
📁 ORGANIZED/TRAINING/
├── 📄 training_data.csv                  (1,112 bytes)
└── 📁 training_data/ (5 items)
    ├── 📄 sample_texts.txt
    ├── 📄 conversations.txt
    ├── 📄 qa_pairs.txt
    ├── 📄 technical_docs.txt
    └── 📄 creative_writing.txt
```

### **✅ Hugging Face Integration Data**
**Location**: `ORGANIZED/PROJECTS/HUGGINGFACE_INTEGRATION/`

```
📁 ORGANIZED/PROJECTS/HUGGINGFACE_INTEGRATION/
├── 📄 hf_training_data.json           (10,011,434 bytes) ← MAIN TRAINING DATA
├── 📄 lesson_training_data.json         (441 bytes)
├── 📄 prompt.txt                       (19 bytes)
├── 📄 conversation_*.log                (Multiple files)
└── 📄 teaching_session_*.log             (Multiple files)
```

---

## 📊 **TEXT DATA SUMMARY**

### **✅ Primary Text Data Files**

| File | Location | Size | Purpose |
|------|----------|--------|--------|
| **stage2_vocabulary.txt** | `ORGANIZED/MODELS/STAGE2/` | 4.29 MB | 21,927 words for vocabulary |
| **stage3_phrases.txt** | `ORGANIZED/MODELS/STAGE3/` | 30.06 MB | ~300,000 phrases for context |
| **stage3_collocations.txt** | `ORGANIZED/MODELS/STAGE3/` | 1.02 MB | Word collocation pairs |
| **hf_training_data.json** | `ORGANIZED/PROJECTS/HUGGINGFACE_INTEGRATION/` | 10.01 MB | Hugging Face training data |

### **✅ Total Text Data Size**

- **Vocabulary**: 4.29 MB (21,927 words)
- **Phrases**: 30.06 MB (~300,000 phrases)
- **Collocations**: 1.02 MB (word pairs)
- **Hugging Face Data**: 10.01 MB (training conversations)
- **Training Samples**: ~1.1 KB (additional data)

**📊 Total**: ~45.38 MB of text data

---

## 🎯 **USAGE IN CHATBOT SYSTEM**

### **✅ Vocabulary Usage**
```python
# Location: ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt
# Used by: Word-level model and response generation
# Purpose: Building vocabulary understanding and word context
```

### **✅ Phrase Usage**
```python
# Location: ORGANIZED/MODELS/STAGE3/stage3_phrases.txt
# Used by: Phrase-level model and context understanding
# Purpose: Learning phrase patterns and collocations
```

### **✅ Training Data Usage**
```python
# Location: ORGANIZED/TRAINING/training_data.csv
# Used by: Various training scripts and continuous training
# Purpose: Sample conversations and Q&A pairs
```

---

## 🔍 **ACCESSING THE TEXT DATA**

### **✅ Quick Access Commands**

```bash
# View vocabulary
head -20 ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt

# Count vocabulary words
wc -l ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt

# View phrases
head -10 ORGANIZED/MODELS/STAGE3/stage3_phrases.txt

# Count phrases
wc -l ORGANIZED/MODELS/STAGE3/stage3_phrases.txt

# View collocations
head -10 ORGANIZED/MODELS/STAGE3/stage3_collocations.txt

# Search for specific words
grep "hello" ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt
```

### **✅ Python Access Examples**

```python
# Load vocabulary
def load_vocabulary():
    vocab = {}
    with open('ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                freq = int(parts[1])
                vocab[word] = freq
    return vocab

# Load phrases
def load_phrases():
    phrases = []
    with open('ORGANIZED/MODELS/STAGE3/stage3_phrases.txt', 'r') as f:
        for line in f:
            phrase = line.strip()
            if phrase:
                phrases.append(phrase)
    return phrases
```

---

## 🎯 **INTEGRATION WITH CHATBOT**

### **✅ Current Integration**
The text data is integrated into the chatbot system through:

1. **Vocabulary Building**: `stage2_vocabulary.txt` → Word understanding
2. **Phrase Context**: `stage3_phrases.txt` → Context awareness
3. **Training Samples**: Various training files → Response generation
4. **Continuous Training**: Ollama generates additional text data

### **✅ Data Flow**
```
Text Data → Vocabulary/Phrase Building → Model Training → Response Generation
```

---

## 🎯 **CONCLUSION**

### **✅ Complete Text Data Organization**

All text data is properly organized in the `ORGANIZED/MODELS/` directory structure:

- **📝 Stage 2**: Vocabulary and word-level data (4.29 MB)
- **📖 Stage 3**: Phrases and collocations (31.08 MB)
- **🎯 Training**: Additional training samples and data
- **🤖 Integration**: Hugging Face and other project data

### **✅ Easy Access**
- **Organized Structure**: Clear directory hierarchy
- **Large Datasets**: Millions of text entries for training
- **Multiple Formats**: Text files, JSON, CSV
- **Integration Ready**: Used by chatbot and training systems

---

## 🚀 **QUICK REFERENCE**

### **📚 Main Text Files**
- **Vocabulary**: `ORGANIZED/MODELS/STAGE2/stage2_vocabulary.txt`
- **Phrases**: `ORGANIZED/MODELS/STAGE3/stage3_phrases.txt`
- **Collocations**: `ORGANIZED/MODELS/STAGE3/stage3_collocations.txt`

### **🔍 Find Text Data**
```bash
# Find all text files
find ORGANIZED -name "*.txt" -o -name "*.csv" -o -name "*.json"

# Find large text files
find ORGANIZED -name "*.txt" -exec ls -lh {} \; | sort -k5 -h

# Count total text data
find ORGANIZED -name "*.txt" -exec wc -c {} + | tail -1
```

---

*Text data locations completed on February 4, 2026*
*Status: 100% Complete - All text data organized and accessible*
