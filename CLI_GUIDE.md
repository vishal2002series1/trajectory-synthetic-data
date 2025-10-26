# Trajectory Synthetic Data Kit - CLI Guide

## 🚀 Quick Start

The CLI provides unified commands for the entire synthetic data generation pipeline.

---

## 📋 **Available Commands**

### **1. Ingest PDFs into Vector Database**

#### Single PDF:
```bash
python main.py ingest data/pdfs/sample.pdf
```

#### Batch Ingestion:
```bash
# Ingest all PDFs in directory
python main.py ingest-batch data/pdfs/

# Recursive search
python main.py ingest-batch data/pdfs/ --recursive

# Parallel processing (4 workers)
python main.py ingest-batch data/pdfs/ --parallel 4

# Skip vision analysis (faster)
python main.py ingest-batch data/pdfs/ --skip-vision
```

---

### **2. Transform Queries**

#### Persona Transformation (×5):
```bash
python main.py transform persona "What is my portfolio allocation?"
```

Output:
```
P1 (First-time Investor): "Can you show me how my money is divided up?"
P2 (Experienced Professional): "Provide current portfolio allocation breakdown"
P3 (Technical Analyst): "Query asset allocation with sector exposure analysis"
P4 (Anxious/Uncertain): "I'm not sure... could you maybe show me my investments?"
P5 (Directive/Assertive): "Show me my portfolio allocation immediately."
```

#### Query Complexity Modification (×3):
```bash
python main.py transform query "What is diversification?"
```

Output:
```
Q- (Simplified): "What does diversification mean?"
Q  (Original):   "What is diversification?"
Q+ (Complex):    "Explain diversification as a risk mitigation strategy..."
```

#### Tool Data Transformation (×2):
```bash
python main.py transform tool "What is my allocation?" \
  --tools search_knowledge_base get_allocation \
  --answer "Your portfolio is 60% stocks, 30% bonds, 10% cash"
```

#### All Transformations (×30):
```bash
python main.py transform all "What is my portfolio allocation?"
```

This generates:
- 5 personas × 3 complexity levels × 2 tool data variants = **30 variations**

Output saved to: `data/output/transformations/YYYYMMDD_HHMMSS/`

---

### **3. Generate Trajectories**

#### From Seed Queries:
```bash
python main.py generate data/seed/seed_queries.json
```

#### With Limit:
```bash
python main.py generate data/seed/seed_queries.json --limit 100
```

Output format (from `config.yaml`):
```json
{
  "Q": "What is my portfolio allocation?",
  "COT": "I need to retrieve the user's current allocation...",
  "Tool Set": [
    {
      "name": "search_knowledge_base",
      "parameters": {"query": "portfolio allocation", "n_results": 3}
    }
  ],
  "Decision": "Your portfolio is currently allocated as..."
}
```

---

### **4. Complete Pipeline**

Run the entire end-to-end workflow:

```bash
# With PDF ingestion
python main.py pipeline data/seed/seed_queries.json \
  --pdf-dir data/pdfs/

# Skip ingestion (use existing ChromaDB)
python main.py pipeline data/seed/seed_queries.json --skip-ingest

# With limit
python main.py pipeline data/seed/seed_queries.json \
  --skip-ingest --limit 100
```

**Pipeline Stages:**
1. ✅ Ingest PDFs (optional)
2. ✅ Load seed queries
3. ✅ Apply all transformations (30× per seed)
4. ✅ Generate trajectories for each variation
5. ✅ Save training data

**Output:**
```
data/output/pipeline/YYYYMMDD_HHMMSS/
├── transformations.jsonl    # All transformed queries
├── training_data.jsonl       # Final training examples
└── pipeline_stats.json       # Statistics
```

---

## 🎯 **Common Workflows**

### **Workflow 1: Start from Scratch**
```bash
# Step 1: Ingest PDFs
python main.py ingest-batch data/pdfs/ --parallel 4

# Step 2: Test transformations
python main.py transform all "What is diversification?"

# Step 3: Generate training data
python main.py pipeline data/seed/seed_queries.json --skip-ingest
```

### **Workflow 2: Quick Test**
```bash
# Transform a single query to see all variations
python main.py transform all "What is my portfolio allocation?" \
  --output test_output/

# Generate trajectories for a few seed queries
python main.py generate data/seed/seed_queries.json --limit 10
```

### **Workflow 3: Full Production Run**
```bash
# Complete pipeline with 1000 training examples
python main.py pipeline data/seed/seed_queries.json \
  --skip-ingest \
  --limit 1000 \
  --output production_run/
```

---

## ⚙️ **Global Options**

```bash
# Verbose logging (DEBUG level)
python main.py --verbose ingest data/pdfs/sample.pdf

# Quiet mode (errors only)
python main.py --quiet ingest-batch data/pdfs/

# Custom config file
python main.py --config custom_config.yaml generate seeds.json
```

---

## 📊 **Output Organization**

All outputs are organized by date:

```
data/output/
├── transformations/
│   └── 20250126_143022/
│       ├── transformations.jsonl
│       └── transformations.json
├── trajectories/
│   └── 20250126_144530/
│       └── trajectories.jsonl
└── pipeline/
    └── 20250126_150000/
        ├── transformations.jsonl
        ├── training_data.jsonl
        └── pipeline_stats.json
```

---

## 🔍 **Checking Progress**

### Check ChromaDB Status:
```python
from src.core import ChromaDBManager
from src.utils import load_config

config = load_config()
manager = ChromaDBManager(
    persist_directory=config.chromadb.persist_directory,
    collection_name=config.chromadb.collection_name
)

print(f"Documents in ChromaDB: {manager.count()}")
print(f"Stats: {manager.get_stats()}")
```

### View Training Data:
```bash
# Count lines in output
wc -l data/output/pipeline/*/training_data.jsonl

# View first example
head -n 1 data/output/pipeline/*/training_data.jsonl | jq
```

---

## 🐛 **Troubleshooting**

### Error: "ChromaDB is empty"
```bash
# Solution: Ingest PDFs first
python main.py ingest data/pdfs/sample.pdf
```

### Error: "Seed file not found"
```bash
# Check path
ls data/seed/

# Use absolute path
python main.py generate /full/path/to/seeds.json
```

### Slow Performance
```bash
# Use parallel processing for batch ingestion
python main.py ingest-batch data/pdfs/ --parallel 4

# Skip vision analysis to speed up
python main.py ingest-batch data/pdfs/ --skip-vision
```

---

## 📖 **Next Steps**

1. ✅ **Phase 1 Complete**: CLI Foundation with 30× expansion
2. 🚧 **Phase 2**: Add PDF Augmentation Transformer (90× expansion)
3. 🚧 **Phase 3**: Add Multi-turn Expansion Transformer (108× expansion)

---

## 💡 **Tips**

- Always start with `--verbose` flag when debugging
- Use `transform all` with a single query first to validate
- Test with `--limit 10` before running full pipeline
- Check ChromaDB document count before generating trajectories
- Review pipeline_stats.json for detailed execution metrics

---

## 📞 **Support**

For issues or questions, check:
- `logs/trajectory_generator.log` for detailed logs
- `data/output/pipeline/*/pipeline_stats.json` for error details
- Configuration in `config/config.yaml`
