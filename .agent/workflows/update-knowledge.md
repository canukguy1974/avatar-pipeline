---
description: How to update the project knowledge base (KNOWLEDGE.md)
---

# Update Knowledge Workflow

Use this workflow whenever you finish a task that modifies the architecture, adds a new service, or fixes a critical "gotcha" bug.

## Steps

1. **Identify Changes**:
   Run the knowledge helper script to see which sections might be affected.
   ```bash
   python tools/knowledge_helper.py
   ```

2. **Review KNOWLEDGE.md**:
   Open [KNOWLEDGE.md](file:///wsl.localhost/Ubuntu/home/canuk/projects/inifinitetalk-local/KNOWLEDGE.md) and locate the suggested sections.

3. **Apply Updates**:
   Update the relevant sections with:
   - Technical details (payloads, file names, ports).
   - Architectural diagrams (if using Mermaid).
   - "Why" the change was made (rationale).

4. **Add Gotchas**:
   If you fixed a bug caused by configuration or environment issues, add it to the **⚠️ Known Gotchas & Patterns** section to prevent future occurrences.

5. **Verify**:
   Read through the updated document to ensure it accurately reflects the current state of the codebase.
