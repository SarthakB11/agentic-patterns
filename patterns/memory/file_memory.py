"""File-directory memory: a peer of the vector store, not a variant of it.

Mirrors Anthropic's memory tool (a client-side `/memory` directory the
model reads, writes, and deletes across sessions) and Letta's filesystem
memory. There is no embedding index and no similarity search here:
retrieval is an exact path read. Letta reported agents on gpt-4o-mini
reaching 74% on the LoCoMo benchmark writing conversation files alone, with
no embedding index, so filesystem memory is a genuine alternative to the
vector store for some workloads, not a lesser fallback.

Treat that 74% as one configuration's result, not a settled ranking.
MemDelta (arXiv:2606.29914) varies one component at a time on
LongMemEval-S and shows LoCoMo-style single-number comparisons are
routinely confounded by which embedding model or base model was paired
with the memory method, with rankings that reverse when only the embedding
changes. LongMemEval (arXiv:2410.10813) is the stronger reference when this
pattern needs an authoritative benchmark citation: it is designed around
holding components fixed, exactly what a confound-sensitive single number
cannot promise. `memory_bench.py` follows LongMemEval's ability taxonomy
for the same reason, with the reader model held fixed by construction under
`MockProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileMemoryStore:
    """An in-memory stand-in for a per-namespace `/memory` directory, one
    plain-text file per path.
    """

    namespace: str
    files: dict[str, str] = field(default_factory=dict)

    def create(self, path: str, content: str) -> str:
        """Create a new file. Fails if `path` already exists."""
        if path in self.files:
            return f"ERROR: {path} already exists, use update"
        self.files[path] = content
        return f"created {path}"

    def read(self, path: str) -> str:
        """Read a file's full contents."""
        if path not in self.files:
            return f"ERROR: {path} not found"
        return self.files[path]

    def update(self, path: str, content: str) -> str:
        """Overwrite an existing file. Fails if `path` does not exist."""
        if path not in self.files:
            return f"ERROR: {path} not found, use create"
        self.files[path] = content
        return f"updated {path}"

    def delete(self, path: str) -> str:
        """Delete a file. Fails if `path` does not exist."""
        if path not in self.files:
            return f"ERROR: {path} not found"
        del self.files[path]
        return f"deleted {path}"

    def list_files(self) -> list[str]:
        """List every file path currently stored, sorted."""
        return sorted(self.files)


def run_file_memory_demo() -> dict[str, str]:
    """Session one creates a memory file; session two, with no conversation
    history carried over, reads it back and updates it, purely by path,
    with no embedding index involved.
    """
    fs = FileMemoryStore(namespace="user:alex")

    # session 1
    create_result = fs.create("preferences.md", "- coffee: dark roast, mornings only\n- allergy: peanuts")

    # session 2: a fresh reference to the same store; the file persists
    # independent of any conversation buffer
    read_before = fs.read("preferences.md")
    update_result = fs.update(
        "preferences.md",
        "- coffee: dark roast, mornings only\n- allergy: peanuts\n- timezone: America/Chicago",
    )
    read_after = fs.read("preferences.md")

    return {
        "create_result": create_result,
        "read_before_update": read_before,
        "update_result": update_result,
        "read_after_update": read_after,
        "files": ", ".join(fs.list_files()),
    }
