import re
import aiofiles
import aiohttp
from typing import List
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType

class SimpleDocumentIndexer:
    """A beginner-friendly document indexer for AutoGen Memory."""
    
    def __init__(self, memory: Memory, chunk_size: int = 1000):
        self.memory = memory
        self.chunk_size = chunk_size
    
    async def _fetch_content(self, source: str) -> str:
        """Get content from a URL or local file."""
        if source.startswith(("http://", "https://")):
            # Fetch from web
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    return await response.text()
        else:
            # Read from local file
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                return await f.read()
    
    def _clean_text(self, text: str) -> str:
        """Remove HTML tags and clean up text."""
        # Remove HTML tags
        text = re.sub(r"<[^>]*>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        return chunks
    
    async def index_documents(self, sources: List[str]) -> int:
        """Index multiple documents into memory."""
        total_chunks = 0
        
        for source in sources:
            try:
                print(f"Indexing: {source}")
                
                # Fetch content
                content = await self._fetch_content(source)
                
                # Clean content if it looks like HTML
                if "<" in content and ">" in content:
                    content = self._clean_text(content)
                
                # Split into chunks
                chunks = self._split_into_chunks(content)
                
                # Add each chunk to memory
                for i, chunk in enumerate(chunks):
                    await self.memory.add(MemoryContent(
                        content=chunk,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={
                            "source": source,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
                
                total_chunks += len(chunks)
                print(f"  Added {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error indexing {source}: {e}")
        
        return total_chunks

