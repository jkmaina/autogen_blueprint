"""
Chapter 6: Web Integration and URL Processing
Example 1: URL Content Summarization

Description:
Demonstrates web content extraction and processing for summarization tasks.
Shows how to fetch web pages, parse HTML content, convert to readable text,
and prepare content for AI-powered summarization with marketing focus.

Prerequisites:
- httpx package for HTTP requests
- beautifulsoup4 for HTML parsing
- html2text for content conversion
- Internet connectivity for web fetching

Usage:
```python
from chapter6.01_summarize_url import summarize_url
content = await summarize_url("https://example.com")
print(content)
```

Expected Output:
Processed web content ready for summarization:
1. Fetches HTML content from specified URL
2. Parses HTML structure with BeautifulSoup
3. Converts to clean readable text format
4. Truncates to 5000 characters for processing
5. Returns formatted prompt for marketing-focused summarization

Key Concepts:
- Web content extraction
- HTML parsing and processing
- Text format conversion
- Content truncation strategies
- Marketing-focused summarization
- Error handling for web requests
- Timeout management for HTTP calls

AutoGen Version: 0.5+
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import httpx
from bs4 import BeautifulSoup
import html2text


async def summarize_url(url: str) -> str:
    """
    Fetch and process web content for summarization.
    
    Args:
        url: The URL to fetch and process
        
    Returns:
        Processed content ready for AI summarization or error message
    """
    try:
        # Fetch web content with timeout
        response = httpx.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Convert to clean text
        text = html2text.html2text(soup.get_text())
        
        # Prepare marketing-focused summarization prompt
        summary_prompt = f"Summarize the following content for a marketer:\n\n{text[:5000]}"
        
        return summary_prompt
        
    except httpx.TimeoutException:
        return f"Error: Request timeout when fetching URL: {url}"
    except httpx.HTTPError as e:
        return f"Error: HTTP error when fetching URL {url}: {str(e)}"
    except Exception as e:
        return f"Error fetching or summarizing URL {url}: {str(e)}"


async def main():
    """Example usage of URL summarization function."""
    # Example URL for testing
    test_url = "https://httpbin.org/html"
    
    print("=== URL Content Summarization ===")
    print(f"Processing URL: {test_url}")
    
    result = await summarize_url(test_url)
    print(f"Result:\n{result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())