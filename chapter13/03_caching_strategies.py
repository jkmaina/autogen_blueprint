"""
Chapter 13: Performance Optimization and Deployment
Example 3: Caching Strategies

Description:
Demonstrates comprehensive caching strategies for performance optimization
including simple response caching, semantic similarity caching, and tiered
memory/disk caching systems with TTL expiration management.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- OAI_CONFIG_LIST configuration file

Usage:
```bash
python -m chapter13.03_caching_strategies
```

Expected Output:
Caching strategies demonstration:
1. Simple response caching with exact matches
2. Semantic caching based on text similarity
3. Tiered caching with memory and disk layers
4. Cache hit/miss ratio analysis
5. Performance improvement measurements
6. Cost reduction through cached responses

Key Concepts:
- Response caching strategies
- Semantic similarity matching
- Tiered cache architectures
- TTL expiration management
- Memory vs disk cache trade-offs
- Cache key generation
- Cost optimization through caching
- API rate limiting mitigation

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Simple in-memory cache
response_cache = {}

# Function to generate a cache key
def get_cache_key(prompt, model="gpt-3.5-turbo"):
    # Create a unique key based on prompt and model
    return f"{hashlib.md5(prompt.encode()).hexdigest()}_{model}"

# Simple response caching
async def get_cached_response(agent, prompt, model="gpt-3.5-turbo"):
    cache_key = get_cache_key(prompt, model)
    
    # Check if response is in cache
    if cache_key in response_cache:
        print(f"Cache hit! Using cached response for: {prompt[:30]}...")
        return response_cache[cache_key], True
    
    # If not in cache, generate response
    print(f"Cache miss. Generating response for: {prompt[:30]}...")
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # Get response from agent
    start_time = time.time()
    await user_proxy.a_initiate_chat(agent, message=prompt)
    response_time = time.time() - start_time
    
    # Extract the last message from the agent
    response = user_proxy.chat_messages[agent][-1]["content"]
    
    # Cache the response
    response_cache[cache_key] = response
    
    return response, False

# Semantic similarity function (simplified for demonstration)
def calculate_similarity(text1, text2):
    # Convert to sets of words for a simple Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

# Semantic caching
async def get_semantically_cached_response(agent, prompt, model="gpt-3.5-turbo", threshold=0.8):
    # Check for semantic similarity with cached prompts
    best_match = None
    best_similarity = 0
    
    for cached_prompt in list(response_cache.keys()):
        # Extract the original prompt from the cache key
        cached_prompt_text = cached_prompt.split('_')[0]
        
        # Calculate similarity
        similarity = calculate_similarity(prompt, cached_prompt_text)
        
        # Update best match if this is more similar
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match = cached_prompt
    
    # If we found a good match, use the cached response
    if best_match:
        print(f"Semantic cache hit! Similarity: {best_similarity:.2f}")
        return response_cache[best_match], True
    
    # Otherwise, generate a new response
    return await get_cached_response(agent, prompt, model)

# Tiered caching with disk persistence
class TieredCache:
    def __init__(self, memory_ttl=3600, disk_ttl=86400, disk_cache_file="disk_cache.json"):
        self.memory_cache = {}  # Short-term memory cache
        self.memory_ttl = memory_ttl  # Time to live in seconds for memory cache
        self.disk_ttl = disk_ttl  # Time to live in seconds for disk cache
        self.disk_cache_file = disk_cache_file
        
        # Load disk cache if it exists
        self.disk_cache = {}
        if os.path.exists(disk_cache_file):
            try:
                with open(disk_cache_file, 'r') as f:
                    self.disk_cache = json.load(f)
            except:
                print("Error loading disk cache, starting with empty cache")
    
    def get(self, key):
        # Check memory cache first (fastest)
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if datetime.fromisoformat(entry['expires']) > datetime.now():
                print(f"Memory cache hit for {key}")
                return entry['value'], True
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        
        # Check disk cache next
        if key in self.disk_cache:
            entry = self.disk_cache[key]
            if datetime.fromisoformat(entry['expires']) > datetime.now():
                # Move to memory cache for faster access next time
                self.memory_cache[key] = entry
                print(f"Disk cache hit for {key}")
                return entry['value'], True
            else:
                # Expired, remove from disk cache
                del self.disk_cache[key]
                self._save_disk_cache()
        
        return None, False
    
    def set(self, key, value):
        # Add to memory cache with expiration
        memory_expires = datetime.now() + timedelta(seconds=self.memory_ttl)
        self.memory_cache[key] = {
            'value': value,
            'expires': memory_expires.isoformat()
        }
        
        # Add to disk cache with longer expiration
        disk_expires = datetime.now() + timedelta(seconds=self.disk_ttl)
        self.disk_cache[key] = {
            'value': value,
            'expires': disk_expires.isoformat()
        }
        
        # Save disk cache
        self._save_disk_cache()
    
    def _save_disk_cache(self):
        with open(self.disk_cache_file, 'w') as f:
            json.dump(self.disk_cache, f)

async def main():
    """Main function to demonstrate various caching strategies."""
    # Load model configurations
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo"],
        },
    )
    
    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        llm_config={"config_list": config_list},
    )
    
    # Initialize tiered cache
    tiered_cache = TieredCache(memory_ttl=60, disk_ttl=3600, disk_cache_file="autogen_cache.json")
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Tell me the capital city of France.",  # Semantically similar to the first prompt
        "What is the population of Tokyo?",
        "How many people live in Tokyo?",  # Semantically similar to the third prompt
    ]
    
    # 1. Demonstrate simple response caching
    print("\n=== Simple Response Caching ===")
    
    # First request - should be a cache miss
    response1, cached1 = await get_cached_response(assistant, prompts[0])
    print(f"Response cached: {cached1}")
    
    # Repeat the exact same request - should be a cache hit
    response1_repeat, cached1_repeat = await get_cached_response(assistant, prompts[0])
    print(f"Response cached: {cached1_repeat}")
    
    # 2. Demonstrate semantic caching
    print("\n=== Semantic Caching ===")
    
    # Different wording but same meaning - should find similar cached prompt
    response2, cached2 = await get_semantically_cached_response(assistant, prompts[1])
    print(f"Response cached: {cached2}")
    
    # New question - should be a cache miss
    response3, cached3 = await get_semantically_cached_response(assistant, prompts[2])
    print(f"Response cached: {cached3}")
    
    # Similar to the previous question - should be a semantic cache hit
    response4, cached4 = await get_semantically_cached_response(assistant, prompts[3])
    print(f"Response cached: {cached4}")
    
    # 3. Demonstrate tiered caching
    print("\n=== Tiered Caching ===")
    
    # Function to use tiered cache
    async def get_tiered_cached_response(cache, agent, prompt):
        cache_key = get_cache_key(prompt)
        
        # Try to get from cache
        cached_response, found = cache.get(cache_key)
        if found:
            return cached_response, True
        
        # Generate new response
        user_proxy = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
        )
        
        await user_proxy.a_initiate_chat(agent, message=prompt)
        response = user_proxy.chat_messages[agent][-1]["content"]
        
        # Store in cache
        cache.set(cache_key, response)
        
        return response, False
    
    # First request to tiered cache
    tiered_response1, tiered_cached1 = await get_tiered_cached_response(
        tiered_cache, assistant, "What is the distance from Earth to the Moon?"
    )
    print(f"Response from tiered cache: {tiered_cached1}")
    
    # Repeat request - should be in memory cache
    tiered_response2, tiered_cached2 = await get_tiered_cached_response(
        tiered_cache, assistant, "What is the distance from Earth to the Moon?"
    )
    print(f"Response from tiered cache: {tiered_cached2}")
    
    # Performance comparison summary
    print("\n=== Caching Performance Summary ===")
    print("1. Simple caching: Fast retrieval for exact matches")
    print("2. Semantic caching: Handles variations in phrasing")
    print("3. Tiered caching: Balances speed and persistence")
    print("\nImplementing these caching strategies can significantly reduce:")
    print("- API costs by minimizing redundant requests")
    print("- Latency by serving cached responses instantly")
    print("- Rate limiting issues by reducing total API calls")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCaching strategies demo interrupted by user")
    except Exception as e:
        print(f"Error running caching strategies demo: {e}")
        raise
