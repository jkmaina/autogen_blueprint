# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("research_assistant")

# Create a function to run the research with error handling
async def run_research_with_error_handling(topic: str):
    try:
        logger.info(f"Starting research on topic: {topic}")
        stream = research_team.run_stream(task=f"Research the topic: {topic}")
        await Console(stream)
        logger.info("Research completed successfully")
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        # Implement recovery strategy here
        print(f"An error occurred: {str(e)}. Please try again with a different topic or approach.")
    finally:
        # Clean up resources
        await model_client.close()

# Run the research assistant
async def main():
    topic = "The impact of artificial intelligence on education"
    await run_research_with_error_handling(topic)

if __name__ == "__main__":
    asyncio.run(main())        