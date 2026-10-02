import os
from notion_client import Client

def load_notion_pages():
    """
    Fetches text content from the authorized Notion workspace/database (Aegis integration).
    Runs independently of Groq and Ollama.
    """
    api_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")

    if not api_key or not database_id:
        return []

    notion = Client(auth=api_key)
    response = notion.databases.query(database_id=database_id)
    
    docs = []
    for page in response.get("results", []):
        properties = page.get("properties", {})
        title_data = properties.get("Name", {}).get("title", [])
        title_text = title_data[0]["plain_text"] if title_data else "Untitled Page"
        
        docs.append({
            "content": f"Title: {title_text}",
            "metadata": {"source": "notion", "page_id": page.get("id")}
        })
    return docs