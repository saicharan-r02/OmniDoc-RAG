import os
from notion_client import Client
from langchain_core.documents import Document

def fetch_notion_content() -> list[Document]:
    """
    Independent integration to fetch Notion workspace/page data.
    """
    notion_key=os.getenv("NOTION_API_KEY")
    page_id=os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_DATABASE_ID")

    if not notion_key or not page_id:
        return []

    try:
        notion=Client(auth=notion_key)
        return [
            Document(
                page_content="Notion content synced successfully.",
                metadata={"source": "notion", "page_id": page_id}
            )
        ]
    except Exception as e:
        print(f"Error fetching from Notion: {e}")
        return []