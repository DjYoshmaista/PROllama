# cli/handlers/data_handler.py
import logging
from database.operations import db_ops
from database.cache import embedding_cache
from inference.embeddings import embedding_service

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data addition and management operations"""
    
    async def handle_add_data(self):
        """Handle manual data addition"""
        content = input("Enter the content to add: ").strip()
        if not content:
            print("Content cannot be empty!")
            return
        
        # Get optional tags
        tags_input = input("Enter tags (comma-separated, optional): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
        
        try:
            print("Generating embedding...")
            embedding = await embedding_service.generate_embedding(content)
            
            if embedding:
                success = db_ops.insert_document(content, tags, embedding)
                if success:
                    print("Data added successfully.")
                    # Invalidate cache
                    embedding_cache.invalidate()
                else:
                    print("Failed to add data to database.")
            else:
                print("Failed to generate embedding.")
                
        except Exception as e:
            logger.error(f"Error adding data: {e}")
            print(f"Failed to add data: {e}")
    
    def get_user_tags(self) -> list:
        """Get tags from user input"""
        tags_input = input("Enter tags (comma-separated, optional): ").strip()
        return [tag.strip() for tag in tags_input.split(",")] if tags_input else []
    
    def validate_content(self, content: str) -> bool:
        """Validate content input"""
        if not content or not content.strip():
            print("Content cannot be empty!")
            return False
        return True