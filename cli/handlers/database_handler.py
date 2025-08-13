# cli/handlers/database_handler.py
import logging
from database.operations import db_ops
from cli.menu import MenuDisplay

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Handles database operations and queries"""
    
    def __init__(self):
        self.menu_display = MenuDisplay()
    
    async def handle_query_database(self):
        """Handle direct database queries"""
        self.menu_display.show_database_query_menu()
        choice = input("Select option (1-4): ").strip()
        
        try:
            if choice == "1":
                await self._count_documents()
            elif choice == "2":
                await self._show_recent_documents()
            elif choice == "3":
                await self._search_by_id()
            elif choice == "4":
                await self._custom_query()
            else:
                print("Invalid option.")
                
        except ValueError:
            print("Invalid input.")
        except Exception as e:
            logger.error(f"Database query error: {e}")
            print(f"Query failed: {e}")
    
    async def handle_list_contents(self):
        """Handle content listing"""
        preview_length = int(input("Preview length (default 200): ") or "200")
        
        try:
            docs = db_ops.list_all_documents(preview_length)
            
            if not docs:
                print("No documents found in database.")
                return
            
            print(f"\nFound {len(docs)} documents:")
            for doc in docs:
                print(f"ID: {doc['id']}, Preview: {doc['content_preview']}")
                
        except Exception as e:
            logger.error(f"Error listing contents: {e}")
            print(f"Failed to list contents: {e}")
    
    async def handle_database_management(self):
        """Handle database management tasks"""
        self.menu_display.show_database_management_menu()
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            await self._initialize_schema()
        elif choice == "2":
            await self._run_maintenance()
        elif choice == "3":
            print("Database backup functionality not implemented yet.")
        elif choice == "4":
            await self._show_statistics()
        else:
            print("Invalid option.")
    
    async def _count_documents(self):
        """Count total documents"""
        count = db_ops.get_document_count()
        print(f"Total documents: {count}")
    
    async def _show_recent_documents(self):
        """Show recent documents"""
        limit = int(input("Number of documents to show (default 10): ") or "10")
        docs = db_ops.get_documents_page(limit, 0)
        
        for doc in docs:
            print(f"\nID: {doc['id']}")
            print(f"Content: {doc['content'][:200]}...")
            print(f"Created: {doc['created_at']}")
    
    async def _search_by_id(self):
        """Search document by ID"""
        doc_id = int(input("Enter document ID: "))
        docs = db_ops.get_documents_by_ids([doc_id])
        
        if docs:
            doc = docs[0]
            print(f"\nID: {doc['id']}")
            print(f"Content: {doc['content']}")
            print(f"Tags: {doc.get('tags', [])}")
        else:
            print("Document not found.")
    
    async def _custom_query(self):
        """Handle custom database queries"""
        print("Custom queries not implemented yet.")
    
    async def _initialize_schema(self):
        """Initialize database schema"""
        confirm = input("This will reinitialize the database schema. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                db_ops.initialize_schema()
                print("Schema initialized.")
            except Exception as e:
                print(f"Schema initialization failed: {e}")
    
    async def _run_maintenance(self):
        """Run database maintenance"""
        try:
            await db_ops.run_maintenance()
            print("Database maintenance completed.")
        except Exception as e:
            print(f"Maintenance failed: {e}")
    
    async def _show_statistics(self):
        """Show database statistics"""
        try:
            metrics = await db_ops.get_batch_metrics()
            count = db_ops.get_document_count()
            
            print(f"\nDatabase Statistics:")
            print(f"Total documents: {count}")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Failed to get statistics: {e}")