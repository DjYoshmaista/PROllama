# cli/menu.py
class MenuDisplay:
    """Handles all menu display operations"""
    
    def show_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("RAG Database System")
        print("="*60)
        print("1. Ask the inference model for information")
        print("2. Add data to the database")
        print("3. Load data from a single file")
        print("4. Load documents from folder")
        print("5. Query the database directly")
        print("6. List database contents")
        print("7. Configure system parameters")
        print("8. Database management")
        print("9. System information")
        print("0. Exit")
    
    def show_database_query_menu(self):
        """Display database query menu"""
        print("\nDatabase Query Options:")
        print("1. Count total documents")
        print("2. Show recent documents")
        print("3. Search by ID")
        print("4. Custom query")
    
    def show_configuration_menu(self):
        """Display configuration menu"""
        print("\nConfiguration Options:")
        print("1. Inference parameters")
        print("2. Database optimization")
        print("3. Cache management")
        print("4. View current configuration")
    
    def show_database_management_menu(self):
        """Display database management menu"""
        print("\nDatabase Management:")
        print("1. Initialize/Reset schema")
        print("2. Run maintenance")
        print("3. Backup (not implemented)")
        print("4. Statistics")
    
    def show_cache_management_menu(self):
        """Display cache management menu"""
        print("\nCache Management:")
        print("1. Load cache")
        print("2. Invalidate cache")
        print("3. Cache information")
        print("4. Refresh cache")
    
    def show_database_configuration_menu(self):
        """Display database configuration menu"""
        print("\nDatabase Configuration:")
        print("1. Optimize PostgreSQL settings")
        print("2. Run database maintenance")
        print("3. View database statistics")