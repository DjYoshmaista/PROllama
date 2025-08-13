# cli/handlers/inference_handler.py
import logging
from core.config import config
from core.memory import memory_manager
from inference.engine import inference_engine

logger = logging.getLogger(__name__)

class InferenceHandler:
    """Handles inference/question-answering operations"""
    
    async def handle_inference(self):
        """Handle inference/question-answering"""
        question = input("Enter your question: ").strip()
        if not question:
            print("Question cannot be empty!")
            return
        
        # Check if user wants to configure search parameters
        print("\nSearch Configuration:")
        print("Press Enter to use defaults, or type 'config' to customize")
        search_config = input("Choice: ").strip().lower()
        
        kwargs = {}
        if search_config == 'config':
            kwargs = self._get_search_configuration()
        
        # Ask about cache usage
        cache_choice = input("Use embedding cache if available? (Y/n): ").strip().lower()
        if cache_choice == 'n':
            kwargs['use_cache'] = False
        
        print("\nProcessing your question...")
        
        try:
            with memory_manager:
                result = await inference_engine.ask_question(question, **kwargs)
            
            self._display_inference_results(result)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            print(f"Failed to process question: {e}")
    
    def _get_search_configuration(self) -> dict:
        """Get custom search configuration from user"""
        kwargs = {}
        
        try:
            top_k = input(f"Top K results (default: {config.inference.top_k}): ").strip()
            if top_k:
                kwargs['top_k'] = int(top_k)
        except ValueError:
            print("Invalid top_k value, using default")
        
        try:
            threshold = input(f"Relevance threshold (default: {config.inference.relevance_threshold}): ").strip()
            if threshold:
                kwargs['relevance_threshold'] = float(threshold)
        except ValueError:
            print("Invalid threshold value, using default")
        
        return kwargs
    
    def _display_inference_results(self, result: dict):
        """Display inference results in a formatted way"""
        # Display answer
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        
        # Display sources if available
        if result['sources']:
            print("\n" + "="*60)
            print("SOURCES:")
            print("="*60)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n[Source {i}] (Similarity: {source['similarity']:.4f})")
                print(f"ID: {source['id']}")
                print(f"Preview: {source['content_preview']}")
                if source.get('tags'):
                    print(f"Tags: {', '.join(source['tags'][:5])}")
        
        # Show metadata
        metadata = result['metadata']
        print(f"\nMatches found: {metadata['matches_found']}")