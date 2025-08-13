# cli/utils/input_utils.py
from typing import Optional, Union

class InputUtils:
    """Utilities for handling user input"""
    
    @staticmethod
    def get_integer_input(prompt: str, default: Optional[int] = None, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Optional[int]:
        """Get integer input from user with validation"""
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            
            if not user_input:
                return None
            
            value = int(user_input)
            
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                return None
            
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                return None
            
            return value
            
        except ValueError:
            print("Invalid integer input")
            return None
    
    @staticmethod
    def get_float_input(prompt: str, default: Optional[float] = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Optional[float]:
        """Get float input from user with validation"""
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            
            if not user_input:
                return None
            
            value = float(user_input)
            
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                return None
            
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                return None
            
            return value
            
        except ValueError:
            print("Invalid float input")
            return None
    
    @staticmethod
    def get_yes_no_input(prompt: str, default: bool = False) -> bool:
        """Get yes/no input from user"""
        default_str = "Y/n" if default else "y/N"
        full_prompt = f"{prompt} ({default_str}): "
        
        user_input = input(full_prompt).strip().lower()
        
        if not user_input:
            return default
        
        return user_input in ['y', 'yes', '1', 'true']
    
    @staticmethod
    def get_choice_input(prompt: str, choices: list, default: Optional[str] = None) -> Optional[str]:
        """Get choice input from user with validation"""
        choices_str = "/".join(choices)
        if default:
            full_prompt = f"{prompt} ({choices_str}, default: {default}): "
        else:
            full_prompt = f"{prompt} ({choices_str}): "
        
        user_input = input(full_prompt).strip().lower()
        
        if not user_input and default:
            return default
        
        if user_input in [choice.lower() for choice in choices]:
            return user_input
        
        print(f"Invalid choice. Please select from: {choices_str}")
        return None
    
    @staticmethod
    def get_string_input(prompt: str, default: Optional[str] = None, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Optional[str]:
        """Get string input from user with validation"""
        if default:
            full_prompt = f"{prompt} (default: {default}): "
        else:
            full_prompt = f"{prompt}: "
        
        user_input = input(full_prompt).strip()
        
        if not user_input and default:
            return default
        
        if not user_input:
            return None
        
        if min_length is not None and len(user_input) < min_length:
            print(f"Input must be at least {min_length} characters long")
            return None
        
        if max_length is not None and len(user_input) > max_length:
            print(f"Input must be at most {max_length} characters long")
            return None
        
        return user_input
    
    @staticmethod
    def get_tags_input(prompt: str = "Enter tags (comma-separated, optional)") -> list:
        """Get comma-separated tags from user input"""
        tags_input = input(f"{prompt}: ").strip()
        if not tags_input:
            return []
        
        tags = [tag.strip() for tag in tags_input.split(",")]
        return [tag for tag in tags if tag]  # Remove empty tags
    
    @staticmethod
    def confirm_action(prompt: str) -> bool:
        """Confirm a potentially destructive action"""
        return InputUtils.get_yes_no_input(f"⚠️  {prompt}", default=False)