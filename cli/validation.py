# cli/validation.py
import re
import os
from typing import Optional, Union, List, Tuple, Any, Callable

class InputValidator:
    """Input validation utilities for CLI interface"""
    
    @staticmethod
    def validate_positive_integer(value: str, min_value: int = 1) -> Optional[int]:
        """Validate positive integer input"""
        try:
            num = int(value)
            if num >= min_value:
                return num
            return None
        except ValueError:
            return None
    
    @staticmethod
    def validate_float_range(value: str, min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
        """Validate float input within range"""
        try:
            num = float(value)
            if min_val <= num <= max_val:
                return num
            return None
        except ValueError:
            return None
    
    @staticmethod
    def validate_file_path(path: str) -> bool:
        """Validate file path exists and is readable"""
        try:
            expanded_path = os.path.expanduser(path)
            return os.path.isfile(expanded_path) and os.access(expanded_path, os.R_OK)
        except Exception:
            return False
    
    @staticmethod
    def validate_directory_path(path: str) -> bool:
        """Validate directory path exists and is readable"""
        try:
            expanded_path = os.path.expanduser(path)
            return os.path.isdir(expanded_path) and os.access(expanded_path, os.R_OK)
        except Exception:
            return False
    
    @staticmethod
    def validate_non_empty_string(value: str) -> bool:
        """Validate string is not empty after stripping"""
        return bool(value.strip())
    
    @staticmethod
    def parse_tags(tags_input: str) -> List[str]:
        """Parse comma-separated tags input"""
        if not tags_input.strip():
            return []
        
        tags = [tag.strip() for tag in tags_input.split(',')]
        return [tag for tag in tags if tag]  # Remove empty tags
    
    @staticmethod
    def validate_menu_choice(choice: str, max_option: int) -> Optional[int]:
        """Validate menu choice input"""
        try:
            num = int(choice)
            if 0 <= num <= max_option:
                return num
            return None
        except ValueError:
            return None
    
    @staticmethod
    def validate_yes_no(value: str, default: bool = False) -> bool:
        """Validate yes/no input with default"""
        value = value.strip().lower()
        if value in ('y', 'yes', 'true', '1'):
            return True
        elif value in ('n', 'no', 'false', '0'):
            return False
        else:
            return default
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem operations"""
        # Remove/replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        # Limit length
        return sanitized[:255]
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_port(port: str) -> Optional[int]:
        """Validate port number (1-65535)"""
        try:
            port_num = int(port)
            if 1 <= port_num <= 65535:
                return port_num
            return None
        except ValueError:
            return None
    
    @staticmethod
    def validate_hostname(hostname: str) -> bool:
        """Validate hostname or IP address"""
        # IP address pattern
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        
        # Hostname pattern
        hostname_pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        
        return bool(re.match(ip_pattern, hostname) or re.match(hostname_pattern, hostname))
    
    @staticmethod
    def validate_database_name(name: str) -> bool:
        """Validate database name (alphanumeric, underscore, hyphen)"""
        if not name or len(name) > 63:
            return False
        pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
        return bool(re.match(pattern, name))
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: float = 100) -> bool:
        """Validate file size is within limits"""
        try:
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False
    
    @staticmethod
    def validate_json_string(json_str: str) -> bool:
        """Validate JSON string format"""
        try:
            import json
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_regex_pattern(pattern: str) -> bool:
        """Validate regex pattern syntax"""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False
    
    @classmethod
    def get_validated_input(
        cls,
        prompt: str,
        validator: Callable[[str], bool],
        error_message: str = "Invalid input. Please try again.",
        max_attempts: int = 3,
        allow_empty: bool = False
    ) -> Optional[str]:
        """Get validated input with retry logic"""
        for attempt in range(max_attempts):
            try:
                value = input(prompt).strip()
                
                if not value and allow_empty:
                    return value
                
                if validator(value):
                    return value
                else:
                    print(error_message)
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"Input error: {e}")
        
        print(f"Max attempts ({max_attempts}) reached.")
        return None
    
    @classmethod
    def get_validated_integer(
        cls,
        prompt: str,
        min_value: int = 1,
        max_value: Optional[int] = None,
        default: Optional[int] = None,
        max_attempts: int = 3
    ) -> Optional[int]:
        """Get validated integer input with range checking"""
        def validator(value: str) -> bool:
            if not value and default is not None:
                return True
            try:
                num = int(value)
                if num < min_value:
                    return False
                if max_value is not None and num > max_value:
                    return False
                return True
            except ValueError:
                return False
        
        range_info = f"({min_value}-{max_value or 'unlimited'})"
        full_prompt = f"{prompt} {range_info}: "
        if default is not None:
            full_prompt = f"{prompt} {range_info} [default: {default}]: "
        
        error_msg = f"Please enter an integer between {min_value} and {max_value or 'unlimited'}."
        
        result = cls.get_validated_input(
            full_prompt, validator, error_msg, max_attempts, allow_empty=(default is not None)
        )
        
        if result is None:
            return None
        elif result == "" and default is not None:
            return default
        else:
            return int(result)
    
    @classmethod
    def get_validated_float(
        cls,
        prompt: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default: Optional[float] = None,
        max_attempts: int = 3
    ) -> Optional[float]:
        """Get validated float input with range checking"""
        def validator(value: str) -> bool:
            if not value and default is not None:
                return True
            try:
                num = float(value)
                return min_value <= num <= max_value
            except ValueError:
                return False
        
        full_prompt = f"{prompt} ({min_value}-{max_value}): "
        if default is not None:
            full_prompt = f"{prompt} ({min_value}-{max_value}) [default: {default}]: "
        
        error_msg = f"Please enter a number between {min_value} and {max_value}."
        
        result = cls.get_validated_input(
            full_prompt, validator, error_msg, max_attempts, allow_empty=(default is not None)
        )
        
        if result is None:
            return None
        elif result == "" and default is not None:
            return default
        else:
            return float(result)
    
    @classmethod
    def get_yes_no_input(
        cls,
        prompt: str,
        default: Optional[bool] = None,
        max_attempts: int = 3
    ) -> Optional[bool]:
        """Get yes/no input with validation"""
        def validator(value: str) -> bool:
            if not value and default is not None:
                return True
            value = value.lower()
            return value in ('y', 'yes', 'n', 'no', 'true', 'false', '1', '0')
        
        suffix = " (y/n): "
        if default is not None:
            default_text = "Y" if default else "N"
            suffix = f" (y/n) [default: {default_text}]: "
        
        full_prompt = prompt + suffix
        error_msg = "Please enter 'y' for yes or 'n' for no."
        
        result = cls.get_validated_input(
            full_prompt, validator, error_msg, max_attempts, allow_empty=(default is not None)
        )
        
        if result is None:
            return None
        elif result == "" and default is not None:
            return default
        else:
            return cls.validate_yes_no(result)
    
    @classmethod
    def get_choice_input(
        cls,
        prompt: str,
        choices: List[str],
        default: Optional[int] = None,
        max_attempts: int = 3
    ) -> Optional[int]:
        """Get choice input with validation"""
        # Display choices
        print(f"\n{prompt}")
        for i, choice in enumerate(choices):
            marker = " (default)" if default == i else ""
            print(f"{i + 1}. {choice}{marker}")
        
        def validator(value: str) -> bool:
            if not value and default is not None:
                return True
            try:
                num = int(value)
                return 1 <= num <= len(choices)
            except ValueError:
                return False
        
        choice_prompt = f"Select option (1-{len(choices)}): "
        if default is not None:
            choice_prompt = f"Select option (1-{len(choices)}) [default: {default + 1}]: "
        
        error_msg = f"Please enter a number between 1 and {len(choices)}."
        
        result = cls.get_validated_input(
            choice_prompt, validator, error_msg, max_attempts, allow_empty=(default is not None)
        )
        
        if result is None:
            return None
        elif result == "" and default is not None:
            return default
        else:
            return int(result) - 1  # Convert to 0-based index
    
    @classmethod
    def get_file_path_input(
        cls,
        prompt: str,
        must_exist: bool = True,
        max_attempts: int = 3
    ) -> Optional[str]:
        """Get file path input with validation"""
        def validator(path: str) -> bool:
            if not must_exist:
                return bool(path.strip())
            return cls.validate_file_path(path)
        
        error_msg = "File not found or not readable." if must_exist else "Please enter a valid path."
        
        return cls.get_validated_input(prompt, validator, error_msg, max_attempts)
    
    @classmethod
    def get_directory_path_input(
        cls,
        prompt: str,
        must_exist: bool = True,
        max_attempts: int = 3
    ) -> Optional[str]:
        """Get directory path input with validation"""
        def validator(path: str) -> bool:
            if not must_exist:
                return bool(path.strip())
            return cls.validate_directory_path(path)
        
        error_msg = "Directory not found or not accessible." if must_exist else "Please enter a valid path."
        
        return cls.get_validated_input(prompt, validator, error_msg, max_attempts)
    
    @classmethod
    def confirm_action(cls, action: str, default: bool = False) -> bool:
        """Get confirmation for potentially destructive actions"""
        prompt = f"Are you sure you want to {action}?"
        result = cls.get_yes_no_input(prompt, default=default)
        return result if result is not None else False