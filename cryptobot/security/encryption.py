"""
Encryption Utilities
=================
Provides encryption and decryption functions for sensitive data
such as API keys and secrets.
"""

import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


# Environment variable for master key
ENV_KEY_NAME = "CRYPTOBOT_MASTER_KEY"

# Default salt (in a real application, this should be stored securely)
DEFAULT_SALT = b"cryptobot_salt_value"


def generate_key(password: str, salt: Optional[bytes] = None) -> bytes:
    """
    Generate an encryption key from a password.
    
    Args:
        password: Password to derive key from
        salt: Salt value for key derivation
        
    Returns:
        bytes: Encryption key
    """
    if salt is None:
        salt = DEFAULT_SALT
        
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def get_encryption_key() -> bytes:
    """
    Get the encryption key from environment variable or generate one.
    
    Returns:
        bytes: Encryption key
    """
    # Check if master key exists in environment
    master_key = os.environ.get(ENV_KEY_NAME)
    
    if not master_key:
        # Use default key for development (in production, this should be set properly)
        logger.warning(f"Environment variable {ENV_KEY_NAME} not set, using default key")
        master_key = "default_development_key_do_not_use_in_production"
        
    # Generate key from master key
    return generate_key(master_key)


def encrypt_data(data: str) -> str:
    """
    Encrypt data using Fernet symmetric encryption.
    
    Args:
        data: Data to encrypt
        
    Returns:
        str: Encrypted data as base64 string
    """
    try:
        key = get_encryption_key()
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        return data  # Return original data on error (not secure)


def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypt data using Fernet symmetric encryption.
    
    Args:
        encrypted_data: Encrypted data as base64 string
        
    Returns:
        str: Decrypted data
    """
    try:
        key = get_encryption_key()
        cipher = Fernet(key)
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = cipher.decrypt(decoded_data)
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return encrypted_data  # Return encrypted data on error


def encrypt_api_keys(api_key: str) -> str:
    """
    Encrypt API key.
    
    Args:
        api_key: API key to encrypt
        
    Returns:
        str: Encrypted API key
    """
    return encrypt_data(api_key)


def decrypt_api_keys(encrypted_api_key: str) -> str:
    """
    Decrypt API key.
    
    Args:
        encrypted_api_key: Encrypted API key
        
    Returns:
        str: Decrypted API key
    """
    return decrypt_data(encrypted_api_key)


def test_encryption() -> bool:
    """
    Test encryption and decryption functionality.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        test_data = "test_api_key_12345"
        encrypted = encrypt_data(test_data)
        decrypted = decrypt_data(encrypted)
        
        if decrypted == test_data:
            logger.info("Encryption test successful")
            return True
        else:
            logger.error("Encryption test failed: decrypted data does not match original")
            return False
    except Exception as e:
        logger.error(f"Encryption test failed with error: {str(e)}")
        return False