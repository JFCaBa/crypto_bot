"""
Telegram Notifier
===============
Sends notifications to Telegram channels or chats.
"""

import aiohttp
from datetime import datetime
from typing import List, Optional, Dict, Any

from loguru import logger


class TelegramNotifier:
    """
    Sends notifications to Telegram channels or chats.
    """
    
    def __init__(
        self,
        token: str,
        chat_ids: List[str] = None
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            token: Telegram bot token
            chat_ids: List of chat IDs to send messages to
        """
        self.token = token
        self.chat_ids = chat_ids or []
        self.base_url = f"https://api.telegram.org/bot{token}"
        
        logger.info("Telegram notifier initialized")
        
    async def send(self, title: str, message: str, level: str = 'info') -> bool:
        """
        Send Telegram notification.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level ('info', 'warning', 'error')
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            if not self.chat_ids:
                logger.warning("No chat IDs configured for Telegram notification")
                return False
                
            # Format message with emoji based on level
            level_emoji = {
                'info': '📊',
                'warning': '⚠️',
                'error': '🚨'
            }.get(level.lower(), '📊')
            
            # Format timestamp
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Construct message
            formatted_message = f"{level_emoji} *{title}*\n\n" \
                               f"📅 *Time:* {current_time}\n" \
                               f"📝 *Message:*\n{message}"
            
            # Send to all chat IDs
            success = True
            for chat_id in self.chat_ids:
                result = await self._send_message(chat_id, formatted_message)
                success = success and result
                
            return success
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {str(e)}")
            return False
            
    async def _send_message(self, chat_id: str, text: str) -> bool:
        """
        Send message to a specific chat.
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            url = f"{self.base_url}/sendMessage"
            params = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Telegram API error: {response.status} - {await response.text()}")
                        return False
                        
                    data = await response.json()
                    return data.get('ok', False)
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
            
    async def send_photo(self, chat_id: str, photo_path: str, caption: str = None) -> bool:
        """
        Send photo to a specific chat.
        
        Args:
            chat_id: Telegram chat ID
            photo_path: Path to photo file
            caption: Optional caption
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            url = f"{self.base_url}/sendPhoto"
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('chat_id', chat_id)
            
            with open(photo_path, 'rb') as photo_file:
                data.add_field('photo', photo_file)
                
            if caption:
                data.add_field('caption', caption)
                data.add_field('parse_mode', 'Markdown')
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        logger.error(f"Telegram API error: {response.status} - {await response.text()}")
                        return False
                        
                    data = await response.json()
                    return data.get('ok', False)
                    
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {str(e)}")
            return False
            
    def add_chat_id(self, chat_id: str) -> bool:
        """
        Add a chat ID to the recipient list.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            bool: True if added, False if already exists
        """
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
            return True
        return False
        
    def remove_chat_id(self, chat_id: str) -> bool:
        """
        Remove a chat ID from the recipient list.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            bool: True if removed, False if not found
        """
        if chat_id in self.chat_ids:
            self.chat_ids.remove(chat_id)
            return True
        return False