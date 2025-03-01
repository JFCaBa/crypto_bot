"""
Email Notifier
============
Sends email notifications for trading events.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Optional, Dict, Any

from loguru import logger


class EmailNotifier:
    """
    Sends email notifications for trading events.
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        sender: str = None,
        recipients: List[str] = None
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender: Sender email address
            recipients: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients or []
        
        logger.info(f"Email notifier initialized with server {smtp_server}:{smtp_port}")
        
    async def send(self, title: str, message: str, level: str = 'info') -> bool:
        """
        Send email notification.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level ('info', 'warning', 'error')
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            if not self.recipients:
                logger.warning("No recipients configured for email notification")
                return False
                
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            
            # Add level indicator to subject
            level_prefix = {
                'info': 'INFO',
                'warning': 'WARNING',
                'error': 'ERROR'
            }.get(level.lower(), 'INFO')
            
            msg['Subject'] = f"[{level_prefix}] {title}"
            
            # Add timestamp and format message
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            html_message = f"""
            <html>
                <body>
                    <h2>{title}</h2>
                    <p><strong>Time:</strong> {current_time}</p>
                    <p><strong>Level:</strong> {level_prefix}</p>
                    <pre>{message}</pre>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_message, 'html'))
            
            # Connect to SMTP server and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                    
                server.send_message(msg)
                
            logger.info(f"Email notification sent: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False
            
    def add_recipient(self, email: str) -> bool:
        """
        Add a recipient email address.
        
        Args:
            email: Email address
            
        Returns:
            bool: True if added, False if already exists
        """
        if email not in self.recipients:
            self.recipients.append(email)
            return True
        return False
        
    def remove_recipient(self, email: str) -> bool:
        """
        Remove a recipient email address.
        
        Args:
            email: Email address
            
        Returns:
            bool: True if removed, False if not found
        """
        if email in self.recipients:
            self.recipients.remove(email)
            return True
        return False