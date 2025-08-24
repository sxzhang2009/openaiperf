"""
Email notification service for OpenAIPerf
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # Email configuration from environment variables or default values
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "notifications@openaiperf.org")
        
        # Enable/disable email notifications
        self.email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
        
    def send_notification_email(self, to_email: str, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """
        Send a notification email
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Plain text email body
            html_body: Optional HTML email body
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not self.email_enabled:
            logger.info(f"Email notifications disabled. Would send: {subject} to {to_email}")
            return True
            
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured. Cannot send email.")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email
            
            # Add plain text part
            text_part = MIMEText(body, "plain")
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                msg.attach(html_part)
            
            # Connect to server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def send_new_submission_notification(self, admin_email: str, submission_id: str, 
                                       submitter_username: str, model_name: str, 
                                       admin_url: str) -> bool:
        """
        Send notification to admin about new submission
        
        Args:
            admin_email: Admin email address
            submission_id: ID of the new submission
            submitter_username: Username of the person who submitted
            model_name: Name of the model submitted
            admin_url: URL to the admin panel
            
        Returns:
            bool: True if email was sent successfully
        """
        subject = f"New Submission: {submission_id}"
        
        # Plain text version
        body = f"""
A new submission has been received on OpenAIPerf and is waiting for approval.

Submission Details:
- ID: {submission_id}
- Submitted by: {submitter_username}
- Model: {model_name or 'Not specified'}

Please review and approve the submission in the admin panel:
{admin_url}

Best regards,
OpenAIPerf System
"""
        
        # HTML version
        html_body = f"""
<html>
<body>
<h2>New Submission Received</h2>
<p>A new submission has been received on OpenAIPerf and is waiting for approval.</p>

<h3>Submission Details:</h3>
<ul>
    <li><strong>ID:</strong> {submission_id}</li>
    <li><strong>Submitted by:</strong> {submitter_username}</li>
    <li><strong>Model:</strong> {model_name or 'Not specified'}</li>
</ul>

<p>
    <a href="{admin_url}" style="background-color: #4f46e5; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Review in Admin Panel
    </a>
</p>

<p>Best regards,<br>OpenAIPerf System</p>
</body>
</html>
"""
        
        return self.send_notification_email(admin_email, subject, body, html_body)
    
    def send_registration_request_notification(self, admin_email: str, username: str, 
                                             email: str, organization: str, admin_url: str) -> bool:
        """
        Send notification to admin about new registration request
        
        Args:
            admin_email: Admin email address
            username: Username of the person requesting registration
            email: Email of the person requesting registration
            organization: Organization of the requester
            admin_url: URL to the admin panel
            
        Returns:
            bool: True if email was sent successfully
        """
        subject = f"New Registration Request: {username}"
        
        # Plain text version
        body = f"""
A new user registration request has been received on OpenAIPerf and is waiting for approval.

Registration Details:
- Username: {username}
- Email: {email}
- Organization: {organization or 'Not specified'}

Please review and approve the registration in the admin panel:
{admin_url}

Best regards,
OpenAIPerf System
"""
        
        # HTML version
        html_body = f"""
<html>
<body>
<h2>New Registration Request</h2>
<p>A new user registration request has been received on OpenAIPerf and is waiting for approval.</p>

<h3>Registration Details:</h3>
<ul>
    <li><strong>Username:</strong> {username}</li>
    <li><strong>Email:</strong> {email}</li>
    <li><strong>Organization:</strong> {organization or 'Not specified'}</li>
</ul>

<p>
    <a href="{admin_url}" style="background-color: #4f46e5; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Review in Admin Panel
    </a>
</p>

<p>Best regards,<br>OpenAIPerf System</p>
</body>
</html>
"""
        
        return self.send_notification_email(admin_email, subject, body, html_body)
    
    def send_registration_approval_notification(self, user_email: str, username: str, login_url: str) -> bool:
        """
        Send notification to user about registration approval
        
        Args:
            user_email: User's email address
            username: Username of the approved user
            login_url: URL to login page
            
        Returns:
            bool: True if email was sent successfully
        """
        subject = "Registration Approved - Welcome to OpenAIPerf!"
        
        # Plain text version
        body = f"""
Congratulations! Your registration request for OpenAIPerf has been approved.

Account Details:
- Username: {username}
- Email: {user_email}

You can now log in to your account and start using OpenAIPerf:
{login_url}

Welcome to the OpenAIPerf community!

Best regards,
OpenAIPerf Team
"""
        
        # HTML version
        html_body = f"""
<html>
<body>
<h2>🎉 Registration Approved!</h2>
<p>Congratulations! Your registration request for OpenAIPerf has been approved.</p>

<h3>Account Details:</h3>
<ul>
    <li><strong>Username:</strong> {username}</li>
    <li><strong>Email:</strong> {user_email}</li>
</ul>

<p>You can now log in to your account and start using OpenAIPerf:</p>

<p>
    <a href="{login_url}" style="background-color: #059669; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Login to Your Account
    </a>
</p>

<p>Welcome to the OpenAIPerf community!</p>

<p>Best regards,<br>OpenAIPerf Team</p>
</body>
</html>
"""
        
        return self.send_notification_email(user_email, subject, body, html_body)


# Global email service instance
email_service = EmailService()
