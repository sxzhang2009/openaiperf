# Email Configuration for OpenAIPerf
# Set these environment variables to enable email notifications

# Enable/disable email notifications (true/false)
export EMAIL_ENABLED=true

# SMTP Server Configuration
export SMTP_SERVER=mail.privateemail.com
export SMTP_PORT=587
export SMTP_USERNAME=admin@openaiperf.org
export SMTP_PASSWORD=zhang2008!

# From email address
export FROM_EMAIL=admin@openaiperf.org

# Example configuration for PrivateEmail (default):
# 1. Use your @openaiperf.org email address as SMTP_USERNAME
# 2. Use your email password as SMTP_PASSWORD
# 3. Set EMAIL_ENABLED=true to activate notifications

# Example configuration for other providers:
# Gmail: smtp.gmail.com:587 (requires app password)
# Outlook: smtp-mail.outlook.com:587
# Yahoo: smtp.mail.yahoo.com:587
# Custom SMTP: your-smtp-server.com:587

# To use these settings, either:
# 1. Source this file: source email_config_example.txt
# 2. Or add them to your system environment variables
# 3. Or create a .env file (requires python-dotenv package)
