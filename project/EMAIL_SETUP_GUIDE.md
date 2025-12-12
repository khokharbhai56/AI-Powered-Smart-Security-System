# Email Alert Setup Guide

## Gmail Configuration (Recommended)

### Step 1: Enable 2-Factor Authentication
1. Go to [myaccount.google.com](https://myaccount.google.com)
2. Click **Security** in the left menu
3. Enable **2-Step Verification** if not already enabled

### Step 2: Generate App Password
1. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Select:
   - App: **Mail**
   - Device: **Windows Computer**
3. Google will generate a **16-character password**
4. Copy the password (remove spaces)

### Step 3: Update config.yaml
Edit `project/config.yaml`:

```yaml
alert_config:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  email_user: "your_email@gmail.com"
  email_password: "xxxx xxxx xxxx xxxx"  # Your 16-char app password (no spaces)
  alert_recipients:
    - "recipient1@gmail.com"
    - "recipient2@company.com"
```

⚠️ **Important**: Use the **App Password**, NOT your regular Gmail password!

---

## Other Email Providers

### Outlook/Hotmail
```yaml
smtp_server: "smtp.outlook.com"
smtp_port: 587
email_user: "your_email@outlook.com"
email_password: "your_password"
```

### Yahoo Mail
```yaml
smtp_server: "smtp.mail.yahoo.com"
smtp_port: 587
email_user: "your_email@yahoo.com"
email_password: "app_password"  # Yahoo also requires app password
```

### Custom SMTP (Corporate Email)
Contact your IT department for:
- SMTP Server address
- SMTP Port (usually 587 or 465)
- Email and password
- TLS/SSL requirements

---

## Testing the Configuration

### Option 1: Via Dashboard
1. Start the Streamlit dashboard
2. Navigate to **Alert Recipients** section
3. Click **Send Test Alert** button
4. Check your email inbox (and spam folder)

### Option 2: Via Command Line
```bash
cd project
python -c "
from real_time_system.detection_pipeline import DetectionPipeline
pipeline = DetectionPipeline()
pipeline._send_email_alert({
    'type': 'test',
    'message': 'Test alert from security system',
    'confidence': 0.95,
    'bbox': [0, 0, 100, 100]
})
"
```

---

## Troubleshooting

### Error: "Username and Password not accepted"
- ✅ **Solution**: Use App Password for Gmail, NOT your regular password
- Check that you've enabled 2FA on your Google Account
- Verify the 16-character password is correctly copied (no extra spaces)

### Error: "Connection timeout"
- ✅ **Solution**: Check SMTP server and port are correct
- Ensure firewall allows outbound connections on port 587
- Try using port 465 with SSL instead

### Emails going to spam
- ✅ **Solution**: 
  - Add your email to trusted senders
  - Check email headers and verify SMTP configuration
  - Consider using official corporate email server

### "SMTPAuthenticationError: Invalid credentials"
- ✅ **Solution**:
  - Verify email_user and email_password are correct
  - Ensure no leading/trailing spaces in config.yaml
  - For Gmail: MUST use App Password, not regular password

---

## Security Best Practices

### Do NOT:
❌ Use your main Gmail/Outlook password  
❌ Commit credentials to Git (keep config.yaml private)  
❌ Share email_password in messages or tickets  

### Do:
✅ Use App Passwords for better security  
✅ Keep config.yaml out of version control  
✅ Use different email for alerts than personal email  
✅ Rotate credentials periodically  

---

## Alert Configuration Options

```yaml
alert_config:
  # Alert Settings
  alert_cooldown: 30              # Minimum seconds between alerts
  audio_enabled: true             # Play alert sound
  email_enabled: true             # Send email alerts
  popup_enabled: true             # Show desktop popup
  log_to_db: true                 # Store in database
  
  # Email Settings
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  email_user: "your_email@gmail.com"
  email_password: "your_app_password"
  
  # Recipients
  alert_recipients:
    - "security@company.com"
    - "manager@company.com"
```

---

## Support

If you still have issues:
1. Check logs in `logs/security_system.log`
2. Enable `logging_config.level: "DEBUG"` in config.yaml
3. Verify SMTP credentials with email provider
4. Test with `telnet smtp.gmail.com 587`

---

**Last Updated**: December 11, 2025
