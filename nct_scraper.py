import json
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 1. Fetch secure credentials from GitHub Secrets
SENDER_EMAIL = os.environ.get("BOT_EMAIL")
SENDER_PASSWORD = os.environ.get("BOT_PASSWORD")
# Filter out any accidental empty spaces in the mailing list
SUBSCRIBERS = [email.strip() for email in os.environ.get("MAILING_LIST", "").split(",") if email.strip()]

def send_daily_digest():
    if not SENDER_EMAIL or not SENDER_PASSWORD or not SUBSCRIBERS:
        print("Missing credentials or mailing list. Skipping email alerts.")
        return

    # 2. Load today's data
    try:
        with open("nct_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("No database found. Skipping emails.")
        return

    # 3. LIVE MODE: Only grab events that the scraper actively flagged as brand new today
    new_events = [e for e in data if e.get("is_new") is True]

    if len(new_events) == 0:
        print("No new trial updates today. No email sent.")
        return

    # 4. Build a clean, professional HTML email
    print(f"Drafting email for {len(new_events)} new events...")
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #005f73; border-bottom: 2px solid #e9ecef; padding-bottom: 10px;">
            Morning Biotech Digest
        </h2>
        <p>Your automated scraper found <strong>{len(new_events)}</strong> new clinical trial updates overnight:</p>
        <ul style="list-style-type: none; padding: 0;">
    """

    for e in new_events:
        sentiment_color = "#6c757d" # Neutral
        if e['sentiment'] == "Positive": sentiment_color = "#2b9348"
        if e['sentiment'] == "Negative": sentiment_color = "#d90429"

        html_content += f"""
        <li style="background: #f8f9fa; margin-bottom: 15px; padding: 15px; border-radius: 6px; border-left: 4px solid {sentiment_color};">
            <strong>{e['ticker']} | {e['drug_name']}</strong><br>
            <span style="font-size: 12px; color: #555;">NCT ID: {e['nct_id']} | Indication: {e['indication']}</span><br><br>
            <em>"{e['notes']}"</em><br><br>
            <a href="{e['source']}" style="color: #0a9396; text-decoration: none; font-weight: bold; font-size: 13px;">View Source Article &rarr;</a>
        </li>
        """

    html_content += """
        </ul>
        <p style="text-align: center; margin-top: 30px;">
            <a href="https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPO_NAME/" style="background: #005f73; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold;">
                Open Full Dashboard
            </a>
        </p>
    </body>
    </html>
    """

    # 5. Connect to Gmail and Blast the Emails!
    msg = MIMEMultipart()
    msg['Subject'] = f"Biotech Tracker: {len(new_events)} New Trial Updates"
    msg['From'] = f"Biotech Automated Tracker <{SENDER_EMAIL}>"
    msg.attach(MIMEText(html_content, 'html'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # Send emails individually so colleagues don't see each other's addresses (BCC style)
        for recipient in SUBSCRIBERS:
            if 'To' in msg:
                del msg['To']
            msg['To'] = recipient
            server.send_message(msg)
            
        server.quit()
        print(f"Success! Alerts sent to {len(SUBSCRIBERS)} subscribers.")
    except Exception as e:
        print(f"Critical Error sending emails: {e}")

if __name__ == "__main__":
    send_daily_digest()
