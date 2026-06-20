from __future__ import annotations
 
import json
import smtplib
import urllib.request
import urllib.error
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
 
from guardianeye.utils.config import (
    ALERT_WEBHOOK_URL,
    ALERT_EMAIL_TO, ALERT_EMAIL_FROM, ALERT_EMAIL_PASS,
    ALERT_SMTP_HOST, ALERT_SMTP_PORT,
    ALERT_TELEGRAM_TOKEN, ALERT_TELEGRAM_CHAT_ID,
)
from guardianeye.utils.logger import get_logger
 
log = get_logger(__name__)
 
 
class AlertDispatcher:
    """Sends emergency alerts via webhook and/or email."""
 
    def dispatch(self, zone_name: str) -> None:
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"[GuardianEye ALERT] {ts}\n"
            f"Inactivity detected!\n"
            f"Zone: {zone_name}\n"
            f"No response 60s after buzzer -> immediate check required."
        )
 
        log.critical("=== EMERGENCY ALERT === zone: %s", zone_name)
 
        sent = False
        if ALERT_WEBHOOK_URL:
            sent |= self._send_webhook(message)
        if ALERT_EMAIL_TO and ALERT_EMAIL_FROM:
            sent |= self._send_email(message, zone_name, ts)
        if ALERT_TELEGRAM_TOKEN and ALERT_TELEGRAM_CHAT_ID:
            sent |= self._send_telegram(message)
        if not sent:
            log.warning("No alert channel configured. Set ALERT_* in config.py. Message: %s", message)
    
    def _send_telegram(self, message: str) -> bool:
        url = f"https://api.telegram.org/bot{ALERT_TELEGRAM_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": ALERT_TELEGRAM_CHAT_ID,
            "text": message,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                log.info("Telegram sent (HTTP %d).", resp.getcode())
                return True
        except urllib.error.URLError as e:
            log.error("Telegram failed: %s", e)
            return False
 
    def _send_webhook(self, message: str) -> bool:
        payload = json.dumps({"text": message}).encode("utf-8")
        try:
            req = urllib.request.Request(
                ALERT_WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                log.info("Webhook sent (HTTP %d).", resp.getcode())
                return True
        except urllib.error.URLError as e:
            log.error("Webhook failed: %s", e)
            return False
 
    def _send_email(self, message: str, zone: str, ts: str) -> bool:
        subject = f"[GuardianEye] ALERT - {zone} ({ts})"
 
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO
 
        msg.attach(MIMEText(message, "plain", "utf-8"))
        msg.attach(MIMEText(f"""
        <html><body>
        <h2 style="color:red;">GuardianEye Emergency Alert</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
          <tr><td><b>Time</b></td><td>{ts}</td></tr>
          <tr><td><b>Zone</b></td><td style="color:red;">{zone}</td></tr>
          <tr><td><b>Status</b></td><td>No response 60s after buzzer</td></tr>
          <tr><td><b>Action</b></td><td><b>Immediate on-site check required</b></td></tr>
        </table>
        </body></html>
        """, "html", "utf-8"))
 
        try:
            with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASS)
                server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())
            log.info("Email sent to %s.", ALERT_EMAIL_TO)
            return True
        except smtplib.SMTPException as e:
            log.error("Email failed: %s", e)
            return False
            
