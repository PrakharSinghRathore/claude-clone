"""
Email adapter for the Hermes Gateway.

Supports IMAP/SMTP integration, HTML email formatting, attachment support,
and auto-reply rules.

Usage::

    from hermes.gateway.config import PlatformConfig
    from hermes.gateway.platforms.email_platform import EmailAdapter

    config = PlatformConfig(
        name="email",
        api_url="imap.gmail.com:993",
        token="email@gmail.com",
        api_key="app-specific-password",
        enabled=True,
        extra={
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "email@gmail.com",
            "smtp_password": "app-specific-password",
            "imap_host": "imap.gmail.com",
            "imap_port": 993,
            "imap_user": "email@gmail.com",
            "imap_password": "app-specific-password",
        },
    )
    adapter = EmailAdapter(config)
    await adapter.connect()
"""

from __future__ import annotations

import asyncio
import email
import email.mime.text
import email.mime.multipart
import email.mime.base
import email.mime.application
import email.utils
import json
import logging
import os
import smtplib
import uuid
from base64 import b64encode
from datetime import datetime, timezone
from email.header import decode_header
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.gateway.runner import IncomingMessage

logger = logging.getLogger("hermes.gateway.platforms.email")

try:
    import aioimaplib
    HAS_AIOIMAP = True
except ImportError:
    HAS_AIOIMAP = False


class EmailAdapter:
    """
    Email adapter using IMAP for receiving and SMTP for sending.

    Parameters
    ----------
    config:
        Platform configuration. Expected ``extra`` keys:
        - ``smtp_host``, ``smtp_port``, ``smtp_user``, ``smtp_password``
        - ``imap_host``, ``imap_port``, ``imap_user``, ``imap_password``
    """

    def __init__(self, config: Any):
        self._config = config
        self._timeout = config.timeout or 30
        self._connected = False
        self._extra = config.extra or {}

        # SMTP settings
        self._smtp_host = self._extra.get("smtp_host", os.environ.get("SMTP_HOST", ""))
        self._smtp_port = int(self._extra.get("smtp_port", os.environ.get("SMTP_PORT", "587")))
        self._smtp_user = self._extra.get("smtp_user", os.environ.get("SMTP_USER", ""))
        self._smtp_password = self._extra.get("smtp_password", os.environ.get("SMTP_PASSWORD", ""))

        # IMAP settings
        self._imap_host = self._extra.get("imap_host", os.environ.get("IMAP_HOST", ""))
        self._imap_port = int(self._extra.get("imap_port", os.environ.get("IMAP_PORT", "993")))
        self._imap_user = self._extra.get("imap_user", os.environ.get("IMAP_USER", ""))
        self._imap_password = self._extra.get("imap_password", os.environ.get("IMAP_PASSWORD", ""))

        self._from_address = self._smtp_user or self._imap_user or config.token or ""
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._poll_task: Optional[asyncio.Task] = None
        self._seen_uids: set = set()
        self._poll_interval = 30

    # ── Common Interface ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to SMTP and IMAP servers."""
        # Verify SMTP
        try:
            await self._send_smtp_test()
            logger.info("Email SMTP connected: %s:%d", self._smtp_host, self._smtp_port)
        except Exception as e:
            logger.error("Email SMTP connection failed: %s", e)

        # Start IMAP polling
        if self._imap_host:
            self._poll_task = asyncio.create_task(self._imap_poll_loop())
            logger.info("Email IMAP polling started: %s:%d", self._imap_host, self._imap_port)
        else:
            logger.warning("No IMAP host configured — email receiving disabled")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from email servers."""
        self._connected = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def is_connected(self) -> bool:
        return self._connected

    async def send_message(
        self, chat_id: str, text: str, **kwargs: Any,
    ) -> Optional[str]:
        """
        Send an email message.

        ``chat_id`` is the recipient email address.
        """
        subject = kwargs.get("subject", "Hermes Gateway")
        html = kwargs.get("html", False)

        return await self._send_email(
            to_addr=chat_id,
            subject=subject,
            body=text,
            html=html,
            reply_to_message_id=kwargs.get("reply_to"),
        )

    async def send_file(
        self, chat_id: str, file_path: str, **kwargs: Any,
    ) -> Optional[str]:
        """Send an email with a file attachment."""
        if not os.path.exists(file_path):
            return None

        return await self._send_email_with_attachment(
            to_addr=chat_id,
            subject=kwargs.get("subject", "Hermes Gateway"),
            body=kwargs.get("body", "Please see the attached file."),
            file_path=file_path,
        )

    async def get_updates(self) -> List[IncomingMessage]:
        """Poll for new email messages."""
        messages: List[IncomingMessage] = []
        while not self._message_queue.empty():
            try:
                msg = self._message_queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    # ── Extended Interface ────────────────────────────────────────────────

    async def send_typing(self, chat_id: str) -> None:
        """Not applicable for email."""
        pass

    def format_html_email(
        self,
        text: str,
        title: str = "Message",
        footer: str = "",
    ) -> str:
        """Format a plain text message as HTML email."""
        paragraphs = text.split("\n\n")
        body_paragraphs = "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 680px; margin: 0 auto; padding: 20px; color: #333;">
<h2 style="color: #1a1a1a;">{title}</h2>
{body_paragraphs}
{'<hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;"><small style="color: #999;">' + footer + '</small>' if footer else ''}
</body>
</html>"""

    # ── SMTP ──────────────────────────────────────────────────────────────

    async def _send_smtp_test(self) -> None:
        """Test SMTP connection."""
        loop = asyncio.get_running_loop()

        def _test():
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._timeout) as server:
                server.starttls()
                server.login(self._smtp_user, self._smtp_password)

        await loop.run_in_executor(None, _test)

    async def _send_email(
        self,
        to_addr: str,
        subject: str,
        body: str,
        html: bool = False,
        reply_to_message_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
    ) -> Optional[str]:
        """Send an email via SMTP."""
        msg = email.mime.multipart.MIMEMultipart("alternative")

        msg["From"] = self._from_address
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["Message-ID"] = email.utils.make_msgid()

        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        # Plain text version
        part_text = email.mime.text.MIMEText(body, "plain", "utf-8")
        msg.attach(part_text)

        # HTML version
        if html:
            html_body = self.format_html_email(body, title=subject)
            part_html = email.mime.text.MIMEText(html_body, "html", "utf-8")
            msg.attach(part_html)

        loop = asyncio.get_running_loop()

        def _send():
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._timeout) as server:
                server.starttls()
                server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._from_address, to_addr, msg.as_string())

        try:
            await loop.run_in_executor(None, _send)
            return msg["Message-ID"]
        except Exception as e:
            logger.error("Email send failed: %s", e)
            return None

    async def _send_email_with_attachment(
        self,
        to_addr: str,
        subject: str,
        body: str,
        file_path: str,
    ) -> Optional[str]:
        """Send an email with attachment."""
        msg = email.mime.multipart.MIMEMultipart("mixed")
        msg["From"] = self._from_address
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["Message-ID"] = email.utils.make_msgid()

        # Body
        body_part = email.mime.text.MIMEText(body, "plain", "utf-8")
        msg.attach(body_part)

        # Attachment
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_data = f.read()

        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".txt": "text/plain",
            ".csv": "text/csv",
        }
        maintype, subtype = "application", "octet-stream"
        for ext_key, mime_type in mime_map.items():
            if ext == ext_key:
                maintype, subtype = mime_type.split("/", 1)
                break

        att = email.mime.base.MIMEBase(maintype, subtype)
        att.set_payload(file_data)
        email.encoders.encode_base64(att)
        att.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(att)

        loop = asyncio.get_running_loop()

        def _send():
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self._timeout) as server:
                server.starttls()
                server.login(self._smtp_user, self._smtp_password)
                server.sendmail(self._from_address, to_addr, msg.as_string())

        try:
            await loop.run_in_executor(None, _send)
            return msg["Message-ID"]
        except Exception as e:
            logger.error("Email with attachment failed: %s", e)
            return None

    # ── IMAP Polling ──────────────────────────────────────────────────────

    async def _imap_poll_loop(self) -> None:
        """Poll IMAP for new messages."""
        if not HAS_AIOIMAP:
            logger.warning("aioimaplib not installed — IMAP polling disabled")
            return

        while self._connected:
            try:
                await self._check_imap()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("IMAP poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_imap(self) -> None:
        """Check for new messages via IMAP."""
        client = aioimaplib.IMAP4_SSL(
            host=self._imap_host, port=self._imap_port,
        )
        await client.wait_hello_from_server()
        await client.login(self._imap_user, self._imap_password)
        await client.select("INBOX")

        # Search for unseen messages
        status, messages = await client.search("UNSEEN")
        if status != "OK" or not messages[0]:
            await client.logout()
            return

        uid_list = messages[0].split()
        for uid in uid_list:
            uid_str = uid.decode() if isinstance(uid, bytes) else uid
            if uid_str in self._seen_uids:
                continue

            self._seen_uids.add(uid_str)
            status, msg_data = await client.fetch(uid_str, "(RFC822)")
            if status == "OK":
                raw_email = msg_data[1][1]
                if isinstance(raw_email, bytes):
                    msg = self._parse_email(raw_email)
                    if msg:
                        await self._message_queue.put(msg)

        await client.logout()

    def _parse_email(self, raw_data: bytes) -> Optional[IncomingMessage]:
        """Parse a raw email into an IncomingMessage."""
        msg = email.message_from_bytes(raw_data)

        from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
        subject = self._decode_header_value(msg.get("Subject", ""))
        message_id = msg.get("Message-ID", "")

        # Extract text body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")

        text = f"Subject: {subject}\n\n{body}"

        return IncomingMessage(
            platform="email",
            chat_id=from_addr,
            user_id=from_addr,
            text=text[:5000],
            message_id=message_id,
            metadata={
                "subject": subject,
                "from": from_addr,
                "to": msg.get("To", ""),
                "has_attachments": any(
                    part.get_content_disposition() == "attachment"
                    for part in msg.walk()
                ),
            },
        )

    @staticmethod
    def _decode_header_value(value: str) -> str:
        """Decode an email header value."""
        if not value:
            return ""
        try:
            decoded = decode_header(value)
            parts = []
            for part, charset in decoded:
                if isinstance(part, bytes):
                    parts.append(part.decode(charset or "utf-8", errors="replace"))
                else:
                    parts.append(part)
            return " ".join(parts)
        except Exception:
            return value
