"""Outbound email, with a log-only fallback.

The project has no mail infrastructure and is normally run locally, so requiring
a working SMTP server just to exercise the password-reset flow would make the
feature untestable on the machine it is developed on. When `smtp_host` is unset
the reset link is written to the application log instead of being sent.

That fallback is a development affordance, not a silent failure: `send_reset_email`
returns whether the message was actually transmitted, and the caller does NOT
change its HTTP response based on the answer. The user is always told the same
thing, because telling them "no email was sent" would reveal whether the address
is registered — the exact enumeration leak the login endpoint is careful to avoid.

The token itself is never returned through the API. It exists in exactly two
places: the recipient's inbox (or the server log), and as a hash in the database.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from ..config import settings

log = logging.getLogger("neuroguard.mailer")

SUBJECT = "Reset your NeuroGuard password"

_BODY = """\
Hello{name},

Someone asked to reset the password for the NeuroGuard account registered to
this address. Open the link below to choose a new one:

    {link}

This link expires in {minutes} minutes and can only be used once.

If you did not request this, you can ignore this email — your password has not
been changed, and nobody can sign in without the link above.
"""


def reset_link(token: str) -> str:
    """Build the frontend URL a reset token is redeemed at."""
    return f"{settings.frontend_base_url.rstrip('/')}/reset-password?token={token}"


def send_reset_email(email: str, token: str, full_name: str | None = None) -> bool:
    """Send (or log) a password-reset link. Returns True if actually sent.

    Never raises: a mail server being down must not turn into a 500 that tells
    the caller their address exists. Failures are logged and reported as False.
    """
    link = reset_link(token)
    body = _BODY.format(
        name=f" {full_name.split()[0]}" if full_name else "",
        link=link,
        minutes=settings.reset_token_minutes,
    )

    if not settings.smtp_host:
        # Deliberately logged at WARNING so it is visible in a default-configured
        # dev server without turning on debug logging.
        log.warning(
            "SMTP is not configured; password reset link for %s (valid %d min):\n    %s",
            email,
            settings.reset_token_minutes,
            link,
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = settings.smtp_from
    msg["To"] = email
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as srv:
            if settings.smtp_starttls:
                srv.starttls()
            if settings.smtp_user and settings.smtp_password:
                srv.login(settings.smtp_user, settings.smtp_password)
            srv.send_message(msg)
        log.info("Sent password reset email to %s", email)
        return True
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.error("Failed to send password reset email to %s: %s", email, exc)
        return False
