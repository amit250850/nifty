"""
login.py — Zerodha Kite Connect authentication helper.

HOW TO USE:
    1. Run this script once each trading day before starting main.py.
       python login.py
    2. It opens the Kite login URL in your browser.
    3. After you log in, Kite redirects to a URL containing ?request_token=xxxxx
    4. Paste the FULL redirect URL (or just the request_token) when prompted.
    5. The script exchanges the token, fetches a fresh access_token, and
       writes it back to the .env file so main.py can pick it up automatically.

NOTE:  Kite access tokens expire at midnight IST every day.
       You must run login.py at the start of each trading session.
"""

import os
import re
import sys
import webbrowser

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

API_KEY    = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")
ENV_FILE   = os.path.join(os.path.dirname(__file__), ".env")


def get_login_url(kite: KiteConnect) -> str:
    """Return the Kite login URL for the user to authenticate."""
    return kite.login_url()


def extract_request_token(user_input: str) -> str:
    """
    Extract request_token from either:
      • The full redirect URL  (e.g. https://127.0.0.1?request_token=abc&action=login)
      • Or just the raw token string itself.

    Args:
        user_input: Raw string pasted by the user.

    Returns:
        The request_token string.

    Raises:
        ValueError: If no token could be found.
    """
    user_input = user_input.strip()
    # Try to extract from URL query string
    match = re.search(r"request_token=([A-Za-z0-9]+)", user_input)
    if match:
        return match.group(1)
    # Assume raw token if no URL pattern found
    if re.fullmatch(r"[A-Za-z0-9]+", user_input):
        return user_input
    raise ValueError(
        f"Could not extract request_token from input: {user_input!r}"
    )


def generate_access_token(kite: KiteConnect, request_token: str) -> str:
    """
    Exchange a request_token for a session access_token.

    Args:
        kite:          Initialised KiteConnect instance.
        request_token: One-time token from Kite login redirect.

    Returns:
        The access_token string.
    """
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    return data["access_token"]


def save_access_token(token: str) -> None:
    """
    Persist the access_token into the .env file so other modules can load it.

    Args:
        token: The Kite access_token to save.
    """
    set_key(ENV_FILE, "KITE_ACCESS_TOKEN", token)
    print(f"[login.py] ✅  access_token saved to {ENV_FILE}")


def main() -> None:
    """Interactive Kite login flow."""
    if not API_KEY or not API_SECRET:
        print(
            "[login.py] ❌  KITE_API_KEY or KITE_API_SECRET not found in .env. "
            "Please populate the .env file first."
        )
        sys.exit(1)

    kite = KiteConnect(api_key=API_KEY)

    # Step 1: Show & open login URL
    login_url = get_login_url(kite)
    print("\n" + "=" * 60)
    print("KITE CONNECT — DAILY LOGIN")
    print("=" * 60)
    print(f"\n📌  Login URL:\n{login_url}\n")
    print("Opening in your browser …")
    webbrowser.open(login_url)

    # Step 2: Accept redirect URL / request_token from user
    print(
        "\nAfter logging in, Kite will redirect to a URL like:\n"
        "  https://127.0.0.1?request_token=XXXXX&action=login&status=success\n"
    )
    user_input = input("Paste the redirect URL or just the request_token here: ").strip()

    try:
        request_token = extract_request_token(user_input)
    except ValueError as exc:
        print(f"[login.py] ❌  {exc}")
        sys.exit(1)

    print(f"[login.py] 🔑  request_token = {request_token}")

    # Step 3: Generate access token
    try:
        access_token = generate_access_token(kite, request_token)
    except Exception as exc:
        print(f"[login.py] ❌  Failed to generate session: {exc}")
        sys.exit(1)

    print(f"[login.py] 🎟️   access_token  = {access_token[:8]}…(hidden)")

    # Step 4: Save to .env
    save_access_token(access_token)

    print("\n✅  Login complete. You can now run:  python main.py\n")


if __name__ == "__main__":
    main()
