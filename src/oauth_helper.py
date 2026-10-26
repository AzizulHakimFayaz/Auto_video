"""Local OAuth 2.0 helper utility to obtain refresh/access tokens safely on your own PC.

Run this locally to generate your `YOUTUBE_REFRESH_TOKEN` via Google's official
OAuth 2.0 browser flow without ever sharing your account password.
"""

from __future__ import annotations

import argparse
import sys


def get_youtube_refresh_token(client_id: str, client_secret: str, port: int = 8080) -> str:
    """Launch a local browser OAuth 2.0 flow to obtain a YouTube Data API v3 refresh token."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError(
            "google-auth-oauthlib is required. Install with: pip install google-auth-oauthlib"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [f"http://localhost:{port}/"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=scopes)
    creds = flow.run_local_server(
        port=port, prompt="consent", access_type="offline"
    )
    if not creds.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh_token. Ensure prompt='consent' and access_type='offline'."
        )
    return str(creds.refresh_token)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local OAuth 2.0 Token Generator for Auto Story Video"
    )
    parser.add_argument(
        "--platform",
        choices=["youtube"],
        default="youtube",
        help="Platform to generate OAuth token for.",
    )
    parser.add_argument("--client-id", required=True, help="OAuth 2.0 Client ID")
    parser.add_argument("--client-secret", required=True, help="OAuth 2.0 Client Secret")
    parser.add_argument("--port", type=int, default=8080, help="Local callback port")
    args = parser.parse_args()

    if args.platform == "youtube":
        print("Opening browser for official Google/YouTube OAuth 2.0 authorization...")
        refresh_token = get_youtube_refresh_token(
            client_id=args.client_id.strip(),
            client_secret=args.client_secret.strip(),
            port=args.port,
        )
        print("\n=== SUCCESS! Copy these values into your .env or GitHub Actions Secrets ===")
        print(f"YOUTUBE_CLIENT_ID={args.client_id.strip()}")
        print(f"YOUTUBE_CLIENT_SECRET={args.client_secret.strip()}")
        print(f"YOUTUBE_REFRESH_TOKEN={refresh_token}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
