from __future__ import annotations

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


def main() -> None:
    load_dotenv()

    creds_path = Path(os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json"))
    token_path = Path("token.pickle")

    if not creds_path.exists():
        raise FileNotFoundError(f"Google OAuth client file not found: {creds_path}")

    creds = None
    if token_path.exists():
        with token_path.open("rb") as handle:
            creds = pickle.load(handle)

    if creds and creds.valid:
        print("Gmail is already connected.")
        print(f"Token file: {token_path.resolve()}")
        return

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        creds = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            open_browser=True,
            authorization_prompt_message="Opening your browser for Gmail sign-in. If it does not open, visit this URL: {url}",
            success_message="Gmail is connected. You can close this tab and return to Buraq.",
        )

    with token_path.open("wb") as handle:
        pickle.dump(creds, handle)

    print("Gmail authentication completed successfully.")
    print(f"Saved token to: {token_path.resolve()}")


if __name__ == "__main__":
    main()
