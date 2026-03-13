# App Onboarding Guide for OAuth2 Authentication

When a user asks the agent (running via `sorcar_agent.py`) to authenticate them with an OAuth2-based app (e.g. Google Calendar, Gmail, Spotify), follow these steps **in order, without trial and error**.

## Prerequisites

The agent has these tools available via `AppAgent` (which extends `SorcarAgent`):

- `check_auth()` — check if already authenticated
- `configure_oauth2(auth_url, token_url, client_id_env, client_secret_env, scopes)` — save OAuth2 config
- `authenticate_oauth2(code, redirect_uri)` — exchange auth code for tokens
- `authenticate(token)` — store a direct access token
- `clear_auth()` — remove stored auth
- `api_call(method, url, body, headers)` — make authenticated REST API calls
- `ask_user_browser_action(instruction, url)` — open a URL and let the user interact
- `ask_user_question(question)` — ask the user a question and wait for their response
- `Bash(command, description)` — run shell commands
- `go_to_url(url)` — navigate browser to a URL
- Browser automation tools: `click`, `type_text`, `press_key`, `scroll`, `get_page_content`, `screenshot`

## Step-by-Step OAuth2 Authentication Flow

### Step 1: Check Existing Authentication

```
check_auth()
```

If already authenticated, skip everything and proceed with the user's task.

### Step 2: Check if OAuth2 Config Exists

The agent should check if `oauth2.json` already exists for the app:

```python
Bash(command='cat ~/.kiss.artifacts/channels/<app_name>/oauth2.json 2>/dev/null || echo "NOT_FOUND"')
```

If config exists, skip to Step 4.

### Step 3: Configure OAuth2

Call `configure_oauth2()` with the correct endpoints for the app. Common configs:

**Google Calendar:**

```
configure_oauth2(
    auth_url='https://accounts.google.com/o/oauth2/v2/auth',
    token_url='https://oauth2.googleapis.com/token',
    client_id_env='GOOGLE_CLIENT_ID',
    client_secret_env='GOOGLE_CLIENT_SECRET',
    scopes='https://www.googleapis.com/auth/calendar'
)
```

**Gmail:**

```
configure_oauth2(
    auth_url='https://accounts.google.com/o/oauth2/v2/auth',
    token_url='https://oauth2.googleapis.com/token',
    client_id_env='GOOGLE_CLIENT_ID',
    client_secret_env='GOOGLE_CLIENT_SECRET',
    scopes='https://www.googleapis.com/auth/gmail.modify'
)
```

**Spotify:**

```
configure_oauth2(
    auth_url='https://accounts.spotify.com/authorize',
    token_url='https://accounts.spotify.com/api/token',
    client_id_env='SPOTIFY_CLIENT_ID',
    client_secret_env='SPOTIFY_CLIENT_SECRET',
    scopes='user-read-private user-read-email playlist-read-private'
)
```

### Step 4: Check Environment Variables for Client Credentials

```python
Bash(command='echo "CLIENT_ID=${<CLIENT_ID_ENV>:+SET}" && echo "CLIENT_SECRET=${<CLIENT_SECRET_ENV>:+SET}"')
```

If **NOT SET**, the agent must help the user create OAuth2 credentials. See [Creating OAuth2 Credentials](#creating-oauth2-credentials) below.

### Step 5: Start Local Callback Server

Start a lightweight HTTP server on port 8585 to capture the OAuth2 authorization code:

```python
Bash(command='''cat > /tmp/oauth_server.py << 'PYEOF'
import http.server, urllib.parse, os

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if 'code' in params:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Authorization successful!</h1><p>You can close this tab.</p>')
            with open('/tmp/oauth_code.txt', 'w') as f:
                f.write(params['code'][0])
            os._exit(0)
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error = params.get('error', ['unknown'])[0]
            self.wfile.write(f'<h1>Error: {error}</h1>'.encode())
            with open('/tmp/oauth_code.txt', 'w') as f:
                f.write(f'ERROR:{error}')
            os._exit(1)
    def log_message(self, format, *args): pass

http.server.HTTPServer(('127.0.0.1', 8585), Handler).serve_forever()
PYEOF
rm -f /tmp/oauth_code.txt
nohup python3 /tmp/oauth_server.py > /tmp/oauth_server.log 2>&1 &
echo "Server PID: $!"
sleep 0.5
cat /tmp/oauth_server.log''', description='Start local OAuth callback server on port 8585')
```

### Step 6: Build the Authorization URL

```python
Bash(command='''python3 -c "
import urllib.parse
params = {
    'client_id': '$(echo $<CLIENT_ID_ENV>)',
    'redirect_uri': 'http://127.0.0.1:8585/callback',
    'response_type': 'code',
    'scope': '<SCOPES>',
    'access_type': 'offline',
    'prompt': 'consent',
}
print('https://<AUTH_DOMAIN>/auth?' + urllib.parse.urlencode(params))
"''', description='Build OAuth2 authorization URL')
```

### Step 7: Open Browser for User Authorization

Use `ask_user_browser_action` to open the authorization URL and let the user sign in and grant access:

```
ask_user_browser_action(
    instruction='Please sign in with your account, grant the requested permissions, and wait for the "Authorization successful!" page to appear. Then click "I\'m Done".',
    url='<AUTHORIZATION_URL_FROM_STEP_6>'
)
```

### Step 8: Read the Authorization Code

After the user completes authorization, the callback server captures the code:

```python
Bash(command='cat /tmp/oauth_code.txt 2>/dev/null || echo "NO_CODE"')
```

### Step 9: Exchange Code for Tokens

Use `authenticate_oauth2()` to exchange the code:

```
authenticate_oauth2(code='<CODE_FROM_STEP_8>', redirect_uri='http://127.0.0.1:8585/callback')
```

**Important:** If `authenticate_oauth2()` fails (e.g. 400 Bad Request), fall back to manual token exchange via curl:

```python
Bash(command='''curl -s -X POST '<TOKEN_URL>' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d "code=<CODE>" \
  -d "client_id=$<CLIENT_ID_ENV>" \
  -d "client_secret=$<CLIENT_SECRET_ENV>" \
  -d "redirect_uri=http://127.0.0.1:8585/callback" \
  -d "grant_type=authorization_code"''')
```

Then save the token using `authenticate(token='<access_token_from_response>')`.

### Step 10: Verify and Clean Up

```
check_auth()
```

Clean up temp files:

```python
Bash(command='rm -f /tmp/oauth_server.py /tmp/oauth_code.txt /tmp/oauth_server.log; kill $(lsof -ti:8585) 2>/dev/null || true')
```

______________________________________________________________________

## Creating OAuth2 Credentials

When the client ID / client secret environment variables are not set, the agent must guide the user through creating them.

### Google (Calendar, Gmail, etc.)

1. **Navigate to Google Cloud Console:**

   ```
   go_to_url(url='https://console.cloud.google.com/apis/credentials')
   ```

1. **Set up OAuth Consent Screen** (if not already configured):

   - Navigate to: `https://console.cloud.google.com/auth/overview/create`
   - Step 1 (App Information): Set app name (e.g. "KISS Agent"), support email
   - Step 2 (Audience): Select "Internal" for organization accounts or "External" for personal
   - Step 3 (Contact Information): Add developer email
   - Step 4 (Finish): Check the "I agree to Google API Services: User Data Policy" checkbox and click Create
   - **Tip:** If the agreement checkbox is not clickable via accessibility tree, use `ask_user_browser_action` to have the user check it manually

1. **Enable the Required API:**

   - Navigate to the API page, e.g. `https://console.cloud.google.com/apis/library/calendar-json.googleapis.com`
   - Click "Enable"
   - Wait for it to finish enabling

1. **Create OAuth Client ID:**

   - Navigate to: `https://console.cloud.google.com/apis/credentials/oauthclient`
   - Select application type: **"Desktop app"** (simplest — no redirect URI configuration needed in Google Console)
   - Click "Create"
   - Copy the Client ID and Client Secret from the confirmation dialog

1. **Save credentials as environment variables:**

   ```python
   Bash(command='''echo 'export GOOGLE_CLIENT_ID="<client_id>"' >> ~/.zshrc
   echo 'export GOOGLE_CLIENT_SECRET="<client_secret>"' >> ~/.zshrc
   export GOOGLE_CLIENT_ID="<client_id>"
   export GOOGLE_CLIENT_SECRET="<client_secret>"
   echo "Credentials saved"''')
   ```

### Key Lessons from Past Runs

- **Use "Desktop app" type** for OAuth clients — it avoids the complexity of configuring authorized redirect URIs in Google Console, which is error-prone via browser automation (the "Add URI" buttons under different sections look identical in the accessibility tree).
- **The local callback server on port 8585 still works** with Desktop app type because Google allows `http://localhost` redirects for desktop apps.
- **Use `ask_user_browser_action`** for steps that are hard to automate (e.g. checking agreement checkboxes, signing in to Google).
- **If `authenticate_oauth2()` fails** with a 400 error, the code is likely still valid — use `curl` to manually exchange it and then call `authenticate(token=...)` with the access token.
- **Environment variables set via `Bash(command='export ...')` are NOT persisted** across Bash calls. Always append to `~/.zshrc` AND export in the same command, or read them inline.
- **Token persistence:** Once authenticated, tokens are saved to `~/.kiss.artifacts/channels/<app>/token.json` and will survive restarts. No need to re-authenticate unless the token expires and has no refresh token.

### Token-Based Auth (non-OAuth2 apps like GitHub)

For apps that use personal access tokens (PATs) instead of OAuth2:

1. `ask_user_question(question='Please provide your <App> personal access token')`
1. `authenticate(token='<user_provided_token>')`
1. `check_auth()`
