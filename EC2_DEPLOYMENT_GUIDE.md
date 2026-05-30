# Deploying the Price Pulse Dashboard on EC2 (Amazon Linux 2023)

A step-by-step guide to host your Streamlit dashboard — including the **OpenAI**
AI features (insight summaries, review pros/cons, dataset chat) — on a free-tier
`t3.micro` instance. Assumes the instance is launched, running, and you have
your `.pem` key file and the instance's **public IPv4 address**.

> Reminder: this instance has **no GPU**. Fine-tune your model elsewhere
> (Colab or your laptop), then bring the finished model + results here.

> Order of operations: do the prerequisites (steps **A → C**) first, then the
> on-instance setup (steps **1 → 9**). The AI smoke test is step **6b**.

---

## 0. Before you start — checklist

- [ ] Instance is **running** (EC2 console shows green "Running")
- [ ] You have your key file (e.g. `ec2pricepulse.pem`)
- [ ] Security group allows inbound **port 22** (SSH) and **port 8501** (Streamlit)
- [ ] You know the instance's **public DNS / IP** (from the EC2 **Connect** screen)
- [ ] **OpenAI API key** created and a spend limit set (see step A)
- [ ] EC2 **IAM role** has S3 read access (see step C)
- [ ] **S3 bucket** created and both parquet files uploaded (see step D)

---

## A. Get your OpenAI API key (do this first)

The AI features call the **OpenAI API** (`gpt-4o-mini`). You need one API key.

1. Go to <https://platform.openai.com/api-keys> → **Create new secret key**.
2. Copy the key (starts with `sk-…`) somewhere safe — you can't view it again.
   **Never commit it or paste it into code** — it goes in an environment
   variable only (step 6b).
3. **Set a spend limit** so a mistake can't drain your account: **Settings →
   Billing → Limits** → set a low monthly hard cap (e.g. **$5**). With prepaid
   credit you're already bounded to what you've loaded, but the cap is a good
   safety net.

**Cost:** `gpt-4o-mini` is ~$0.15 / $0.60 per million input/output tokens. A
demo's worth of summaries and chat costs a few cents.

---

## C. Give the EC2 instance permission to read S3

Attach an **IAM role** so the instance can download your parquet files without
storing AWS keys. (OpenAI needs no IAM — it uses the API key from step A.)

1. AWS Console → **IAM** → **Roles** → **Create role** → trusted entity **EC2**.
2. Attach the AWS-managed policy **`AmazonS3ReadOnlyAccess`**.
   (Tighter: a custom policy allowing only `s3:GetObject` on your bucket.)
3. Name it e.g. `price-pulse-ec2-role` and create it.
4. EC2 Console → select your instance → **Actions** → **Security** →
   **Modify IAM role** → choose `price-pulse-ec2-role` → **Update**.

No restart needed — the role is available to the instance immediately.

> Prefer not to use an IAM role? You can also `scp` the parquet files straight
> to the instance (step 4, Option B) and skip S3 entirely.

---

## D. Create the S3 bucket and upload your results

Do this from your **laptop** (where the results file is). You need the AWS CLI
installed locally and configured once with `aws configure` (paste your access
keys from IAM → Security credentials).

1. **Create the bucket** (bucket names are globally unique — pick your own and
   use it everywhere below):

   ```bash
   aws s3 mb s3://pricepulse-data --region us-east-1
   ```

   > Use the **same region** as your EC2 instance where possible — it's simplest
   > and avoids cross-region transfer quirks. (OpenAI is a public API and doesn't
   > care about your AWS region.)

2. **Upload your processed results — BOTH parquet files** (the app needs each
   one: `results.parquet` drives the dashboard aggregates, `products.parquet`
   drives the Product drill-down page):

   ```bash
   aws s3 cp results.parquet s3://pricepulse-data/results.parquet
   aws s3 cp products.parquet s3://pricepulse-data/products.parquet
   ```

3. **(Optional) upload the raw dataset too**, if you want everything in one place:

   ```bash
   aws s3 cp data/amazon.csv s3://pricepulse-data/raw/amazon.csv
   ```

4. **Verify they're there:**

   ```bash
   aws s3 ls s3://pricepulse-data/
   ```

**Cost note:** at this size (a few MB) S3 is effectively free — well within the
5 GB free tier, and pennies even beyond it. Keep the bucket **private** (the
default); the EC2 IAM role from step C is what grants read access, so you never
need to make it public.

> **Re-upload whenever results change:** every time you re-run the ABSA pipeline
> on your laptop/Colab and get a new `results.parquet`, repeat step D.2 to push
> the new version up. The instance picks it up next time it downloads (step 6).

---

## 1. Fix key file permissions (local machine)

SSH refuses to use a key file that others can read. Run this once, on your own
computer, in the folder where the `.pem` lives:

```bash
chmod 400 ec2pricepulse.pem
```

> **On Windows (PowerShell)** there is no `chmod`. Use `icacls` instead, run
> from the folder containing the key:
>
> ```powershell
> icacls.exe ec2pricepulse.pem /reset
> icacls.exe ec2pricepulse.pem /grant:r "$($env:USERNAME):(R)"
> icacls.exe ec2pricepulse.pem /inheritance:r
> ```
>
> If the key sits in a OneDrive folder and SSH still complains, copy it to a
> plain local folder (e.g. `C:\Users\<you>\.ssh\`) and run `icacls` there.

---

## 2. Connect to the instance

```bash
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>
```

Replace `<PUBLIC_IP>` with your address. Type `yes` when asked to trust the host.
The username is **`ec2-user`** for Amazon Linux 2023 (it was on your launch screen).

---

## 3. Update the system and install Python tools

Amazon Linux 2023 ships with Python 3.9+. Update packages and install `pip`
and `git`:

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-pip git
```

Verify:

```bash
python3 --version
pip3 --version
```

---

## 4. Get your project onto the instance

Pick **one** of these.

### Option A — clone from GitHub (recommended)

```bash
git clone https://github.com/<your-username>/price-pulse.git
cd price-pulse
```

### Option B — copy from your laptop with scp

Run this **on your laptop**, not on the instance:

```bash
scp -i ec2pricepulse.pem -r ./price-pulse ec2-user@<PUBLIC_IP>:~/
```

Then back on the instance: `cd price-pulse`

---

## 5. Set up a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, at minimum:

```bash
pip install streamlit pandas pyarrow openai
```

(`pyarrow` lets pandas read Parquet; `openai` is the client for the AI
features. S3 downloads use the AWS CLI + the instance IAM role, so no AWS
SDK is needed here.)

---

## 6. Bring your results onto the instance (from S3)

Run this **on the instance** (in your SSH/VS Code terminal where the prompt
shows `ec2-user@ip-...`), not on your laptop. Download **both** parquet files
into the **project root** (alongside `app.py`) — that's exactly where the app
looks for them by default (`utils/helpers.py`), so no env vars are needed:

```bash
cd ~/price-pulse
aws s3 cp s3://pricepulse-data/results.parquet ./results.parquet
aws s3 cp s3://pricepulse-data/products.parquet ./products.parquet
```

This works with no credentials because the **IAM role from step C** grants the
instance S3 read access. Confirm they arrived:

```bash
ls -lh results.parquet products.parquet
```

> Replace `pricepulse-data` with whatever bucket name you actually created in
> step D. If you get `AccessDenied`, the IAM role is missing/not attached
> (revisit step C). If you get `NoSuchBucket`, check the name and region.

> **Alternative — read straight from S3 (no download):** the app also accepts
> `RESULTS_S3_URI` / `PRODUCTS_S3_URI` env vars pointing at `s3://...` paths.
> That route needs `s3fs` installed (`pip install s3fs`). Downloading to disk as
> above is simpler and is what this guide assumes.

**Whenever you update results:** re-upload both files from your laptop (step
D.2), then re-run these `aws s3 cp` commands on the instance to pull the new
versions.

---

## 6b. Set the OpenAI API key + smoke test

The app reads your key from the **`OPENAI_API_KEY`** environment variable. So
that it's present for the tmux session *and* the optional systemd service, put
it in `~/.bashrc` (paste your real `sk-…` key from step A):

```bash
echo 'export OPENAI_API_KEY="sk-your-real-key-here"' >> ~/.bashrc
source ~/.bashrc
```

> The key lives only in this environment variable on the instance — never in
> the repo. If you ever rotate it, edit `~/.bashrc` and `source` it again.
> Optional: `export OPENAI_MODEL="gpt-4o-mini"` to pin the model explicitly.

Now confirm the instance can actually reach OpenAI. With the venv active:

```bash
python3 - <<'EOF'
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from the environment
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=5,
    messages=[{"role": "user", "content": "Reply with the single word: OK"}],
)
print(resp.choices[0].message.content)
EOF
```

- If it prints `OK` — the key works and the AI features will light up.
- `AuthenticationError` → the key is missing/invalid (re-check step A and that
  `source ~/.bashrc` ran).
- `RateLimitError` / insufficient_quota → out of credit or over your spend cap
  (top up or raise the limit in the OpenAI billing page).

---

## 7. Test that it runs

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The `--server.address 0.0.0.0` part is **essential** — it tells Streamlit to
accept connections from outside the instance. Without it, only the instance
itself can reach the app.

> **Important:** Streamlit will print `URL: http://0.0.0.0:8501`. **Do NOT open
> that in your browser** — `0.0.0.0` is a "listen on all interfaces" address,
> not a real destination. It will fail with `ERR_ADDRESS_INVALID`. Use your
> instance's **public DNS or IP** instead (next).

Now open a browser on your own machine — use the **public DNS** from the EC2
Connect screen:

```
http://ec2-XX-XX-XX-XX.compute-1.amazonaws.com:8501
```

(or `http://<PUBLIC_IP>:8501`). Note it's **`http://`** not `https://`, and the
**`:8501`** suffix is required.

If the page loads — success. If it spins forever or won't connect, your security
group is almost certainly missing the **port 8501** inbound rule (see
troubleshooting). The public DNS/IP also **changes on every stop/start** — grab
the current one from the Connect screen each session.

Press `Ctrl+C` to stop it for now.

---

## 8. Keep the app running after you disconnect

If you just run `streamlit run`, the app dies the moment you close SSH. Use a
`tmux` session so it survives.

```bash
sudo dnf install -y tmux
tmux new -s dashboard
```

Inside the tmux session, start the app:

```bash
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Now **detach** without stopping it: press `Ctrl+B`, then `D`.

You can safely close SSH — the dashboard keeps running.

To come back later:

```bash
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>
tmux attach -t dashboard
```

To stop the app: attach, then `Ctrl+C`.

---

## 9. When you're done for the day — STOP the instance

In the EC2 console: select the instance → **Instance state** → **Stop**.

- **Stop** = compute billing pauses, disk + setup preserved. Use this daily.
- **Terminate** = deletes everything. Only when the project is fully finished.

> Note: a stopped instance gets a **new public IP** when you start it again.
> If you want a permanent address, allocate an **Elastic IP** (free while
> attached to a running instance). For a student demo, just grab the new IP
> each time — simpler.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser hangs / can't connect on :8501 | Security group missing port 8501 | Add inbound rule: Custom TCP, port 8501, source `0.0.0.0/0` (or your IP) |
| `Permission denied (publickey)` on SSH | Wrong key, wrong user, or key perms | Use `ec2-user`, check `chmod 400`, confirm right `.pem` |
| App works in SSH but dies after logout | Not using tmux | Run inside tmux (step 8) |
| `streamlit: command not found` | venv not activated | `source venv/bin/activate` |
| S3 `AccessDenied` | No IAM role / wrong creds | Attach S3-read IAM role to instance (step C) |
| Page loads only on `localhost` | Missing `--server.address 0.0.0.0` | Add that flag |
| Browser shows `ERR_ADDRESS_INVALID` on `0.0.0.0:8501` | Tried to open the literal `0.0.0.0` URL Streamlit printed | Use the public DNS/IP instead, e.g. `http://<public-dns>:8501` |
| AI features show "needs an OpenAI API key" caption | `OPENAI_API_KEY` not set in this shell | `export` it in `~/.bashrc`, `source ~/.bashrc`, restart the app (step 6b) |
| OpenAI `AuthenticationError` | Key missing/invalid/rotated | Re-check the key in `~/.bashrc`; create a fresh one if needed (step A) |
| OpenAI `RateLimitError` / insufficient_quota | Out of credit or over spend cap | Top up credit / raise the limit in OpenAI billing |
| OpenAI `NotFoundError` for the model | `OPENAI_MODEL` set to a model you can't access | Use `gpt-4o-mini` (or unset `OPENAI_MODEL`) |

---

## Quick reference — daily workflow

```bash
# Connect
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>

# Resume the dashboard
tmux attach -t dashboard
# ...do stuff, then Ctrl+B then D to detach

# Done for the day: Stop the instance in the EC2 console
```

> **Cost hygiene:** stopping the EC2 instance pauses *compute* charges. OpenAI
> only bills when the app actually calls it (button click / chat submit), so a
> stopped instance makes zero OpenAI calls. Keep the spend limit from step A
> active as your safety net throughout the project.

---

## Optional: auto-start on reboot (nice-to-have)

If you want the dashboard to launch automatically whenever the instance boots,
create a small systemd service. Skip this for a basic prototype — tmux is
enough — but here it is if you want it:

```bash
sudo tee /etc/systemd/system/pricepulse.service > /dev/null <<'EOF'
[Unit]
Description=Price Pulse Streamlit Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/price-pulse
Environment=OPENAI_API_KEY=sk-your-real-key-here
ExecStart=/home/ec2-user/price-pulse/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now pricepulse
```

Check status with `sudo systemctl status pricepulse`.

> systemd does **not** read `~/.bashrc`, so the `Environment=OPENAI_API_KEY=…`
> line in the unit is how the service gets the key (the `~/.bashrc` export from
> step 6b only covers tmux/interactive runs). Because the key sits in this file,
> keep it readable only by root: `sudo chmod 600 /etc/systemd/system/pricepulse.service`.
